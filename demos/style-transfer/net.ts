import {Scalar, Array1D, Array3D, Array4D, CheckpointLoader, GPGPUContext, NDArray, NDArrayMathCPU, NDArrayMathGPU} from '../deeplearn';

import * as imagenet_util from './imagenet_util';

const IMAGE_SIZE = 227;
const GOOGLE_CLOUD_STORAGE_DIR =
//    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';
    'http://localhost:8080/demos/imagenet/';

export class TransformNet {
  private variables: {[varName: string]: NDArray};

  private preprocessInputShader: WebGLShader;

  constructor(private gpgpu: GPGPUContext, private math: NDArrayMathGPU) {
    this.preprocessInputShader =
        imagenet_util.getUnpackAndPreprocessInputShader(
            gpgpu, [IMAGE_SIZE, IMAGE_SIZE]);
  }

  /**
   * Loads necessary variables for SqueezeNet. Resolves the promise when the
   * variables have all been loaded.
   */
  loadVariables(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const checkpointLoader =
          new CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + 'squeezenet1_1/');
      checkpointLoader.getAllVariables().then(variables => {
        this.variables = variables;
        resolve();
      });
    });
  }

  /**
   * Preprocess an RGB color texture before inferring through squeezenet.
   * @param rgbTexture The RGB color texture to process into an Array3D.
   * @param imageDimensions The 2D dimensions of the image.
   */
  preprocessColorTextureToArray3D(rgbTexture: WebGLTexture, imageDimensions: [
    number, number
  ]): Array3D {
    const preprocessResultShapeRC: [number, number] =
        [imageDimensions[0], imageDimensions[0] * 3];

    const preprocessResultTexture =
        this.math.getTextureManager().acquireTexture(preprocessResultShapeRC);

    imagenet_util.preprocessInput(
        this.gpgpu, this.preprocessInputShader, rgbTexture,
        preprocessResultTexture, preprocessResultShapeRC);
    return NDArray.make<Array3D>([imageDimensions[0], imageDimensions[0], 3], {
      texture: preprocessResultTexture,
      textureShapeRC: preprocessResultShapeRC
    });
  }

  /**
   * Infer through TransformNet, assumes variables have been loaded. This does
   * standard ImageNet pre-processing before inferring through the model. This
   * method returns named activations as well as pre-softmax logits. The user
   * needs to clean up namedActivations after inferring.
   *
   * @param preprocessedInput preprocessed input Array.
   * @return Named activations and the pre-softmax logits.
   */
  infer(preprocessedInput: Array3D):
      {namedActivations: {[activationName: string]: Array3D}, logits: Array1D} {
    const namedActivations: {[key: string]: Array3D} = {};

    const avgpool10 = this.math.scope((keep) => {
      const conv1 = this.math.conv2d(
          preprocessedInput, this.variables['conv1_W:0'] as Array4D,
          this.variables['conv1_b:0'] as Array1D, 2, 0);
      const conv1relu = keep(this.math.relu(conv1));
      namedActivations['conv_1'] = conv1relu;

      const pool1 = keep(this.math.maxPool(conv1relu, 3, 2, 0));
      namedActivations['maxpool_1'] = pool1;

      const fire2 = keep(this.fireModule(pool1, 2));
      namedActivations['fire2'] = fire2;

      const fire3 = keep(this.fireModule(fire2, 3));
      namedActivations['fire3'] = fire3;

      // Because we don't have uneven padding yet, manually pad the ndarray on
      // the right.
      const fire3Reshape2d =
          fire3.as2D(fire3.shape[0], fire3.shape[1] * fire3.shape[2]);
      const fire3Sliced2d = this.math.slice2D(
          fire3Reshape2d, [0, 0],
          [fire3.shape[0] - 1, (fire3.shape[1] - 1) * fire3.shape[2]]);
      const fire3Sliced = fire3Sliced2d.as3D(
          fire3.shape[0] - 1, fire3.shape[1] - 1, fire3.shape[2]);
      const pool2 = keep(this.math.maxPool(fire3Sliced, 3, 2, 0));
      namedActivations['maxpool_2'] = pool2;

      const fire4 = keep(this.fireModule(pool2, 4));
      namedActivations['fire4'] = fire4;

      const fire5 = keep(this.fireModule(fire4, 5));
      namedActivations['fire5'] = fire5;

      const pool3 = keep(this.math.maxPool(fire5, 3, 2, 0));
      namedActivations['maxpool_3'] = pool3;

      const fire6 = keep(this.fireModule(pool3, 6));
      namedActivations['fire6'] = fire6;

      const fire7 = keep(this.fireModule(fire6, 7));
      namedActivations['fire7'] = fire7;

      const fire8 = keep(this.fireModule(fire7, 8));
      namedActivations['fire8'] = fire8;

      const fire9 = keep(this.fireModule(fire8, 9));
      namedActivations['fire9'] = fire9;

      const conv10 = keep(this.math.conv2d(
          fire9, this.variables['conv10_W:0'] as Array4D,
          this.variables['conv10_b:0'] as Array1D, 1, 0));
      namedActivations['conv10'] = conv10;

      return this.math.avgPool(conv10, conv10.shape[0], 1, 0).as1D();
    });

    return {namedActivations, logits: avgpool10};
  }

  private convLayer(input: Array3D, strides: number, 
    relu: boolean, varId: number) {
    var y = this.math.conv2d(input, 
      this.variables[this.varName(varId)] as Array4D, 
      null, [strides, strides], 'same');

    // y = this.instanceNorm(y, varId + 1);

    if (relu) {
      y = this.math.relu(y);
    }

    return y;
  }

  private instanceNorm(input: Array3D, varId: number) {
    const [height, width, inDepth] = input.shape;
    const [mu, sigma_sq]: [Array3D, Array3D] = this.instanceMoments(input);
    const shift = this.variables[this.varName(varId)] as Array1D;
    const scale = this.variables[this.varName(varId + 1)] as Array1D;
    const epsilon = Scalar.new(1e-3);
    const normalized = this.math.divideStrict(this.math.subStrict(input, mu), 
      this.math.sqrt(this.math.add(sigma_sq, epsilon)));
    const shifted = this.math.add(this.math.multiply(scale, normalized), shift);
    return shifted;
  }

  /**
   * Copies behavior of tf.nn.moments but purely for instance normalization.
   * Equivalent to tf.nn.moments(net, [0, 1], keep_dims=True) for a tensor 
   * of shape ()
   *
   * @param input Array3D shape [height, width, inDepth]
   * @return mean and variance per channel. Same shape as input.
   */
  private instanceMoments(input: Array3D): [Array3D, Array3D] {
    const [height, width, inDepth] = input.shape;
    const hWProduct = height * width;

    // Create explicit MathCPU for unimplemented GPU operations
    const mathCPU = new NDArrayMathCPU;

    // Switch dims for easier slicing. Operation is now on CPU
    const switched = mathCPU.switchDim(input, [2, 0, 1]);
    const switchedValues = switched.getValues();

    // Calculate mean and variance per channel
    const means = [];
    const variances = [];
    for (let i = 0; i < inDepth; i ++) {
      const curr = switchedValues.slice(i*hWProduct, (i+1)*hWProduct);

      var sum = 0;
      for (let j = 0; j < curr.length; j++) {
        sum += curr[j];
      }
      const avg = sum / curr.length;
      means.push(avg);

      var diffSum = 0;
      for (let j = 0; j < curr.length; j++) {
        diffSum += (avg - curr[j]) * (avg - curr[j]);
      }
      variances.push(diffSum / curr.length);
    }

    // "Broadcast" means and variances back to original shape
    var keepDimMeans: number[] = [];
    var keepDimVariances: number[] = [];
    for (let i = 0; i < hWProduct; i ++) {
      keepDimMeans = keepDimMeans.concat(means);
      keepDimVariances = keepDimVariances.concat(variances);
    }

    const meansArray = Array3D.new(input.shape, keepDimMeans);
    const variancesArray = Array3D.new(input.shape, keepDimVariances);

    return [meansArray, variancesArray];
  }

  private fireModule(input: Array3D, fireId: number) {
    const y1 = this.math.conv2d(
        input, this.variables['fire' + fireId + '/squeeze1x1_W:0'] as Array4D,
        this.variables['fire' + fireId + '/squeeze1x1_b:0'] as Array1D, 1, 0);
    const y2 = this.math.relu(y1);
    const left1 = this.math.conv2d(
        y2, this.variables['fire' + fireId + '/expand1x1_W:0'] as Array4D,
        this.variables['fire' + fireId + '/expand1x1_b:0'] as Array1D, 1, 0);
    const left2 = this.math.relu(left1);

    const right1 = this.math.conv2d(
        y2, this.variables['fire' + fireId + '/expand3x3_W:0'] as Array4D,
        this.variables['fire' + fireId + '/expand3x3_b:0'] as Array1D, 1, 1);
    const right2 = this.math.relu(right1);

    return this.math.concat3D(left2, right2, 2);
  }

  private varName(varId: number): string {
    if (varId === 0) {
      return 'Variable';
    }
    else {
      return 'Variable_' + varId;
    }
  }
}
