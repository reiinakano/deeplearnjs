import {Scalar, Array1D, Array3D, Array4D, CheckpointLoader, GPGPUContext, NDArray, NDArrayMathCPU, NDArrayMathGPU} from '../deeplearn';

import * as imagenet_util from './imagenet_util';

const GOOGLE_CLOUD_STORAGE_DIR =
//    'https://storage.googleapis.com/learnjs-data/checkpoint_zoo/';
    'http://127.0.0.1:8080/demos/style-transfer/ckpts/';

export class TransformNet {
  private variables: {[varName: string]: NDArray};

  private preprocessInputShader: WebGLShader;

  constructor(private gpgpu: GPGPUContext, 
    private math: NDArrayMathGPU, private style: string) {}

  /**
   * Loads necessary variables for SqueezeNet. Resolves the promise when the
   * variables have all been loaded.
   */
  loadVariables(): Promise<void> {
    return new Promise<void>((resolve, reject) => {
      const checkpointLoader =
          new CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + this.style + '/');
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
    const preprocessInputShader =
        imagenet_util.getUnpackAndPreprocessInputShader(
            this.gpgpu, [imageDimensions[0], imageDimensions[1]]);

    const preprocessResultShapeRC: [number, number] =
        [imageDimensions[0], imageDimensions[1] * 3];

    const preprocessResultTexture =
        this.math.getTextureManager().acquireTexture(preprocessResultShapeRC);

    imagenet_util.preprocessInput(
        this.gpgpu, preprocessInputShader, rgbTexture,
        preprocessResultTexture, preprocessResultShapeRC);
    return NDArray.make<Array3D>([imageDimensions[0], imageDimensions[1], 3], {
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
   * @return Array3D containing pixels of output img
   */
  infer(preprocessedInput: Array3D): Array3D {

    const img = this.math.scope((keep) => {
      console.log('conv1');
      const conv1 = this.convLayer(preprocessedInput, 1, true, 0);
      console.log('conv2');
      const conv2 = this.convLayer(conv1, 2, true, 3);
      console.log('conv3');
      const conv3 = this.convLayer(conv2, 2, true, 6);
      console.log('resid1');
      const resid1 = this.residualBlock(conv3, 9);
      console.log('resid2');
      const resid2 = this.residualBlock(resid1, 15);
      console.log('resid3');
      const resid3 = this.residualBlock(resid2, 21);
      console.log('resid4');
      const resid4 = this.residualBlock(resid3, 27);
      console.log('resid5');
      const resid5 = this.residualBlock(resid4, 33);
      console.log('conv_t1');
      const conv_t1 = this.convTransposeLayer(resid5, 64, 2, 39);
      console.log('conv_t2');
      const conv_t2 = this.convTransposeLayer(conv_t1, 32, 2, 42);
      console.log('conv_t3');
      const conv_t3 = this.convLayer(conv_t2, 1, false, 45);
      console.log('out');
      const out_tanh = this.math.tanh(conv_t3);
      const scaled = this.math.scalarTimesArray(Scalar.new(150), out_tanh);
      const shifted = this.math.scalarPlusArray(Scalar.new(255./2), scaled);

      return shifted;
    });

    return img;
  }

  private convLayer(input: Array3D, strides: number, 
    relu: boolean, varId: number): Array3D {
    console.log('convLayer.conv2d' + varId);
    const y = this.math.conv2d(input, 
      this.variables[this.varName(varId)] as Array4D, 
      null, [strides, strides], 'same');

    console.log('convLayer.instanceNorm' + (varId + 1));
    const y2 = this.instanceNorm(y, varId + 1);

    if (relu) {
      return this.math.relu(y2);
    }

    return y2;
  }

  private convTransposeLayer(input: Array3D, numFilters: number,
    strides: number, varId: number): Array3D {
    const [height, width, inDepth]: [number, number, number] = input.shape;
    const newRows = height * strides;
    const newCols = width * strides;
    const newShape: [number, number, number] = [newRows, newCols, numFilters];

    const y = this.math.conv2dTranspose(input,
      this.variables[this.varName(varId)] as Array4D,
      newShape, [strides, strides], 'same');

    const y2 = this.instanceNorm(y, varId + 1);

    const y3 = this.math.relu(y2);

    return y3;
  }

  private residualBlock(input: Array3D, varId: number): Array3D {
    const conv1 = this.convLayer(input, 1, true, varId);
    const conv2 = this.convLayer(conv1, 1, false, varId + 3);
    return this.math.addStrict(conv2, input); 
  }

  private instanceNorm(input: Array3D, varId: number): Array3D {
    const [height, width, inDepth]: [number, number, number] = input.shape;
    const [mu, sigma_sq]: [Array3D, Array3D] = this.instanceMoments(input);
    const shift = this.variables[this.varName(varId)] as Array1D;
    const scale = this.variables[this.varName(varId + 1)] as Array1D;
    const epsilon = Scalar.new(1e-3);
    const normalized = this.math.divideStrict(this.math.subStrict(input, mu), 
      this.math.sqrt(this.math.add(sigma_sq, epsilon)));
    const shifted = this.math.add(this.math.multiply(scale, normalized), shift);
    return shifted.as3D(height, width, inDepth);
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
    console.log('instanceMoments: switching dims');
    const switched = mathCPU.switchDim(input, [2, 0, 1]);
    console.log('instanceMoments: switched dims');
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
    console.log('instanceMoments: calculated means and variances');

    // "Broadcast" means and variances back to original shape
    var toConcatDimMeans: number[][] = [];
    var toConcatDimVariances: number[][] = [];
    for (let i = 0; i < hWProduct; i ++) {
      toConcatDimMeans.push(means);
      toConcatDimVariances.push(variances);
    }
    const keepDimMeans = [].concat.apply([], toConcatDimMeans);
    const keepDimVariances = [].concat.apply([], toConcatDimVariances);
    console.log('instanceMoments: "Broadcasted" to orig shape');

    const meansArray = Array3D.new(input.shape, keepDimMeans);
    const variancesArray = Array3D.new(input.shape, keepDimVariances);

    return [meansArray, variancesArray];
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
