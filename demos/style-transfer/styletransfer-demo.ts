import '../demo-header';
import '../demo-footer';
// tslint:disable-next-line:max-line-length
import {Array3D, gpgpu_util, GPGPUContext, NDArrayMathCPU, NDArrayMathGPU} from '../deeplearn';
// import * as imagenet_util from '../models/imagenet_util';
import {TransformNet} from './net';
import {PolymerElement, PolymerHTMLElement} from '../polymer-spec';

// tslint:disable-next-line:variable-name
export const StyleTransferDemoPolymer: new () => PolymerHTMLElement =
    PolymerElement({
      is: 'styletransfer-demo',
      properties: {
        contentNames: Array,
        selectedContentName: String,
        styleNames: Array,
        selectedStyleName: String,
        totalSteps: Number,
        secsPerStep: Number,
        contentLoss: Number,
        styleLoss: Number,
        applicationState: Number
      }
    });

export enum ApplicationState {
  IDLE = 1,
  TRAINING = 2
}

const CONTENT_NAMES = ['stata', 'face'];
const STYLE_NAMES = ['udnie', 'scream'];

export class StyleTransferDemo extends StyleTransferDemoPolymer {
  // DeeplearnJS stuff
  private math: NDArrayMathGPU;
  private mathCPU: NDArrayMathCPU;
  private gl: WebGLRenderingContext;
  private gpgpu: GPGPUContext;

  private transformNet: TransformNet;

  // DOM Elements
  private contentImgElement: HTMLImageElement;
  private styleImgElement: HTMLImageElement;
  private currentImgElement: HTMLImageElement;

  private trainButton: HTMLButtonElement;
  private stopButton: HTMLButtonElement;

  // Polymer properties
  private contentNames: string[];
  private selectedContentName: string;
  private styleNames: string[];
  private selectedStyleName: string;

  private totalSteps: number;
  private secsPerStep: number;
  private contentLoss: number;
  private styleLoss: number;

  private applicationState: ApplicationState;

  ready() {
    // Initialize DeeplearnJS stuff
    this.math = new NDArrayMathGPU;
    this.mathCPU = new NDArrayMathCPU;
    this.gl = gpgpu_util.createWebGLContext(this.inferenceCanvas);
    this.gpgpu = new GPGPUContext(this.gl);

    // Initialize polymer properties
    this.applicationState = ApplicationState.IDLE;
    this.totalSteps = 0;
    this.secsPerStep = 0;
    this.contentLoss = 0;
    this.styleLoss = 0;

    // Retrieve DOM for images
    this.contentImgElement =
        this.querySelector('#contentImg') as HTMLImageElement;
    this.styleImgElement = 
        this.querySelector('#styleImg') as HTMLImageElement;
    this.currentImgElement = 
        this.querySelector('#currentImg') as HTMLImageElement;

    // Render DOM for images
    this.contentNames = CONTENT_NAMES;
    this.selectedContentName = 'stata';
    this.contentImgElement.src = 'images/stata.jpg';
    this.contentImgElement.height = 227;

    this.styleNames = STYLE_NAMES;
    this.selectedStyleName = 'udnie';
    this.styleImgElement.src = 'images/udnie.jpg';
    this.styleImgElement.height = 227;

    this.currentImgElement.src = 'images/noise.jpg';
    this.currentImgElement.height = 227;

    // Add listener to drop downs
    const contentDropdown = this.querySelector('#content-dropdown');
    // tslint:disable-next-line:no-any
    contentDropdown.addEventListener('iron-activate', (event: any) => {
      this.contentImgElement.src = 'images/' + event.detail.selected + '.jpg';
    });

    const styleDropdown = this.querySelector('#style-dropdown');
    // tslint:disable-next-line:no-any
    styleDropdown.addEventListener('iron-activate', (event: any) => {
      this.styleImgElement.src = 'images/' + event.detail.selected + '.jpg';
    });

    // Add listener to train
    this.trainButton = this.querySelector('#train') as HTMLButtonElement;
    this.trainButton.addEventListener('click', () => {
      // this.createModel();
      // this.startTraining();
    });

    // Add listener to stop
    this.stopButton = this.querySelector('#stop') as HTMLButtonElement;
    this.stopButton.addEventListener('click', () => {
      this.applicationState = ApplicationState.IDLE;
      // this.graphRunner.stopTraining();
    });

    // Initialize TransformNet
    this.transformNet = new TransformNet(this.gpgpu, this.math);

    this.myDebug();
  }

  myDebug() {
    var foo = [];
    for (let i = 0; i < 108; i++) {
       foo.push(i);
    }

    console.log('debug!');
    const a = Array3D.new([3, 3, 12], foo);
    console.log(a);
    console.log(a.getValues());
    console.log(a.get(0, 0, 0));
    console.log(a.get(0, 1, 0));
    console.log(a.get(0, 2, 0));
    console.log(a.get(1, 0, 0));
    console.log(a.get(1, 1, 0));
    console.log(a.get(1, 2, 0));
    console.log(a.get(2, 0, 0));
    console.log(a.get(2, 1, 0));
    console.log(a.get(2, 2, 0));
    const switched = this.mathCPU.switchDim(a, [2, 0, 1]);
    console.log(switched);
    const switchedValues = switched.getValues();
    console.log(switchedValues);
    var means = [];
    var variances = [];
    for (let i = 0; i < 12; i ++) {
      var curr = switchedValues.slice(i*9, (i+1)*9);

      var sum = 0;
      for (let j = 0; j < curr.length; j++) {
        sum += curr[j];
      }
      var avg = sum / curr.length;
      means.push(avg);

      var diffSum = 0;
      for (let j = 0; j < curr.length; j++) {
        diffSum += (avg - curr[j]) * (avg - curr[j]);
      }
      variances.push(diffSum / curr.length);

      console.log(curr);
      console.log(means);
      console.log(variances);
    }
  }
}

document.registerElement(StyleTransferDemo.prototype.is, StyleTransferDemo);
