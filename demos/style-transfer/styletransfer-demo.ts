import '../demo-header';
import '../demo-footer';
// tslint:disable-next-line:max-line-length
import {Array1D, Array3D, gpgpu_util, GPGPUContext, NDArrayMathCPU, NDArrayMathGPU} from '../deeplearn';
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

const CONTENT_NAMES = ['stata', 'face', 'face2', 'face3', 'diana', 'Upload from file'];
const STYLE_NAMES = ['udnie', 'scream'];

export class StyleTransferDemo extends StyleTransferDemoPolymer {
  // DeeplearnJS stuff
  private math: NDArrayMathGPU;
  private mathCPU: NDArrayMathCPU;
  private gpgpu: GPGPUContext;

  private transformNet: TransformNet;

  // DOM Elements
  private contentImgElement: HTMLImageElement;
  private styleImgElement: HTMLImageElement;

  private canvas: HTMLCanvasElement;
  private canvasContext: CanvasRenderingContext2D;
  private imageData: ImageData;

  private startButton: HTMLButtonElement;

  private camDialog: any;
  private stream: MediaStream;
  private webcamVideoElement: HTMLVideoElement;
  private takePicButton: HTMLButtonElement;
  private closeModal: HTMLButtonElement;

  private fileSelect: HTMLButtonElement;

  // Polymer properties
  private contentNames: string[];
  private selectedContentName: string;
  private styleNames: string[];
  private selectedStyleName: string;

  private status: string;

  private applicationState: ApplicationState;

  ready() {
    // Initialize DeeplearnJS stuff
    this.gpgpu = new GPGPUContext;
    this.math = new NDArrayMathGPU(this.gpgpu);
    this.mathCPU = new NDArrayMathCPU;

    // Initialize polymer properties
    this.applicationState = ApplicationState.IDLE;
    this.status = '';

    // Retrieve DOM for images
    this.contentImgElement =
        this.querySelector('#contentImg') as HTMLImageElement;
    this.styleImgElement = 
        this.querySelector('#styleImg') as HTMLImageElement;

    // Render DOM for images
    this.contentNames = CONTENT_NAMES;
    this.selectedContentName = 'face';
    this.contentImgElement.src = 'images/face.jpg';
    this.contentImgElement.height = 250;

    this.styleNames = STYLE_NAMES;
    this.selectedStyleName = 'udnie';
    this.styleImgElement.src = 'images/udnie.jpg';
    this.styleImgElement.height = 250;

    this.canvas = this.querySelector('#imageCanvas') as HTMLCanvasElement;
    this.canvas.width = 0;
    this.canvas.height = 0;
    this.canvasContext = this.canvas.getContext('2d') as CanvasRenderingContext2D;
    this.canvas.style.display = 'none';

    this.initWebcamVariables();

    this.fileSelect = this.querySelector('#fileSelect') as HTMLButtonElement;
    this.fileSelect.addEventListener('change', (event: any) => {
      const f: File = event.target.files[0];
      const fileReader: FileReader = new FileReader;
      fileReader.onload = ((e) => {
        const target: FileReader = e.target as FileReader;
        this.contentImgElement.src = target.result;
      });
      fileReader.readAsDataURL(f);
    });

    // Add listener to drop downs
    const contentDropdown = this.querySelector('#content-dropdown');
    // tslint:disable-next-line:no-any
    contentDropdown.addEventListener('iron-activate', (event: any) => {
      if (event.detail.selected === 'Use webcam') {
        this.openWebcamModal();
      }
      else if (event.detail.selected === 'Upload from file') {
        this.fileSelect.click();
      }
      else {
        this.contentImgElement.src = 'images/' + event.detail.selected + '.jpg';
      }
    });

    const styleDropdown = this.querySelector('#style-dropdown');
    // tslint:disable-next-line:no-any
    styleDropdown.addEventListener('iron-activate', (event: any) => {
      this.styleImgElement.src = 'images/' + event.detail.selected + '.jpg';
    });

    // Add listener to start
    this.startButton = this.querySelector('#start') as HTMLButtonElement;
    this.startButton.addEventListener('click', () => {
      this.startButton.textContent = 'Starting style transfer.. Downloading + running model';
      this.startButton.disabled = true;
      this.transformNet = new TransformNet(this.gpgpu, this.math,
        this.selectedStyleName);
      this.transformNet.loadVariables().then(() => {
        this.startButton.textContent = 'Processing image';
        this.runInference();
        this.startButton.textContent = 'Start Style Transfer';
        this.startButton.disabled = false;
      });
    });
  }

  private initWebcamVariables() {
    this.camDialog = this.querySelector('#webcam-dialog');
    this.webcamVideoElement = this.querySelector('#webcamVideo') as HTMLVideoElement;
    this.takePicButton = this.querySelector('#takePicButton') as HTMLButtonElement;
    this.closeModal = this.querySelector('#closeModal') as HTMLButtonElement;

    // Check if webcam is even available
    // tslint:disable-next-line:no-any
    const navigatorAny = navigator as any;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;
    if (navigator.getUserMedia) {
      const contentNames = CONTENT_NAMES.slice();
      contentNames.unshift('Use webcam');
      this.contentNames = contentNames;
    }

    this.closeModal.addEventListener('click', () => {
      this.stream.getTracks()[0].stop();
    });

    this.takePicButton.addEventListener('click', () => {
      var hiddenCanvas: HTMLCanvasElement = 
        this.querySelector('#hiddenCanvas') as HTMLCanvasElement;
      var hiddenContext: CanvasRenderingContext2D = hiddenCanvas.getContext('2d');
      hiddenCanvas.width = this.webcamVideoElement.width;
      hiddenCanvas.height = this.webcamVideoElement.height;
      hiddenContext.drawImage(this.webcamVideoElement, 0, 0, 
        hiddenCanvas.width, hiddenCanvas.height);
      var imageDataURL = hiddenCanvas.toDataURL('image/jpg');
      this.contentImgElement.src = imageDataURL;
      this.stream.getTracks()[0].stop();
    });
  }

  private openWebcamModal() {
    this.camDialog.open();
    navigator.getUserMedia(
      {
        video: true
      },
      (stream) => {
        this.stream = stream;
        this.webcamVideoElement.src = window.URL.createObjectURL(stream);
        this.webcamVideoElement.play();
      },
      (err) => {
        console.error(err);
      }
    );
  }

  runInference() {
    const canvasTextureShape: [number, number] = [this.contentImgElement.height, 
    this.contentImgElement.width];
    console.log(canvasTextureShape);
    
    this.math.scope((keep, track) => {
      const canvasTexture =
          this.math.getTextureManager().acquireTexture(canvasTextureShape);
      this.gpgpu.uploadPixelDataToTexture(canvasTexture, this.contentImgElement);
      console.log(canvasTexture);

      const preprocessed = this.transformNet.preprocessColorTextureToArray3D(
        canvasTexture, canvasTextureShape);

      const inferenceResult = this.transformNet.infer(preprocessed);
      this.drawOnCanvas(inferenceResult);
    });
  }

  private setCanvasShape(shape: number[]) {
    this.canvas.width = shape[1];
    this.canvas.height = shape[0];
    this.canvas.style.width = shape[1] + 'px';
    this.canvas.style.height = shape[0] + 'px';
  }

  private drawOnCanvas(ndarray: Array3D) {
    this.setCanvasShape(ndarray.shape);
    this.imageData = this.canvasContext.createImageData(
        this.canvas.width, this.canvas.height);

    let pixelOffset = 0;
    for (let i = 0; i < ndarray.shape[0]; i++) {
      for (let j = 0; j < ndarray.shape[1]; j++) {
        this.imageData.data[pixelOffset++] = ndarray.get(i, j, 0);
        this.imageData.data[pixelOffset++] = ndarray.get(i, j, 1);
        this.imageData.data[pixelOffset++] = ndarray.get(i, j, 2);
        this.imageData.data[pixelOffset++] = 255;
      }
    }

    this.canvas.style.display = '';
    this.canvasContext.putImageData(this.imageData, 0, 0);
  }
}

document.registerElement(StyleTransferDemo.prototype.is, StyleTransferDemo);
