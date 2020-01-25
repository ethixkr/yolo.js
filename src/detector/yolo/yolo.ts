import { Detection, DetectorOutput, IDetector, ImageOptions, YOLODetectorConfig } from '../../interfaces';
import { tf } from '../../tf';
import { DetectorInput } from '../../types';
import { isTensorArray, loadModel, now, preProcess } from '../../utils';

export class YOLODetector implements IDetector, YOLODetectorConfig {
	public model: tf.LayersModel;

	public modelName: string;
	public modelURL: string;
	public modelSize: number[];

	public iouThreshold: number;
	public classProbThreshold: number;
	public maxOutput: number;

	public labels: string[];
	public anchors: number[][];
	public masks: number[][];
	public resizeOption: ImageOptions;

	constructor(options) {
		this.modelName = options.modelName;
		this.modelURL = options.modelURL;
		this.modelSize = options.modelSize; // height, width

		this.iouThreshold = options.iouThreshold;
		this.classProbThreshold = options.classProbThreshold;
		this.maxOutput = options.maxOutput;

		this.labels = options.labels;
		this.anchors = options.anchors;
		this.masks = options.masks;

		this.resizeOption = options.resizeOption;
	}

	/**
	 * Loads the model from `modelURL`
	 */
	public async load(): Promise<void> {
		try {
			this.model = await loadModel(this.modelURL);
		} catch (error) {
			throw error;
		}
	}

	/**
	 * Caches the model
	 */
	public async cache(): Promise<void> {
		try {
			const dummy = tf.zeros([...this.modelSize, 3]);
			await this.detect(dummy);
			tf.dispose(dummy);
		} catch (error) {
			throw error;
		}
	}
	public cacheSync(): void {
		try {
			const dummy = tf.zeros([...this.modelSize, 3]);
			this.detect(dummy);
			tf.dispose(dummy);
		} catch (error) {
			throw error;
		}
	}

	/**
	 * Dispose of the tensors allocated by the model. You should call this when you
	 * are done with the detection.
	 */
	public dispose(): void {
		if (this.model) {
			this.model.dispose();
		}
	}

	public detect(image: DetectorInput): DetectorOutput {
		return this.detectV0(image);
	}
	public detectV0(image: DetectorInput): DetectorOutput {
		const out = tf.tidy(() => {
			const perf = {};
			const t0 = now();
			const [originalImageShape, data] = this.preProcess(image);
			const t1 = now();
			const raw = this.predict(data);
			const t2 = now();
			let [rawBoxes, rawScores] = this.postProcessRawPredictions(raw);
			// remove batch dim;
			rawBoxes = rawBoxes.squeeze([0]);
			rawScores = rawScores.squeeze([0]);
			const [boxes, scores, classes] = this.postProcessBoxes(rawBoxes, rawScores);
			const dectections = this.createDetectionArray(boxes, scores, classes, this.modelSize, originalImageShape);
			const t3 = now();
			// tslint:disable: no-string-literal
			perf['pre-processing'] = t1 - t0;
			perf['infrence'] = t2 - t1;
			perf['post-processing'] = t3 - t2;
			let sum = 0;
			for (const time in perf) {
				if (perf.hasOwnProperty(time)) {
					sum += perf[time];
				}
			}
			perf['total'] = sum;
			const output: DetectorOutput = {
				detections: dectections,
				imageSize: originalImageShape,
				inputSize: this.modelSize,
				timings: perf,
			};
			return output as any;
		});
		return out as DetectorOutput;
	}

	/**
	 * a small and compact implementation (not worth it)
	 * @param image
	 */
	public detectV1(image: DetectorInput): DetectorOutput {
		const out = tf.tidy(() => {
			const modelSize = this.modelSize;
			const perf = {};
			const t0 = now();
			let input: tf.Tensor;
			if (image instanceof tf.Tensor) {
				input = image;
			} else {
				input = tf.browser.fromPixels(image);
			}
			const originalImageShape: tf.Shape = [input.shape[0], input.shape[1]]; // width, height
			// const [originalImageShape, data] = this.preProcess(image);
			input = input.div(255);
			input = tf.image.resizeNearestNeighbor(input as tf.Tensor<tf.Rank.R3>, [modelSize[0],modelSize[1]]).expandDims(0);
			const t1 = now();
			let raw = this.model.predict(input);
			// let  raw = this.predict(input);
			const t2 = now();

			const layers: tf.Tensor[] = [];
			let isV3 = false;
			if (isTensorArray(raw)) {
				for (let i = 0; i < raw.length; i++) {
					layers.push(raw[i]);
				}
				isV3 = true;
			} else {
				layers.push(raw);
			}

			const allboxes: tf.Tensor[] = [];
			const allprobs: tf.Tensor[] = [];

			for (let i = 0; i < layers.length; i++) {
				const mask = this.masks[i];
				const prediction = tf.squeeze(layers[i], [0]);

				let anchors = tf.gather(this.anchors, mask).expandDims(0);
				const numAnchors = anchors.shape[1];

				const [outputHeight, outputWidth, data] = prediction.shape;
				const numBoxes = outputWidth * outputHeight * numAnchors;
				const dataTensorLen = data / numAnchors;
				const numClasses = dataTensorLen - 5;

				const reshaped = tf.reshape(prediction, [outputHeight, outputWidth, numAnchors, dataTensorLen]);
				const scalefactor: number[] = [modelSize[0] / outputHeight, modelSize[1] / outputWidth];

				let [boxXY, boxWH, boxConfidence, boxClassProbs] = tf.split(reshaped, [2, 2, 1, numClasses], 3);

				const gridX = tf.tile(tf.reshape(tf.range(0, outputWidth), [1, -1, 1, 1]), [outputHeight, 1, 1, 1]);
				const gridY = tf.tile(tf.reshape(tf.range(0, outputHeight), [-1, 1, 1, 1]), [1, outputWidth, 1, 1]);
				const indexGrid = tf.concat([gridX, gridY], 3);

				if (isV3) {
					anchors = anchors.div(scalefactor);
				}

				boxXY = tf.div(tf.add(tf.sigmoid(boxXY), indexGrid), [outputWidth, outputHeight]);
				boxWH = tf.div(tf.mul(tf.exp(boxWH), anchors), [outputWidth, outputHeight]);

				const boxYX = tf.concat(tf.split(boxXY, 2, 3).reverse(), 3);
				const boxHW = tf.concat(tf.split(boxWH, 2, 3).reverse(), 3);

				// XY WH to XmaxYmax XminYmin
				const boxMins = tf.mul(tf.sub(boxYX, tf.div(boxHW, 2)), modelSize);
				const boxMaxes = tf.mul(tf.add(boxYX, tf.div(boxHW, 2)), modelSize);

				const boxes = tf.concat([...tf.split(boxMins, 2, 3), ...tf.split(boxMaxes, 2, 3)], 3).reshape([numBoxes, 4]);

				boxConfidence = tf.sigmoid(boxConfidence);
				boxClassProbs = tf.softmax(boxClassProbs);

				const classProbs = tf.mul(boxConfidence, boxClassProbs).reshape([numBoxes, numClasses]);

				// const [box, prob] = this.processLayer(layers[i], anchors, this.modelSize, isV3);
				allboxes.push(boxes);
				allprobs.push(classProbs);
			}
			const boxesTensor = tf.concat(allboxes, 0);
			const probsTensor = tf.concat(allprobs, 0);
			// let [rawBoxes, rawScores] = this.postProcessRawPredictions(raw);

			const scores = tf.max(probsTensor, -1);
			const classes = tf.argMax(probsTensor, -1);

			const indiceTensor = tf.image.nonMaxSuppression(boxesTensor as tf.Tensor<tf.Rank.R2>,
				scores as tf.Tensor<tf.Rank.R1>, this.maxOutput, this.iouThreshold, this.classProbThreshold);

			const filteredBoxes = tf.gather(boxesTensor, indiceTensor).arraySync() as any; // [n, 4]
			const filteredScores = tf.gather(scores, indiceTensor).arraySync() as any; // [n]
			const filteredClasses = tf.gather(classes, indiceTensor).arraySync() as any; // [n]
			// const [boxesa, scores, classes] = this.postProcessBoxes(boxesTensor, probsTensor);

			const numDetections = filteredClasses.length; // || scores.length;
			const detections: Detection[] = [];
			for (let i = 0; i < numDetections; i += 1) {
				// debugger;
				const topY = filteredBoxes[i][0];
				const topX = filteredBoxes[i][1];
				const bottomY = filteredBoxes[i][2];
				const bottomX = filteredBoxes[i][3];

				const w = bottomX - topX;
				const h = bottomY - topY;
				const scaleX = originalImageShape[1] / modelSize[1]; // width
				const scaleY = originalImageShape[0] / modelSize[0]; // height

				const classIndex = filteredClasses[i];
				const _label = this.labels[classIndex];

				const _score = filteredScores[i];

				const detection: Detection = {
					raw: {
						topY,
						topX,
						bottomY,
						bottomX,
						modelSize,
					},
					labelIndex: classIndex,
					label: _label,
					score: _score,
					x: topX * scaleX,
					y: topY * scaleY,
					w: w * scaleX,
					h: h * scaleY,
				};
				detections.push(detection);
			}

			// const dectections = this.createDetectionArray(boxesa, scores, classes, this.modelSize, originalImageShape);

			const t3 = now();
			// tslint:disable: no-string-literal
			perf['pre-processing'] = t1 - t0;
			perf['infrence'] = t2 - t1;
			perf['post-processing'] = t3 - t2;
			let sum = 0;
			for (const time in perf) {
				if (perf.hasOwnProperty(time)) {
					sum += perf[time];
				}
			}
			perf['total'] = sum;
			const output: DetectorOutput = {
				detections,
				imageSize: originalImageShape,
				inputSize: this.modelSize,
				timings: perf,
			};
			return output as any;
		});
		return out as DetectorOutput;
	}

	public async detectAsync(image: DetectorInput): Promise<DetectorOutput> {
		throw new Error('not implemented');
	}

	public detectMultiple(...image: DetectorInput[]): DetectorOutput {
		throw new Error('not implemented');
	}
	public async detectMultipleAsync(...image: DetectorInput[]): Promise<DetectorOutput> {
		throw new Error('not implemented');
	}

	private preProcess(image): [tf.Shape, tf.Tensor] {
		const [oldShape, imageTensor] = preProcess(image, this.modelSize, this.resizeOption);
		return [oldShape, imageTensor.expandDims(0)];
	}

	private predict(data): tf.Tensor {
		return this.model.predict(data) as tf.Tensor;
	}

	/**
	 * the postprocessing function for the yolo object detection algorithm
	 * @param rawPrediction can be a `tf.Tensor` representing a single output (yolov2)
	 * or a `tf.Tensor[]` representing multiple outputs (yolov3 has 3 outputs ).
	 * each output has the shape of `[batch, size, size, ( numClasses + 5 ) * numAnchors]`
	 * with the 5 representing: Box Coodinates [4] + BoxConfidence [1]
	 *
	 * @return a `tf.Tensor[]` that contain `[Boxes, Scores]`
	 * `Boxes` with a shape of `[batch, numBoxes, 4]`
	 * Each `box` is defined by `[topY, topX, bottomY, bottomX]`
	 *
	 * `Scores` with a shape of `[batch, numBoxes, numClasses]`
	 */
	private postProcessRawPredictions(rawPrediction: tf.Tensor[] | tf.Tensor): tf.Tensor[] {
		const layers: tf.Tensor[] = [];
		let isV3 = false;
		if (isTensorArray(rawPrediction)) {
			for (let i = 0; i < rawPrediction.length; i++) {
				layers.push(rawPrediction[i]);
			}
			isV3 = true;
		} else {
			layers.push(rawPrediction);
		}
		const boxes: tf.Tensor[] = [];
		const probs: tf.Tensor[] = [];

		for (let i = 0; i < layers.length; i++) {
			const mask = this.masks[i];
			const anchors = tf.gather(this.anchors, mask).expandDims(0);
			const [box, prob] = this.processLayer(layers[i], anchors, this.modelSize, isV3);
			boxes.push(box);
			probs.push(prob);
		}
		const boxesTensor = tf.concat(boxes, 1);
		const probsTensor = tf.concat(probs, 1);

		return [boxesTensor, probsTensor];
	}

	/**
	 * Process 1 layer of the yolo output
	 * @param prediction a `tf.Tensor` representing 1 output of  the neural net
	 * @param anchorsTensor a `tf.Tensor` representing the anchors that correspond with the output
	 * shape: `[numAnchors, 2]`
	 * @param modelSize the input size for the neural net
	 * @param classesLen the number of classes/labels that the neural net predicts
	 * @param version yolo version `v2` || `v3`
	 *
	 * @return a `tf.Tensor[]` that containes `[boxes , Scores]` that correspond to the specific layer
	 */
	private processLayer(prediction: tf.Tensor, anchors: tf.Tensor, modelSize: tf.Shape, isV3: boolean): tf.Tensor[] {
		const numAnchors = anchors.shape[1];
		const [batch, outputHeight, outputWidth, data] = prediction.shape;
		const numBoxes = outputWidth * outputHeight * numAnchors;
		const dataTensorLen = data / numAnchors;
		const numClasses = dataTensorLen - 5;

		const reshaped = tf.reshape(prediction, [batch, outputHeight, outputWidth, numAnchors, dataTensorLen]);
		const scalefactor: number[] = [modelSize[0] / outputHeight, modelSize[1] / outputWidth];

		let [boxXY, boxWH, boxConfidence, boxClassProbs] = tf.split(reshaped, [2, 2, 1, numClasses], 4);

		const gridX = tf.tile(tf.reshape(tf.range(0, outputWidth), [1, -1, 1, 1]), [outputHeight, 1, 1, 1]);
		const gridY = tf.tile(tf.reshape(tf.range(0, outputHeight), [-1, 1, 1, 1]), [1, outputWidth, 1, 1]);
		let indexGrid = tf.concat([gridX, gridY], 3);
		indexGrid = tf.tile(indexGrid, [1, 1, numAnchors, 1]);

		if (isV3) {
			anchors = anchors.div(scalefactor);
		}

		boxXY = tf.div(tf.add(tf.sigmoid(boxXY), indexGrid), [outputWidth, outputHeight]);
		boxWH = tf.div(tf.mul(tf.exp(boxWH), anchors), [outputWidth, outputHeight]);

		const boxYX = tf.concat(tf.split(boxXY, 2, 4).reverse(), 4);
		const boxHW = tf.concat(tf.split(boxWH, 2, 4).reverse(), 4);

		// XY WH to XmaxYmax XminYmin
		const boxMins = tf.mul(tf.sub(boxYX, tf.div(boxHW, 2)), modelSize);
		const boxMaxes = tf.mul(tf.add(boxYX, tf.div(boxHW, 2)), modelSize);

		const boxes = tf.concat([...tf.split(boxMins, 2, 4), ...tf.split(boxMaxes, 2, 4)], 4).reshape([batch, numBoxes, 4]);

		boxConfidence = tf.sigmoid(boxConfidence);
		boxClassProbs = tf.softmax(boxClassProbs);

		const classProbs = tf.mul(boxConfidence, boxClassProbs).reshape([batch, numBoxes, numClasses]);

		return [boxes, classProbs];
	}

	private postProcessBoxes(rawBoxes: tf.Tensor, rawScores: tf.Tensor): [number[][], number[][], number[][]] {
		const scores = tf.max(rawScores, -1);
		const classes = tf.argMax(rawScores, -1);

		// tslint:disable-next-line: max-line-length
		const indiceTensor = tf.image.nonMaxSuppression(rawBoxes as tf.Tensor<tf.Rank.R2>, scores as tf.Tensor<tf.Rank.R1>, this.maxOutput, this.iouThreshold, this.classProbThreshold);
		const filteredBoxes = tf.gather(rawBoxes, indiceTensor).arraySync() as any; // [ batch, n, 4]
		const filteredScores = tf.gather(scores, indiceTensor).arraySync() as any; // [ batch, n,  ]
		const filteredClasses = tf.gather(classes, indiceTensor).arraySync() as any; // [ batch, n,  ]
		return [filteredBoxes, filteredScores, filteredClasses];
	}
	/**
	 * The final phase in the post processing that outputs the final `Detection[]`
	 * @param finalBoxes an array containing the raw box information
	 * @param imageDims a `Shape` containing the original image dimensions `[height, width]`
	 * @param inputDims a `Shape` containing the model input dimensions `[height, width]`
	 * @return a `DetectorOutput` with the final collected boxes
	 */
	private createDetectionArray(boxes: any[], scores: any[], classes: any[], modelSize: tf.Shape, originalInputSize: tf.Shape): Detection[] {
		const numDetections = classes.length; // || scores.length;
		const detections: Detection[] = [];
		for (let i = 0; i < numDetections; i += 1) {
			// debugger;
			const topY = boxes[i][0];
			const topX = boxes[i][1];
			const bottomY = boxes[i][2];
			const bottomX = boxes[i][3];

			const w = bottomX - topX;
			const h = bottomY - topY;
			const scaleX = originalInputSize[1] / modelSize[1]; // width
			const scaleY = originalInputSize[0] / modelSize[0]; // height

			const classIndex = classes[i];
			const _label = this.labels[classIndex];

			const _score = scores[i];

			const detection: Detection = {
				raw: {
					topY,
					topX,
					bottomY,
					bottomX,
					modelSize,
				},
				labelIndex: classIndex,
				label: _label,
				score: _score,
				x: topX * scaleX,
				y: topY * scaleY,
				w: w * scaleX,
				h: h * scaleY,
			};
			detections.push(detection);
		}
		return detections;
	}
}
