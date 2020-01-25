import { Classification, ClassificationGroupOutput, ClassificationOutput, IClassifier, IClassifierConfig, ImageOptions } from '../../interfaces';
import { tf } from '../../tf';
import { Input } from '../../types';
import { loadModel, now, preProcess } from '../../utils';
// tslint:disable: member-ordering
export class DarknetClassifier implements IClassifier, IClassifierConfig {
	public model: tf.LayersModel;
	public modelName: string;
	public modelURL: string;
	public modelSize: tf.Shape;
	public classProbThreshold: number;
	public topK: number;
	public labels: string[];
	public resizeOption: ImageOptions;

	constructor(options: IClassifierConfig) {
		this.modelName = options.modelName;
		this.modelURL = options.modelURL;
		this.modelSize = options.modelSize;
		this.classProbThreshold = options.classProbThreshold;
		this.topK = options.topK;
		this.labels = options.labels;
		this.resizeOption = options.resizeOption;
		this.resizeOption.expnadDims = true;
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
			await this.classifyAsync(dummy);
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

	public classify(image: Input): ClassificationOutput {
		return tf.tidy(() => {
			const perf = {};
			const t0 = now();
			const [originalImageShapes, data] = this.preProcess(image);
			const t1 = now();
			const classes = this.predict(data);
			const t2 = now();
			const items = this.postProcess(classes, [originalImageShapes], 1);
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
			// tslint:enable: no-string-literal
			return {
				...items[0],
				timings: perf,
			} as any;
		}) as ClassificationOutput;
	}
	public async classifyAsync(image: Input): Promise<ClassificationOutput> {
		await tf.nextFrame();
		const perf = {};
		let t0: number;
		let t1: number;
		let t2: number;
		const [imageShape, classes] = tf.tidy(() => {
			t0 = now();
			const [originalImageShapes, data] = this.preProcess(image);
			t1 = now();
			const _classes = this.predict(data);
			t2 = now();
			return [originalImageShapes, _classes];
		});
		const items = await this.postProcessAsync(classes, [imageShape], 1);
		tf.dispose(classes);
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
		// tslint:enable: no-string-literal
		return {
			...items[0],
			timings: perf,
		};
	}

	/**
	 * classifies a set of images and their topk as an array
	 * @param images  a group of images that can be in different formats
	 */
	public classiftyMultiple(...images: Input[]): ClassificationGroupOutput {
		return this.classiftyMultipleV1(...images);
	}
	public classiftyMultipleV0(...images: Input[]): ClassificationGroupOutput {
		return tf.tidy(() => {
			const perf = {};
			const t0 = now();
			// create input tensor form multiple images
			const inputs: tf.Tensor[] = [];
			const inputsShapes: tf.Shape[] = []; // height, width

			for (const image of images) {
				const [imageShape, imageTensor] = this.preProcess(image);
				inputs.push(imageTensor);
				inputsShapes.push(imageShape); // height, width
			}
			const inputTensor = tf.concat(inputs, 0);
			const t1 = now();
			const classes = this.predict(inputTensor);
			const t2 = now();
			const classifications = this.postProcess(classes, inputsShapes, inputsShapes.length);
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
			// tslint:enable: no-string-literal
			return {
				classifications,
				timings: perf,
			} as any;
		}) as ClassificationGroupOutput;
	}
	public classiftyMultipleV1(...images: Input[]): ClassificationGroupOutput {
		const result: any[] = [];
		for (const image of images) {
			result.push(this.classify(image));
		}
		return {
			classifications: result,
		};
	}

	public async classifyMultipleAsync(...images: Input[]): Promise<ClassificationGroupOutput> {
		return await this.classifyMultipleAsyncV2(...images);
	}
	public async classifyMultipleAsyncV0(...images: Input[]): Promise<ClassificationGroupOutput> {
		await tf.nextFrame();
		const perf = {};
		let t0: number;
		let t1: number;
		let t2: number;
		const [imageShapes, classes] = tf.tidy(() => {
			t0 = now();
			// create input tensor form multiple images
			const inputs: tf.Tensor[] = [];
			const inputsShapes: tf.Shape[] = []; // height, width

			for (const image of images) {
				const [imageShape, imageTensor] = this.preProcess(image);
				inputs.push(imageTensor);
				inputsShapes.push(imageShape); // height, width
			}
			const inputTensor = tf.concat(inputs, 0);
			t1 = now();
			const _classes = this.predict(inputTensor);
			t2 = now();
			return [inputsShapes, _classes];
		});

		const classifications = await this.postProcessAsync(classes, imageShapes, 1);
		tf.dispose(classes);
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
		// tslint:enable: no-string-literal
		return {
			classifications,
			timings: perf,
		};
	}
	public async classifyMultipleAsyncV1(...images: Input[]): Promise<ClassificationGroupOutput> {
		const result: any[] = [];
		for (const image of images) {
			result.push(await this.classifyAsync(image));
		}
		return {
			classifications: result,
		};
	}
	public async classifyMultipleAsyncV2(...images: Input[]): Promise<ClassificationGroupOutput> {
		const promises: any[] = [];
		for (const image of images) {
			promises.push(this.classifyAsync(image));
		}
		const result = await Promise.all(promises);
		return {
			classifications: result,
		};
	}

	private createClassificationsArray(values: number[], indices: number[]): Classification[] {
		const classifications: Classification[] = [];
		for (let i = 0; i < indices.length; i++) {
			const c: Classification = {
				label: this.labels[indices[i]],
				labelIndex: indices[i],
				score: values[i],
			};
			classifications.push(c);
		}
		return classifications;
	}

	private preProcess(image): [tf.Shape, tf.Tensor] {
		const [oldShape, imageTensor] = preProcess(image, this.modelSize, this.resizeOption);
		return [oldShape, imageTensor.expandDims(0)];
	}

	private predict(data): tf.Tensor {
		return tf.softmax(this.model.predict(data) as tf.Tensor, -1);
	}

	private postProcess(classes: tf.Tensor, originalImageShapes: tf.Shape[], imageCount: number): ClassificationOutput[] {
		const { values, indices } = tf.topk(classes, this.topK);
		const valuesArray = values.arraySync() as number[][];
		const indicesArray = indices.arraySync() as number[][];
		const classificationsArray: ClassificationOutput[] = [];
		for (let i = 0; i < imageCount; i++) {
			const classifications = this.createClassificationsArray(valuesArray[i], indicesArray[i]);
			classificationsArray.push({ classifications, imageSize: originalImageShapes[i], inputSize: this.modelSize });
		}
		return classificationsArray;
	}

	private async postProcessAsync(classes: tf.Tensor, originalImageShapes: number[][], imageCount: number): Promise<ClassificationOutput[]> {
		const { values, indices } = tf.topk(classes, this.topK);
		const [valuesArray, indicesArray] = await Promise.all([values.array(), indices.array()]);
		tf.dispose(values);
		tf.dispose(indices);
		const classificationsArray: ClassificationOutput[] = [];
		for (let i = 0; i < imageCount; i++) {
			const classifications = this.createClassificationsArray(valuesArray[i], indicesArray[i]);
			classificationsArray.push({ classifications, imageSize: originalImageShapes[i], inputSize: this.modelSize });
		}
		return classificationsArray;
	}
}
