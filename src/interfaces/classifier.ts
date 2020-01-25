import { Shape } from '@tensorflow/tfjs';
import { ClassifierInput } from '../types';

import { ImageOptions } from './imageoptions';
import { ClassifierOutput } from './output';

export interface IClassifier {
	load(): Promise<void>;
	cache(): void;

	dispose(): void;

	classify(image: ClassifierInput): ClassifierOutput;
	classiftyMultiple(...image: ClassifierInput[]): ClassifierOutput;

	classifyAsync(image: ClassifierInput): Promise<ClassifierOutput>;
	classifyMultipleAsync(...images: ClassifierInput[]): Promise<ClassifierOutput>;
}

export interface IClassifierConfig {
	// model definition
	modelName: string;
	modelURL: string;
	// tslint:: indent
	modelSize: Shape;

	// variables defenition
	classProbThreshold: number;
	topK: number;
	labels: string[];

	// misc
	resizeOption?: ImageOptions;
}
