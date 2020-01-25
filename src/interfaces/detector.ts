import { DetectorInput, Shape } from '../types';
import { ImageOptions } from './imageoptions';
import { DetectorOutput } from './output';

export interface IDetector {
	load(): Promise<void>;
	cache(): Promise<void>;

	detect(image: DetectorInput): DetectorOutput;
	detectMultiple(...images: DetectorInput[]): DetectorOutput;

	detectAsync(image: DetectorInput): Promise<DetectorOutput>;
	detectMultipleAsync(...images: DetectorInput[]): Promise<DetectorOutput>;

	dispose(): void;
}

export interface IDetectorConfig {
	modelName: string;
	modelURL: string;

	iouThreshold: number;
	classProbThreshold: number;

	maxOutput: number;

	labels?: string[];
	resizeOption?: ImageOptions;
	modelSize?: Shape;
}

export interface YOLODetectorConfig extends IDetectorConfig {
	anchors: number[][];
	masks: number[][];
}
