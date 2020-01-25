export interface Classification {
	label: string;
	labelIndex: number;
	score: number;
}
export interface Detection extends Classification {
	x: number;
	y: number;
	w: number;
	h: number;
	raw?: any;
}
export interface BaseOutput {
	imageSize?: number[];
	inputSize?: number[];
	timings?: any;
}
export interface ClassificationOutput extends BaseOutput {
	classifications: Classification[];
}
export interface ClassificationGroupOutput extends BaseOutput {
	classifications: ClassificationOutput[];
}

export interface DetectionOutput extends BaseOutput {
	detections: Detection[];
}
export interface DetectionGroupOutput extends BaseOutput {
	detections: DetectionOutput[];
}

export type ClassifierOutput = ClassificationOutput | ClassificationGroupOutput;
export type DetectorOutput = DetectionOutput | DetectionGroupOutput;
