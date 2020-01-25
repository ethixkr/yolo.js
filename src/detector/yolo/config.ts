import { ImageOptions, YOLODetectorConfig } from '../../interfaces';
import { cocoLabels } from '../../shared/coco_labels';

const defaultResizeOption: ImageOptions = {
	ResizeOption: 'NearestNeighbor',
	AlignCorners: true,
};
// tslint:disable: variable-name
export const YOLOV3TinyConfig: YOLODetectorConfig = {
	modelName: 'tiny-yolo-v3',
	modelURL: '',
	modelSize: [224, 224],
	iouThreshold: 0.5,
	classProbThreshold: 0.5,
	maxOutput: 10,
	resizeOption: defaultResizeOption,
	labels: cocoLabels,
	anchors: [ [10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
	masks: [ [3, 4, 5], [0, 1, 2] ],
};

export const YOLOV2TinyConfig: YOLODetectorConfig = {
	modelName: 'tiny-yolo-v2',
	modelURL: '',
	modelSize: [224, 224],
	iouThreshold: 0.5,
	classProbThreshold: 0.5,
	maxOutput: 10,
	resizeOption: defaultResizeOption,
	labels: cocoLabels,
	masks: [[0, 1, 2, 3, 4]],
	anchors: [ [0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]],
};

export const YOLOV3Config: YOLODetectorConfig = {
	modelName: 'yolo-v3',
	modelURL: '',
	modelSize: [224, 224],
	iouThreshold: 0.5,
	classProbThreshold: 0.5,
	maxOutput: 10,
	resizeOption: defaultResizeOption,
	labels: cocoLabels,
	anchors: [ [10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]],
	masks: [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
};

export const YOLOLiteConfig: YOLODetectorConfig = {
	modelName: 'tiny-yolo-v2-lite',
	modelURL: '',
	modelSize: [224, 224],
	iouThreshold: 0.2,
	classProbThreshold: 0.4,
	maxOutput: 10,
	resizeOption: defaultResizeOption,
	labels: cocoLabels,
	masks: [[0, 1, 2, 3, 4]],
	anchors: [ [1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],
};
