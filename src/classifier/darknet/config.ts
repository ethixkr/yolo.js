import { IClassifierConfig } from '../../interfaces';
import { darknet9000Labels } from '../../shared/darknet9000_labels';
import { darknetImagnetLabels } from '../../shared/darknet_imagenet_labels';

const darknetRefrenceConfig: IClassifierConfig = {
	modelName: 'darknet-refrence',
	modelURL: '',
	modelSize: [256, 256],
	classProbThreshold: 0.6,
	topK: 5,
	resizeOption: {
		AlignCorners: true,
		ResizeOption: 'Bilinear',
	},
	labels: darknetImagnetLabels,
};
const darknetTinyConfig: IClassifierConfig = {
	modelName: 'tiny-darknet',
	modelURL: '',
	modelSize: [224, 224],
	classProbThreshold: 0.6,
	topK: 5,
	resizeOption: {
		AlignCorners: true,
		ResizeOption: 'Bilinear',
	},
	labels: darknetImagnetLabels,
};
const darknet19Config: IClassifierConfig = {
	modelName: 'darknet-19',
	modelURL: '',
	modelSize: [416, 416],
	classProbThreshold: 0.6,
	topK: 5,
	resizeOption: {
		AlignCorners: true,
		ResizeOption: 'Bilinear',
	},
	labels: darknetImagnetLabels,
};
const darknet9000Config: IClassifierConfig = {
	modelName: 'darknet-9000',
	modelURL: '',
	modelSize: [416, 416],
	classProbThreshold: 0.6,
	topK: 5,
	resizeOption: {
		AlignCorners: true,
		ResizeOption: 'Bilinear',
	},
	labels: darknet9000Labels,
};
export { darknetRefrenceConfig, darknet19Config, darknetTinyConfig, darknet9000Config };
