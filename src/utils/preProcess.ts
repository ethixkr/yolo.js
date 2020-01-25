import * as tf from '@tensorflow/tfjs';
import { ImageOptions } from '../interfaces';
import { Input, Shape } from '../types';

/**
 * Performs the pre processing ops for the yolo/darknet CNN
 *
 * @param input can be  `HTMLCanvasElement` || `HTMLVideoElement` || `ImageData` || `HTMLImageElement` || `Tensor`;
 * @param size model input size
 * @param options some options regarding image resizing
 *
 * @return  `Shape` representing original image height and width
 *          a 3D tensor with the shape of `[size[0],size[1],3]`
 */
export function preProcess(input: Input, size: Shape, options: ImageOptions): [Shape, tf.Tensor] {
	let image: tf.Tensor;
	if (input instanceof tf.Tensor) {
		image = input;
	} else {
		image = tf.browser.fromPixels(input);
	}
	const imageShape: Shape = [image.shape[0], image.shape[1]]; // height, width

	// Normalize the image from [0, 255] to [0, 1].
	const normalized = image.div(255);
	let resized = normalized;

	if (normalized.shape[0] !== size[0] || normalized.shape[1] !== size[1]) {
		const alignCorners = options.AlignCorners;
		if (options.ResizeOption === 'Bilinear') {
			resized = tf.image.resizeNearestNeighbor(normalized as tf.Tensor<tf.Rank.R3>, [size[0], size[1]], alignCorners);
		} else {
			resized = tf.image.resizeBilinear(normalized as tf.Tensor<tf.Rank.R3>, [size[0], size[1]], alignCorners);
		}
	}
	if (resized.dtype === 'int32') {
		resized = resized.toFloat();
	}
	return [imageShape, resized];
}
