import { tf } from './tf';

export type Input = HTMLCanvasElement | HTMLVideoElement | ImageData | HTMLImageElement | tf.Tensor;

export type DetectorInput = Input;
export type ClassifierInput = Input;

export type ImageResizeOption = 'NearestNeighbor' | 'Bilinear';
export type Shape = tf.Shape;
