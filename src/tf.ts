// tslint:disable: no-console

import * as tf from '@tensorflow/tfjs';
const version = tf.version.tfjs;
// import '@tensorflow/tfjs-backend-wasm';
// import { setWasmPath } from '@tensorflow/tfjs-backend-wasm';
// setWasmPath('/tfjs-backend-wasm.wasm');
// tf.setBackend('wasm');

console.log(`using Tensorflow.js : ${version}`);
tf.ready().then(() => {
	const backEnd = tf.getBackend();
	console.log(`Using backend : ${backEnd}`);
});

export { tf };
