randomImages = ['../shared/img/bird.jpg', '../shared/img/dog.jpg', '../shared/img/eagle.jpg', '../shared/img/giraffe.jpg', '../shared/img/horses.jpg', '../shared/img/kite.jpg', '../shared/img/person.jpg', '../shared/img/scream.jpg'];

console.log('Loaded');
console.log(YOLO);
const detectorConfg = {
	...YOLO.YOLOV3Config,
	modelURL: '../models/objectdetection/yolov3/model_default.json',
	modelSize: [416, 416],
};

const detector = new YOLO.YOLODetector(detectorConfg);
console.log(detector);

onMediaSizeChanged = () => {
	const _mediaSourceRef = document.getElementById('media-element');
	const _canvasRef = document.getElementById('media-element-canvas');
	_mediaSourceRef.setAttribute('width', `${_mediaSourceRef.offsetWidth}px`);
	_mediaSourceRef.setAttribute('height', `${_mediaSourceRef.offsetHeight}px`);
	_canvasRef.width = _mediaSourceRef.offsetWidth;
	_canvasRef.height = _mediaSourceRef.offsetHeight;
};

// media element size changed
let MediaSourceRef = document.getElementById('media-element');
resizeObserver = new ResizeObserver(onMediaSizeChanged).observe(MediaSourceRef);
let canClassify = false;

onLoadClicked = async () => {
	console.log('loading...');
	disableBtns();
	await detector.load();
	await detector.cache();
	console.log('Cached');
	canClassify = true;
	enableBtns();
};

onDetectClicked = () => {
	if (canDetect) {
		const image = document.getElementById('media-element');
		let p0 = performance.now();
		const detections = detector.detect(image);
		let p1 = performance.now();
		console.warn(`Took ${p1 - p0}ms`);
		// // add results to screen
		// const resultContainer = document.getElementById('result-data');
		// var documentFragment = document.createDocumentFragment();

		// for (const classif of classification.classifications) {
		// 	const div = document.createElement('div');
		// 	div.innerHTML = ` Class :  ${classif.label} /  Accuracy : ${classif.score.toFixed(10)}`;
		// 	documentFragment.appendChild(div);
		// }
		// while (resultContainer.firstChild) {
		// 	resultContainer.removeChild(resultContainer.firstChild);
		// }
		// resultContainer.appendChild(documentFragment);
		console.log(detections);
		const canvas = document.getElementById('media-element-canvas');

		YOLO.draw(detections.detections, canvas);
	} else {
		console.warn('Please Load the model first');
	}
};

onDetectAsyncClicked = () => {
	if (canDetect) {
		const image = document.getElementById('media-element');

		let p0 = performance.now();
		classifier.DetectAsync(image).then(classifications => {
			console.log(classifications);

			let p1 = performance.now();
			console.warn(`Took ${p1 - p0}ms`);
			// add results to screen
			const resultContainer = document.getElementById('result-data');
			var documentFragment = document.createDocumentFragment();

			for (const classif of classifications.classifications) {
				const div = document.createElement('div');
				div.innerHTML = ` Class :  ${classif.label} /  Accuracy : ${classif.score.toFixed(10)}`;
				documentFragment.appendChild(div);
			}
			while (resultContainer.firstChild) {
				resultContainer.removeChild(resultContainer.firstChild);
			}
			resultContainer.appendChild(documentFragment);
		});
	} else {
		console.warn('Please Load the model first');
	}
};

onBenchMark = () => {
	if (canDetect) {
	} else {
	}
};

getRandomArbitrary = (min, max) => {
	return Math.random() * (max - min) + min;
};

onRandomImage = () => {
	const newIndex = Math.round(getRandomArbitrary(0, randomImages.length));
	const ImageUrl = randomImages[newIndex];
	console.log(newIndex, ImageUrl);
	const image = document.getElementById('media-element');
	image.src = ImageUrl;
};

OnGroupDetect = () => {
	const container = document.getElementById('group-image-container');
	const images = container.getElementsByTagName('img');
	const array = Array.from(images);
	let p0 = performance.now();
	const classifications = classifier.classiftyMultiple(...array);
	let p1 = performance.now();
	console.warn(`Took ${p1 - p0}ms`);
	console.log(classifications);
};

OnGroupDetectAsync = () => {
	const container = document.getElementById('group-image-container');
	const images = container.getElementsByTagName('img');
	const array = Array.from(images);
	let p0 = performance.now();
	classifier.DetectMultipleAsync(...array).then(classifications => {
		let p1 = performance.now();
		console.warn(`Took ${p1 - p0}ms`);
		console.log(classifications);
	});
};

disableBtns = () => {
	const loadBtns = document.getElementsByClassName('load-Btn');
	for (const btn of loadBtns) {
		btn.innerHTML = '<div class="spinner-grow spinner-grow-sm">';
	}

	const actionBtns = document.getElementsByClassName('action-Btn');
	for (const btn of actionBtns) {
		btn.setAttribute('disabled', true);
	}
};
enableBtns = () => {
	const loadBtns = document.getElementsByClassName('load-Btn');

	for (const btn of loadBtns) {
		btn.innerHTML = 'Loaded';
		btn.setAttribute('disabled', true);
	}
	const actionBtns = document.getElementsByClassName('action-btn');
	for (const btn of actionBtns) {
		btn.removeAttribute('disabled');
	}
};
