console.log('Loaded');
console.log(YOLO);
randomImages = ['../shared/img/bird.jpg', '../shared/img/dog.jpg', '../shared/img/eagle.jpg', '../shared/img/giraffe.jpg', '../shared/img/horses.jpg', '../shared/img/kite.jpg', '../shared/img/person.jpg', '../shared/img/scream.jpg'];
const classifierConfg = {
	// ...YOLO.darknetRefrenceConfig,
	// modelURL: '../models/classifiers/darknet-reference/model.json',
	// modelSize: [224, 224],
	...YOLO.darknetTinyConfig,
	modelURL: '../models/classifiers/darknet-tiny/model.json',
	modelSize: [224, 224],
};
const classifier = new YOLO.DarknetClassifier(classifierConfg);
console.log(classifier);
let canClassify = false;

onLoadClicked = async () => {
	console.log('loading...');
	disableBtns();
	await classifier.load();
	await classifier.cache();
	console.log('Cached');
	canClassify = true;
	enableBtns();
};

onClassifyClicked = () => {
	if (canClassify) {
		const image = document.getElementById('media-element');
		let p0 = performance.now();
		const classification = classifier.classify(image);
		let p1 = performance.now();
		console.warn(`[Classify] Took ${p1 - p0}ms`);
		// add results to screen
		const resultContainer = document.getElementById('result-data');
		resultContainer.innerHTML = '';
		displayResults(resultContainer, classification);
		console.log(classification);
	} else {
		console.warn('Please Load the model first');
	}
};

onClassifyAsyncClicked = () => {
	if (canClassify) {
		const image = document.getElementById('media-element');
		let p0 = performance.now();
		classifier.classifyAsync(image).then(classification => {
			let p1 = performance.now();
			console.warn(`[Classify Async] Took ${p1 - p0}ms`);
			const resultContainer = document.getElementById('result-data');
			resultContainer.innerHTML = '';
			displayResults(resultContainer, classification);
			// add results to screen
		});
	} else {
		console.warn('Please Load the model first');
	}
};

onBenchMark = () => {
	if (canClassify) {
	} else {
	}
};

onRandomImage = () => {
	const min = 0;
	const max = randomImages.length;
	const newIndex = Math.round(Math.random() * (max - min) + min);
	const ImageUrl = randomImages[newIndex];
	console.log(newIndex, ImageUrl);
	const image = document.getElementById('media-element');
	image.src = ImageUrl;
};

OnGroupClassify = () => {
	const container = document.getElementById('group-image-container');
	const images = container.getElementsByTagName('img');
	const array = Array.from(images);
	let p0 = performance.now();
	const classifications = classifier.classiftyMultiple(...array);
	let p1 = performance.now();
	console.warn(`Took ${p1 - p0}ms`);
	console.log(classifications);
};

OnGroupClassifyAsync = () => {
	const container = document.getElementById('group-image-container');
	const images = container.getElementsByTagName('img');
	const array = Array.from(images);
	let p0 = performance.now();
	classifier.classifyMultipleAsync(...array).then(classifications => {
		let p1 = performance.now();
		console.warn(`Took ${p1 - p0}ms`);
		console.log(classifications);
	});
};

//////////////////////////// utils /////////////////////
displayResults = (resultContainer,classification) => {
	var documentFragment = document.createDocumentFragment();
	for (const entry of classification.classifications) {
		const div = document.createElement('div');
		div.innerHTML = `[ ${entry.label} ] : ${ entry.score.toFixed(10) * 100}%`;
		documentFragment.appendChild(div);
	}
	resultContainer.appendChild(documentFragment);
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
