import { Detection } from '../interfaces';

export function draw(detections: Detection[], canvas: HTMLCanvasElement, labelsLength?: number) {
	const ctx = canvas.getContext('2d');
	ctx.clearRect(0, 0, canvas.width, canvas.height);

	for (const det of detections) {
		drawObject(det, ctx);
	}
}

function drawObject(item: Detection, ctx: CanvasRenderingContext2D) {
	const lineWidth = 3;
	const textBoxHeight = 20;
	const label = `${item.label}:${(item.score * 100).toFixed(1)}%`;
	const textBoxWidth = ctx.measureText(label).width + 25;
	// console.log(ctx.measureText(item.label));
	ctx.font = '1rem Segoe UI';
	ctx.lineWidth = lineWidth;
	const color = hsl(item.labelIndex, 0, 50);
	ctx.strokeStyle = color;
	ctx.fillStyle = color;
	ctx.beginPath();
	ctx.rect(item.x, item.y, item.w, item.h);
	ctx.stroke();
	ctx.fillStyle = color;
	const textBoxX = item.x - lineWidth / 2;
	const textBoxY = item.y + lineWidth / 2;
	if (item.y - 25 >= 0) {
		// Box on top
		ctx.fillRect(textBoxX, textBoxY, textBoxWidth, -textBoxHeight);
		ctx.fillStyle = '#000000';
		ctx.fillText(label, textBoxX + 5, textBoxY - 5);
	} else {
		// Box Inside
		ctx.fillRect(textBoxY, textBoxY, textBoxWidth, textBoxHeight);
		ctx.fillStyle = '#000000';
		ctx.fillText(label, textBoxX, textBoxY);
	}
}
function hsl(num: number, min: number, max: number) {
	return `hsla(${map(num, min, max)}, 100%, 50%,1)`;
}
function map(num: number, inmin: number, inmax: number) {
	return ((num - inmin) * 360) / (inmax - inmin);
}
