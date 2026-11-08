/**
 * MNIST-compatible preprocessing.
 *
 * Whatever the user gives us (a 280x280 drawing canvas or a photo of a digit)
 * has to look like a training sample before it reaches the CNN:
 *   1. flatten to luminance and drop the background
 *   2. invert if the image is dark ink on light paper
 *   3. threshold (Otsu) and crop to the ink bounding box
 *   4. scale the longest side to 20px using 4x4 supersampled area integration
 *   5. recentre on the centre of mass and normalize peak contrast to 255
 */

export const FRAME_SIZE = 28;
export const CONTENT_SIZE = 20;

export class PreprocessError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "PreprocessError";
  }
}

export type PreprocessResult = {
  pixels: Uint8Array;
  base64: string;
  inverted: boolean;
  threshold: number;
  inkPixels: number;
  bbox: { x: number; y: number; width: number; height: number };
  centerOfMass: { x: number; y: number };
  scaledSize: { width: number; height: number };
};

function otsuThreshold(gray: Uint8Array): number {
  const histogram = new Array<number>(256).fill(0);
  for (let i = 0; i < gray.length; i++) histogram[gray[i]]++;

  const total = gray.length;
  let sum = 0;
  for (let v = 0; v < 256; v++) sum += v * histogram[v];

  let sumBackground = 0;
  let weightBackground = 0;
  let bestVariance = -1;
  let threshold = 128;

  for (let v = 0; v < 256; v++) {
    weightBackground += histogram[v];
    if (weightBackground === 0) continue;
    const weightForeground = total - weightBackground;
    if (weightForeground === 0) break;

    sumBackground += v * histogram[v];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const variance =
      weightBackground * weightForeground * (meanBackground - meanForeground) ** 2;

    if (variance > bestVariance) {
      bestVariance = variance;
      threshold = v;
    }
  }
  return threshold;
}

function bilinear(
  source: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
): number {
  const cx = Math.min(Math.max(x, 0), width - 1);
  const cy = Math.min(Math.max(y, 0), height - 1);
  const x0 = Math.floor(cx);
  const y0 = Math.floor(cy);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = cx - x0;
  const fy = cy - y0;
  const top = source[y0 * width + x0] * (1 - fx) + source[y0 * width + x1] * fx;
  const bottom = source[y1 * width + x0] * (1 - fx) + source[y1 * width + x1] * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * 4x4 supersampled area average over the source footprint of one target pixel,
 * preventing stroke aliasing when downscaling from 280x280 -> 20x20.
 */
function areaSample(
  source: Float32Array,
  width: number,
  height: number,
  xStart: number,
  yStart: number,
  xSpan: number,
  ySpan: number,
): number {
  const taps = 4;
  let sum = 0;
  for (let sy = 0; sy < taps; sy++) {
    const py = yStart + ((sy + 0.5) / taps) * ySpan;
    for (let sx = 0; sx < taps; sx++) {
      const px = xStart + ((sx + 0.5) / taps) * xSpan;
      sum += bilinear(source, width, height, px, py);
    }
  }
  return sum / (taps * taps);
}

/**
 * @param gray luminance values 0..255, row-major (width * height)
 */
export function preprocessGray(
  gray: Uint8Array,
  width: number,
  height: number,
): PreprocessResult {
  if (gray.length !== width * height) {
    throw new PreprocessError("Unexpected pixel buffer length.");
  }
  if (width < 8 || height < 8) {
    throw new PreprocessError("Image is too small to read.");
  }

  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const mean = sum / gray.length;
  const inverted = mean > 127;

  const values = new Float32Array(gray.length);
  for (let i = 0; i < gray.length; i++) {
    values[i] = inverted ? 255 - gray[i] : gray[i];
  }

  const threshold = Math.min(
    Math.max(
      otsuThreshold(
        gray.length ? Uint8Array.from(values, (v) => Math.round(v)) : gray,
      ),
      24,
    ),
    210,
  );
  const denominator = Math.max(255 - threshold, 1);

  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let inkPixels = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const index = y * width + x;
      const v = values[index];
      if (v > threshold) {
        values[index] = Math.min(255, ((v - threshold) / denominator) * 255);
        inkPixels++;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      } else {
        values[index] = 0;
      }
    }
  }

  if (inkPixels < 12 || maxX < 0) {
    throw new PreprocessError("No digit detected — draw a digit with thick strokes.");
  }

  const bbox = { x: minX, y: minY, width: maxX - minX + 1, height: maxY - minY + 1 };
  const scale = CONTENT_SIZE / Math.max(bbox.width, bbox.height);
  const scaledWidth = Math.max(1, Math.round(bbox.width * scale));
  const scaledHeight = Math.max(1, Math.round(bbox.height * scale));

  const framed = new Float32Array(FRAME_SIZE * FRAME_SIZE);
  const offsetX = Math.floor((FRAME_SIZE - scaledWidth) / 2);
  const offsetY = Math.floor((FRAME_SIZE - scaledHeight) / 2);
  const xSpan = bbox.width / scaledWidth;
  const ySpan = bbox.height / scaledHeight;

  for (let ty = 0; ty < scaledHeight; ty++) {
    for (let tx = 0; tx < scaledWidth; tx++) {
      const sx = bbox.x + tx * xSpan - 0.5;
      const sy = bbox.y + ty * ySpan - 0.5;
      framed[(offsetY + ty) * FRAME_SIZE + offsetX + tx] = areaSample(
        values,
        width,
        height,
        sx,
        sy,
        xSpan,
        ySpan,
      );
    }
  }

  // Recentre on the centre of mass in the same way the MNIST digits are framed.
  let mass = 0;
  let massX = 0;
  let massY = 0;
  for (let y = 0; y < FRAME_SIZE; y++) {
    for (let x = 0; x < FRAME_SIZE; x++) {
      const v = framed[y * FRAME_SIZE + x];
      mass += v;
      massX += v * (x + 0.5);
      massY += v * (y + 0.5);
    }
  }
  if (mass <= 0) {
    throw new PreprocessError("No digit detected — try a thicker stroke.");
  }
  const centerX = massX / mass;
  const centerY = massY / mass;
  const shiftX = Math.max(-4, Math.min(4, FRAME_SIZE / 2 - centerX));
  const shiftY = Math.max(-4, Math.min(4, FRAME_SIZE / 2 - centerY));

  const shifted = new Float32Array(FRAME_SIZE * FRAME_SIZE);
  let peak = 0;
  for (let y = 0; y < FRAME_SIZE; y++) {
    for (let x = 0; x < FRAME_SIZE; x++) {
      const v = bilinear(framed, FRAME_SIZE, FRAME_SIZE, x - shiftX, y - shiftY);
      shifted[y * FRAME_SIZE + x] = v;
      if (v > peak) peak = v;
    }
  }

  // Normalize peak contrast to 255 (matches MNIST full dynamic range)
  const normScale = peak > 10 ? 255 / peak : 1;
  const pixels = new Uint8Array(FRAME_SIZE * FRAME_SIZE);
  for (let i = 0; i < shifted.length; i++) {
    pixels[i] = Math.max(0, Math.min(255, Math.round(shifted[i] * normScale)));
  }

  return {
    pixels,
    base64: Buffer.from(pixels).toString("base64"),
    inverted,
    threshold,
    inkPixels,
    bbox,
    centerOfMass: {
      x: Number((centerX + shiftX).toFixed(2)),
      y: Number((centerY + shiftY).toFixed(2)),
    },
    scaledSize: { width: scaledWidth, height: scaledHeight },
  };
}

/** Convert RGBA pixels (browser ImageData) to luminance. */
export function rgbaToGray(
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Uint8Array {
  const gray = new Uint8Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    const a = rgba[i * 4 + 3] / 255;
    gray[i] = Math.round((0.299 * r + 0.587 * g + 0.114 * b) * a);
  }
  return gray;
}

export function base64ToBytes(base64: string): Uint8Array {
  const normalized = base64.replace(/^data:[^;]+;base64,/, "");
  return new Uint8Array(Buffer.from(normalized, "base64"));
}

export function bytesToBase64(bytes: Uint8Array): string {
  return Buffer.from(bytes).toString("base64");
}
