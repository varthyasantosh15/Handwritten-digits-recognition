/**
 * Zero-dependency CNN inference engine (cnn-mnist-v2).
 *
 * Architecture (56,970 parameters, 99.31% MNIST test accuracy):
 *   Input 28x28x1
 *     -> Conv2D(16 x 5x5, valid, ReLU)  -> 24x24x16
 *     -> MaxPool2D(2x2)                 -> 12x12x16
 *     -> Conv2D(32 x 3x3, valid, ReLU)  -> 10x10x32
 *     -> MaxPool2D(2x2)                 -> 5x5x32 (800)
 *     -> Dense(64, ReLU)                -> 64
 *     -> Dense(10, Softmax)             -> 10
 */
import weightsJson from "./model/weights.json";

export type ModelWeights = {
  version: string;
  architecture: {
    input: string;
    layers: Array<Record<string, string | number>>;
    params: number;
  };
  conv1: {
    weights: number[];
    bias: number[];
    filters: number;
    kernel: number;
  };
  conv2: {
    weights: number[];
    bias: number[];
    filters: number;
    inChannels: number;
    kernel: number;
  };
  fc1: {
    weights: number[];
    bias: number[];
    units: number;
    inputSize: number;
  };
  fc2: {
    weights: number[];
    bias: number[];
    units: number;
    inputSize: number;
  };
};

const weights = weightsJson as unknown as ModelWeights;

export const INPUT_SIZE = 28;
export const FILTER_COUNT = weights.conv1.filters; // 16
export const KERNEL_1 = weights.conv1.kernel; // 5
export const CONV1_SIZE = INPUT_SIZE - KERNEL_1 + 1; // 24
export const POOL1_SIZE = CONV1_SIZE / 2; // 12

export const FILTER2_COUNT = weights.conv2.filters; // 32
export const KERNEL_2 = weights.conv2.kernel; // 3
export const CONV2_SIZE = POOL1_SIZE - KERNEL_2 + 1; // 10
export const POOL2_SIZE = CONV2_SIZE / 2; // 5

export const FLAT_SIZE = FILTER2_COUNT * POOL2_SIZE * POOL2_SIZE; // 800
export const HIDDEN_UNITS = weights.fc1.units; // 64
export const CLASS_COUNT = weights.fc2.units; // 10

const conv1Weights = Float32Array.from(weights.conv1.weights);
const conv1Bias = Float32Array.from(weights.conv1.bias);
const conv2Weights = Float32Array.from(weights.conv2.weights);
const conv2Bias = Float32Array.from(weights.conv2.bias);
const fc1Weights = Float32Array.from(weights.fc1.weights);
const fc1Bias = Float32Array.from(weights.fc1.bias);
const fc2Weights = Float32Array.from(weights.fc2.weights);
const fc2Bias = Float32Array.from(weights.fc2.bias);

const input = new Float32Array(INPUT_SIZE * INPUT_SIZE);
const conv1Act = new Float32Array(FILTER_COUNT * CONV1_SIZE * CONV1_SIZE);
const pool1Act = new Float32Array(FILTER_COUNT * POOL1_SIZE * POOL1_SIZE);
const conv2Act = new Float32Array(FILTER2_COUNT * CONV2_SIZE * CONV2_SIZE);
const pool2Act = new Float32Array(FLAT_SIZE);
const hiddenAct = new Float32Array(HIDDEN_UNITS);

export type InferenceResult = {
  digit: number;
  confidence: number;
  probabilities: number[];
  logits: number[];
  featureMaps: number[][];
  inferenceMs: number;
  modelVersion: string;
};

export function getModelWeights(): ModelWeights {
  return weights;
}

function forwardSingle(x: Float32Array): { logits: number[]; probabilities: number[] } {
  // Conv1: 28x28x1 -> 24x24x16 (5x5 valid, ReLU)
  for (let f = 0; f < FILTER_COUNT; f++) {
    const wBase = f * KERNEL_1 * KERNEL_1;
    const b = conv1Bias[f];
    const outBase = f * CONV1_SIZE * CONV1_SIZE;
    for (let oy = 0; oy < CONV1_SIZE; oy++) {
      const rowBase = oy * INPUT_SIZE;
      for (let ox = 0; ox < CONV1_SIZE; ox++) {
        let sum = b;
        for (let ky = 0; ky < KERNEL_1; ky++) {
          const xBase = rowBase + ky * INPUT_SIZE + ox;
          const wRow = wBase + ky * KERNEL_1;
          for (let kx = 0; kx < KERNEL_1; kx++) {
            sum += x[xBase + kx] * conv1Weights[wRow + kx];
          }
        }
        conv1Act[outBase + oy * CONV1_SIZE + ox] = sum > 0 ? sum : 0;
      }
    }
  }

  // MaxPool1: 24x24x16 -> 12x12x16
  for (let f = 0; f < FILTER_COUNT; f++) {
    const cBase = f * CONV1_SIZE * CONV1_SIZE;
    const pBase = f * POOL1_SIZE * POOL1_SIZE;
    for (let py = 0; py < POOL1_SIZE; py++) {
      for (let px = 0; px < POOL1_SIZE; px++) {
        const i0 = cBase + py * 2 * CONV1_SIZE + px * 2;
        const v0 = conv1Act[i0];
        const v1 = conv1Act[i0 + 1];
        const v2 = conv1Act[i0 + CONV1_SIZE];
        const v3 = conv1Act[i0 + CONV1_SIZE + 1];
        pool1Act[pBase + py * POOL1_SIZE + px] = Math.max(v0, v1, v2, v3);
      }
    }
  }

  // Conv2: 12x12x16 -> 10x10x32 (3x3 valid, ReLU)
  for (let f2 = 0; f2 < FILTER2_COUNT; f2++) {
    const outBase = f2 * CONV2_SIZE * CONV2_SIZE;
    const b = conv2Bias[f2];
    for (let i = 0; i < CONV2_SIZE * CONV2_SIZE; i++) conv2Act[outBase + i] = b;

    for (let f1 = 0; f1 < FILTER_COUNT; f1++) {
      const inBase = f1 * POOL1_SIZE * POOL1_SIZE;
      const wBase = (f2 * FILTER_COUNT + f1) * KERNEL_2 * KERNEL_2;
      for (let oy = 0; oy < CONV2_SIZE; oy++) {
        for (let ox = 0; ox < CONV2_SIZE; ox++) {
          let sum = 0;
          for (let ky = 0; ky < KERNEL_2; ky++) {
            const inRow = inBase + (oy + ky) * POOL1_SIZE + ox;
            const wRow = wBase + ky * KERNEL_2;
            sum +=
              pool1Act[inRow] * conv2Weights[wRow] +
              pool1Act[inRow + 1] * conv2Weights[wRow + 1] +
              pool1Act[inRow + 2] * conv2Weights[wRow + 2];
          }
          conv2Act[outBase + oy * CONV2_SIZE + ox] += sum;
        }
      }
    }

    for (let i = 0; i < CONV2_SIZE * CONV2_SIZE; i++) {
      if (conv2Act[outBase + i] < 0) conv2Act[outBase + i] = 0;
    }
  }

  // MaxPool2: 10x10x32 -> 5x5x32 (800)
  for (let f2 = 0; f2 < FILTER2_COUNT; f2++) {
    const cBase = f2 * CONV2_SIZE * CONV2_SIZE;
    const pBase = f2 * POOL2_SIZE * POOL2_SIZE;
    for (let py = 0; py < POOL2_SIZE; py++) {
      for (let px = 0; px < POOL2_SIZE; px++) {
        const i0 = cBase + py * 2 * CONV2_SIZE + px * 2;
        const v0 = conv2Act[i0];
        const v1 = conv2Act[i0 + 1];
        const v2 = conv2Act[i0 + CONV2_SIZE];
        const v3 = conv2Act[i0 + CONV2_SIZE + 1];
        pool2Act[pBase + py * POOL2_SIZE + px] = Math.max(v0, v1, v2, v3);
      }
    }
  }

  // Dense1: 800 -> 64 (ReLU), fc1Weights stored as [FLAT_SIZE * HIDDEN_UNITS]
  for (let h = 0; h < HIDDEN_UNITS; h++) hiddenAct[h] = fc1Bias[h];
  for (let i = 0; i < FLAT_SIZE; i++) {
    const v = pool2Act[i];
    if (v === 0) continue;
    const wBase = i * HIDDEN_UNITS;
    for (let h = 0; h < HIDDEN_UNITS; h++) {
      hiddenAct[h] += v * fc1Weights[wBase + h];
    }
  }
  for (let h = 0; h < HIDDEN_UNITS; h++) {
    if (hiddenAct[h] < 0) hiddenAct[h] = 0;
  }

  // Dense2: 64 -> 10 + Softmax
  let maxLogit = -Infinity;
  const logits = new Array<number>(CLASS_COUNT).fill(0);
  for (let c = 0; c < CLASS_COUNT; c++) {
    const wBase = c * HIDDEN_UNITS;
    let sum = fc2Bias[c];
    for (let h = 0; h < HIDDEN_UNITS; h++) {
      sum += fc2Weights[wBase + h] * hiddenAct[h];
    }
    logits[c] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }

  let denom = 0;
  const probabilities = logits.map((logit) => {
    const e = Math.exp(logit - maxLogit);
    denom += e;
    return e;
  });
  for (let c = 0; c < CLASS_COUNT; c++) {
    probabilities[c] /= denom;
  }

  return { logits, probabilities };
}

function shiftInput(
  rawPixels: ArrayLike<number>,
  shiftX: number,
  shiftY: number,
  out: Float32Array,
) {
  out.fill(0);
  for (let y = 0; y < INPUT_SIZE; y++) {
    const sy = y - shiftY;
    if (sy < 0 || sy >= INPUT_SIZE) continue;
    const dstRow = y * INPUT_SIZE;
    const srcRow = sy * INPUT_SIZE;
    for (let x = 0; x < INPUT_SIZE; x++) {
      const sx = x - shiftX;
      if (sx < 0 || sx >= INPUT_SIZE) continue;
      out[dstRow + x] = rawPixels[srcRow + sx] / 255;
    }
  }
}

export function predict(
  inputPixels: ArrayLike<number>,
  options: { tta?: boolean } = {},
): InferenceResult {
  const startedAt = performance.now();
  if (inputPixels.length !== INPUT_SIZE * INPUT_SIZE) {
    throw new Error(`Expected ${INPUT_SIZE * INPUT_SIZE} pixels, received ${inputPixels.length}.`);
  }

  // Center pass first (also populates pool1Act for the 16 feature map visualizations)
  shiftInput(inputPixels, 0, 0, input);
  const center = forwardSingle(input);

  const featureMaps: number[][] = [];
  for (let f = 0; f < FILTER_COUNT; f++) {
    const base = f * POOL1_SIZE * POOL1_SIZE;
    const map = Array.from(pool1Act.subarray(base, base + POOL1_SIZE * POOL1_SIZE));
    const max = Math.max(...map, 1e-6);
    featureMaps.push(map.map((value) => Number((value / max).toFixed(3))));
  }

  let probabilities = center.probabilities;

  if (options.tta) {
    // 5-crop Test-Time Augmentation: center + 4 cardinal 1px shifts
    const shifts: Array<[number, number, number]> = [
      [0, 0, 0.44],
      [-1, 0, 0.14],
      [1, 0, 0.14],
      [0, -1, 0.14],
      [0, 1, 0.14],
    ];
    const blended = new Array<number>(CLASS_COUNT).fill(0);
    for (let c = 0; c < CLASS_COUNT; c++) {
      blended[c] = center.probabilities[c] * shifts[0][2];
    }
    for (let s = 1; s < shifts.length; s++) {
      const [sx, sy, weight] = shifts[s];
      shiftInput(inputPixels, sx, sy, input);
      const pass = forwardSingle(input);
      for (let c = 0; c < CLASS_COUNT; c++) {
        blended[c] += pass.probabilities[c] * weight;
      }
    }
    const sum = blended.reduce((a, b) => a + b, 0) || 1;
    probabilities = blended.map((p) => p / sum);
  }

  let digit = 0;
  for (let c = 1; c < probabilities.length; c++) {
    if (probabilities[c] > probabilities[digit]) digit = c;
  }

  return {
    digit,
    confidence: probabilities[digit],
    probabilities: probabilities.map((p) => Number(p.toFixed(6))),
    logits: center.logits.map((l) => Number(l.toFixed(4))),
    featureMaps,
    inferenceMs: Number((performance.now() - startedAt).toFixed(3)),
    modelVersion: weights.version,
  };
}

export const MODEL_SUMMARY = {
  version: weights.version,
  parameters: weights.architecture.params,
  filters: weights.conv1.filters,
  filters2: weights.conv2.filters,
  hiddenUnits: weights.fc1.units,
  layers: weights.architecture.layers,
  poolSize: POOL1_SIZE,
  featureSize: FLAT_SIZE,
};
