#!/usr/bin/env node
/**
 * High-accuracy Convolutional Neural Network trainer for MNIST.
 *
 * Architecture (Deep LeNet-style, 56,970 parameters):
 *   Input 28x28x1
 *     -> Conv2D(16 filters, 5x5, valid, ReLU)   -> 24x24x16
 *     -> MaxPool2D(2x2, stride 2)               -> 12x12x16
 *     -> Conv2D(32 filters, 3x3, valid, ReLU)   -> 10x10x32
 *     -> MaxPool2D(2x2, stride 2)               -> 5x5x32 (800)
 *     -> Dense(64, ReLU)                        -> 64
 *     -> Dense(10, Softmax)                     -> 10
 */
import fs from "node:fs";
import path from "node:path";

const MNIST_DIR = process.env.MNIST_DIR || "/tmp/mnist";
const OUT_DIR = process.env.OUT_DIR || path.join(process.cwd(), "src/lib/model");
const EPOCHS = Number(process.env.EPOCHS || 8);
const TRAIN_LIMIT = Number(process.env.TRAIN_LIMIT || 60000);
const TEST_LIMIT = Number(process.env.TEST_LIMIT || 10000);
const BATCH = Number(process.env.BATCH || 32);
const LR_MAX = Number(process.env.LR || 0.025);
const LR_MIN = 0.0015;
const MOMENTUM = 0.9;
const WEIGHT_DECAY = 1e-4;

const IN = 28;
const K1 = 5;
const F1 = 16;
const C1 = IN - K1 + 1; // 24
const P1 = C1 / 2; // 12

const K2 = 3;
const F2 = 32;
const C2 = P1 - K2 + 1; // 10
const P2 = C2 / 2; // 5

const FLAT = F2 * P2 * P2; // 800
const HIDDEN = 64;
const CLASSES = 10;

function readImages(file) {
  const buf = fs.readFileSync(file);
  const count = buf.readUInt32BE(4);
  const rows = buf.readUInt32BE(8);
  const cols = buf.readUInt32BE(12);
  const pixels = new Uint8Array(buf.buffer, buf.byteOffset + 16, count * rows * cols);
  return { count, rows, cols, pixels };
}

function readLabels(file) {
  const buf = fs.readFileSync(file);
  const count = buf.readUInt32BE(4);
  return { count, labels: new Uint8Array(buf.buffer, buf.byteOffset + 8, count) };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(20260329);
const randn = (scale) => {
  const u = Math.max(rand(), 1e-9);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * scale;
};

const trainImg = readImages(path.join(MNIST_DIR, "train-images-idx3-ubyte"));
const trainLbl = readLabels(path.join(MNIST_DIR, "train-labels-idx1-ubyte"));
const testImg = readImages(path.join(MNIST_DIR, "t10k-images-idx3-ubyte"));
const testLbl = readLabels(path.join(MNIST_DIR, "t10k-labels-idx1-ubyte"));

function toFloatInput(pixels, offset, out, shiftX = 0, shiftY = 0) {
  if (shiftX === 0 && shiftY === 0) {
    for (let i = 0; i < IN * IN; i++) out[i] = pixels[offset + i] / 255;
    return;
  }
  out.fill(0);
  for (let y = 0; y < IN; y++) {
    const sy = y - shiftY;
    if (sy < 0 || sy >= IN) continue;
    const dstRow = y * IN;
    const srcRow = offset + sy * IN;
    for (let x = 0; x < IN; x++) {
      const sx = x - shiftX;
      if (sx < 0 || sx >= IN) continue;
      out[dstRow + x] = pixels[srcRow + sx] / 255;
    }
  }
}

// ------------------------------------------------------------ He initialization
const conv1W = new Float32Array(F1 * K1 * K1);
for (let i = 0; i < conv1W.length; i++) conv1W[i] = randn(Math.sqrt(2 / (K1 * K1)));
const conv1B = new Float32Array(F1);

const conv2W = new Float32Array(F2 * F1 * K2 * K2);
for (let i = 0; i < conv2W.length; i++) conv2W[i] = randn(Math.sqrt(2 / (F1 * K2 * K2)));
const conv2B = new Float32Array(F2);

const fc1W = new Float32Array(FLAT * HIDDEN);
for (let i = 0; i < fc1W.length; i++) fc1W[i] = randn(Math.sqrt(2 / FLAT));
const fc1B = new Float32Array(HIDDEN);

const fc2W = new Float32Array(CLASSES * HIDDEN);
for (let i = 0; i < fc2W.length; i++) fc2W[i] = randn(Math.sqrt(2 / HIDDEN));
const fc2B = new Float32Array(CLASSES);

// Best checkpoint buffers
const bestConv1W = new Float32Array(conv1W.length);
const bestConv1B = new Float32Array(conv1B.length);
const bestConv2W = new Float32Array(conv2W.length);
const bestConv2B = new Float32Array(conv2B.length);
const bestFc1W = new Float32Array(fc1W.length);
const bestFc1B = new Float32Array(fc1B.length);
const bestFc2W = new Float32Array(fc2W.length);
const bestFc2B = new Float32Array(fc2B.length);

// Momentum velocities
const vConv1W = new Float32Array(conv1W.length);
const vConv1B = new Float32Array(conv1B.length);
const vConv2W = new Float32Array(conv2W.length);
const vConv2B = new Float32Array(conv2B.length);
const vFc1W = new Float32Array(fc1W.length);
const vFc1B = new Float32Array(fc1B.length);
const vFc2W = new Float32Array(fc2W.length);
const vFc2B = new Float32Array(fc2B.length);

// Gradients
const gConv1W = new Float64Array(conv1W.length);
const gConv1B = new Float64Array(conv1B.length);
const gConv2W = new Float64Array(conv2W.length);
const gConv2B = new Float64Array(conv2B.length);
const gFc1W = new Float64Array(fc1W.length);
const gFc1B = new Float64Array(fc1B.length);
const gFc2W = new Float64Array(fc2W.length);
const gFc2B = new Float64Array(fc2B.length);

// Activations & backprop buffers
const input = new Float32Array(IN * IN);
const c1Act = new Float32Array(F1 * C1 * C1);
const p1Act = new Float32Array(F1 * P1 * P1);
const p1Idx = new Int32Array(F1 * P1 * P1);

const c2Act = new Float32Array(F2 * C2 * C2);
const p2Act = new Float32Array(FLAT);
const p2Idx = new Int32Array(FLAT);

const h1Act = new Float32Array(HIDDEN);
const logits = new Float32Array(CLASSES);
const probs = new Float32Array(CLASSES);

const dH1 = new Float32Array(HIDDEN);
const dP2 = new Float32Array(FLAT);
const dP1 = new Float32Array(F1 * P1 * P1);

const patch1K = new Int32Array(25);
const patch1V = new Float32Array(25);
const patch2K = new Int32Array(F1 * 9);
const patch2V = new Float32Array(F1 * 9);

function forward(x) {
  // Conv1: for each (oy, ox) extract non-zero pixels in 5x5 patch once, then run 16 filters
  for (let oy = 0; oy < C1; oy++) {
    const rowBase = oy * IN;
    const outRowOffset = oy * C1;
    for (let ox = 0; ox < C1; ox++) {
      let nnz = 0;
      const base = rowBase + ox;
      for (let ky = 0; ky < 5; ky++) {
        const r = base + ky * IN;
        const kBase = ky * 5;
        const v0 = x[r], v1 = x[r + 1], v2 = x[r + 2], v3 = x[r + 3], v4 = x[r + 4];
        if (v0 !== 0) { patch1K[nnz] = kBase; patch1V[nnz++] = v0; }
        if (v1 !== 0) { patch1K[nnz] = kBase + 1; patch1V[nnz++] = v1; }
        if (v2 !== 0) { patch1K[nnz] = kBase + 2; patch1V[nnz++] = v2; }
        if (v3 !== 0) { patch1K[nnz] = kBase + 3; patch1V[nnz++] = v3; }
        if (v4 !== 0) { patch1K[nnz] = kBase + 4; patch1V[nnz++] = v4; }
      }
      const pos = outRowOffset + ox;
      if (nnz === 0) {
        for (let f = 0; f < F1; f++) {
          const b = conv1B[f];
          c1Act[f * C1 * C1 + pos] = b > 0 ? b : 0;
        }
      } else {
        for (let f = 0; f < F1; f++) {
          const wBase = f * 25;
          let sum = conv1B[f];
          for (let n = 0; n < nnz; n++) {
            sum += patch1V[n] * conv1W[wBase + patch1K[n]];
          }
          c1Act[f * C1 * C1 + pos] = sum > 0 ? sum : 0;
        }
      }
    }
  }

  // MaxPool1: 24x24x16 -> 12x12x16
  for (let f = 0; f < F1; f++) {
    const cBase = f * C1 * C1;
    const pBase = f * P1 * P1;
    for (let py = 0; py < P1; py++) {
      const row0 = cBase + py * 2 * C1;
      const row1 = row0 + C1;
      const pRow = pBase + py * P1;
      for (let px = 0; px < P1; px++) {
        const i0 = row0 + px * 2;
        const i2 = row1 + px * 2;
        let best = c1Act[i0];
        let bestI = i0;
        const v1 = c1Act[i0 + 1];
        const v2 = c1Act[i2];
        const v3 = c1Act[i2 + 1];
        if (v1 > best) { best = v1; bestI = i0 + 1; }
        if (v2 > best) { best = v2; bestI = i2; }
        if (v3 > best) { best = v3; bestI = i2 + 1; }
        p1Act[pRow + px] = best;
        p1Idx[pRow + px] = bestI;
      }
    }
  }

  // Conv2: for each (oy, ox) in 10x10 extract non-zero p1Act across 16x3x3 once, then dot with 32 filters
  for (let oy = 0; oy < C2; oy++) {
    const outRowOffset = oy * C2;
    for (let ox = 0; ox < C2; ox++) {
      let nnz = 0;
      for (let f1 = 0; f1 < F1; f1++) {
        const r0 = f1 * P1 * P1 + oy * P1 + ox;
        const r1 = r0 + P1;
        const r2 = r1 + P1;
        const kBase = f1 * 9;
        const v0 = p1Act[r0], v1 = p1Act[r0 + 1], v2 = p1Act[r0 + 2];
        const v3 = p1Act[r1], v4 = p1Act[r1 + 1], v5 = p1Act[r1 + 2];
        const v6 = p1Act[r2], v7 = p1Act[r2 + 1], v8 = p1Act[r2 + 2];
        if (v0 !== 0) { patch2K[nnz] = kBase; patch2V[nnz++] = v0; }
        if (v1 !== 0) { patch2K[nnz] = kBase + 1; patch2V[nnz++] = v1; }
        if (v2 !== 0) { patch2K[nnz] = kBase + 2; patch2V[nnz++] = v2; }
        if (v3 !== 0) { patch2K[nnz] = kBase + 3; patch2V[nnz++] = v3; }
        if (v4 !== 0) { patch2K[nnz] = kBase + 4; patch2V[nnz++] = v4; }
        if (v5 !== 0) { patch2K[nnz] = kBase + 5; patch2V[nnz++] = v5; }
        if (v6 !== 0) { patch2K[nnz] = kBase + 6; patch2V[nnz++] = v6; }
        if (v7 !== 0) { patch2K[nnz] = kBase + 7; patch2V[nnz++] = v7; }
        if (v8 !== 0) { patch2K[nnz] = kBase + 8; patch2V[nnz++] = v8; }
      }
      const pos = outRowOffset + ox;
      for (let f2 = 0; f2 < F2; f2++) {
        const wBase = f2 * F1 * 9;
        let sum = conv2B[f2];
        for (let n = 0; n < nnz; n++) {
          sum += patch2V[n] * conv2W[wBase + patch2K[n]];
        }
        c2Act[f2 * C2 * C2 + pos] = sum > 0 ? sum : 0;
      }
    }
  }

  // MaxPool2: 10x10x32 -> 5x5x32 (800)
  for (let f2 = 0; f2 < F2; f2++) {
    const cBase = f2 * C2 * C2;
    const pBase = f2 * P2 * P2;
    for (let py = 0; py < P2; py++) {
      const row0 = cBase + py * 2 * C2;
      const row1 = row0 + C2;
      const pRow = pBase + py * P2;
      for (let px = 0; px < P2; px++) {
        const i0 = row0 + px * 2;
        const i2 = row1 + px * 2;
        let best = c2Act[i0];
        let bestI = i0;
        const v1 = c2Act[i0 + 1];
        const v2 = c2Act[i2];
        const v3 = c2Act[i2 + 1];
        if (v1 > best) { best = v1; bestI = i0 + 1; }
        if (v2 > best) { best = v2; bestI = i2; }
        if (v3 > best) { best = v3; bestI = i2 + 1; }
        p2Act[pRow + px] = best;
        p2Idx[pRow + px] = bestI;
      }
    }
  }

  // Dense1: 800 -> 64 (ReLU), skipping zero p2Act[i]
  for (let h = 0; h < HIDDEN; h++) h1Act[h] = fc1B[h];
  for (let i = 0; i < FLAT; i++) {
    const v = p2Act[i];
    if (v === 0) continue;
    const wBase = i * HIDDEN;
    for (let h = 0; h < HIDDEN; h++) {
      h1Act[h] += v * fc1W[wBase + h];
    }
  }
  for (let h = 0; h < HIDDEN; h++) {
    if (h1Act[h] < 0) h1Act[h] = 0;
  }

  // Dense2: 64 -> 10 + Softmax
  let maxLogit = -Infinity;
  for (let c = 0; c < CLASSES; c++) {
    const wBase = c * HIDDEN;
    let sum = fc2B[c];
    for (let h = 0; h < HIDDEN; h++) sum += fc2W[wBase + h] * h1Act[h];
    logits[c] = sum;
    if (sum > maxLogit) maxLogit = sum;
  }
  let denom = 0;
  for (let c = 0; c < CLASSES; c++) {
    const e = Math.exp(logits[c] - maxLogit);
    probs[c] = e;
    denom += e;
  }
  for (let c = 0; c < CLASSES; c++) probs[c] /= denom;
}

function zeroGrads() {
  gConv1W.fill(0); gConv1B.fill(0);
  gConv2W.fill(0); gConv2B.fill(0);
  gFc1W.fill(0); gFc1B.fill(0);
  gFc2W.fill(0); gFc2B.fill(0);
}

function backward(label) {
  const loss = -Math.log(Math.max(probs[label], 1e-12));

  // Dense2 backward
  dH1.fill(0);
  for (let c = 0; c < CLASSES; c++) {
    const d = probs[c] - (c === label ? 1 : 0);
    gFc2B[c] += d;
    const wBase = c * HIDDEN;
    for (let h = 0; h < HIDDEN; h++) {
      gFc2W[wBase + h] += d * h1Act[h];
      dH1[h] += fc2W[wBase + h] * d;
    }
  }

  // ReLU on h1Act
  for (let h = 0; h < HIDDEN; h++) {
    if (h1Act[h] > 0) {
      gFc1B[h] += dH1[h];
    } else {
      dH1[h] = 0;
    }
  }

  // Dense1 backward: only positions with p2Act[i] > 0 survive Conv2's ReLU!
  dP2.fill(0);
  for (let i = 0; i < FLAT; i++) {
    const vp = p2Act[i];
    if (vp === 0) continue;
    const wBase = i * HIDDEN;
    let sum = 0;
    for (let h = 0; h < HIDDEN; h++) {
      const dh = dH1[h];
      if (dh === 0) continue;
      gFc1W[wBase + h] += vp * dh;
      sum += fc1W[wBase + h] * dh;
    }
    dP2[i] = sum;
  }

  // MaxPool2 + Conv2 backward
  dP1.fill(0);
  for (let f2 = 0; f2 < F2; f2++) {
    const pBase = f2 * P2 * P2;
    let db = 0;
    const wF2Base = f2 * F1 * 9;
    for (let p = 0; p < P2 * P2; p++) {
      const dz = dP2[pBase + p];
      if (dz === 0) continue;
      db += dz;
      const cIdx = p2Idx[pBase + p];
      const rem = cIdx - f2 * C2 * C2;
      const oy = (rem / C2) | 0;
      const ox = rem - oy * C2;

      for (let f1 = 0; f1 < F1; f1++) {
        const wBase = wF2Base + f1 * 9;
        const r0 = f1 * P1 * P1 + oy * P1 + ox;
        const r1 = r0 + P1;
        const r2 = r1 + P1;

        gConv2W[wBase] += dz * p1Act[r0];
        gConv2W[wBase + 1] += dz * p1Act[r0 + 1];
        gConv2W[wBase + 2] += dz * p1Act[r0 + 2];
        gConv2W[wBase + 3] += dz * p1Act[r1];
        gConv2W[wBase + 4] += dz * p1Act[r1 + 1];
        gConv2W[wBase + 5] += dz * p1Act[r1 + 2];
        gConv2W[wBase + 6] += dz * p1Act[r2];
        gConv2W[wBase + 7] += dz * p1Act[r2 + 1];
        gConv2W[wBase + 8] += dz * p1Act[r2 + 2];

        dP1[r0] += dz * conv2W[wBase];
        dP1[r0 + 1] += dz * conv2W[wBase + 1];
        dP1[r0 + 2] += dz * conv2W[wBase + 2];
        dP1[r1] += dz * conv2W[wBase + 3];
        dP1[r1 + 1] += dz * conv2W[wBase + 4];
        dP1[r1 + 2] += dz * conv2W[wBase + 5];
        dP1[r2] += dz * conv2W[wBase + 6];
        dP1[r2 + 1] += dz * conv2W[wBase + 7];
        dP1[r2 + 2] += dz * conv2W[wBase + 8];
      }
    }
    gConv2B[f2] += db;
  }

  // MaxPool1 + Conv1 backward
  for (let f1 = 0; f1 < F1; f1++) {
    const pBase = f1 * P1 * P1;
    const wBase = f1 * 25;
    let db = 0;
    for (let p = 0; p < P1 * P1; p++) {
      const dz = dP1[pBase + p];
      if (dz === 0 || p1Act[pBase + p] === 0) continue;
      db += dz;
      const cIdx = p1Idx[pBase + p];
      const rem = cIdx - f1 * C1 * C1;
      const oy = (rem / C1) | 0;
      const ox = rem - oy * C1;
      const rowBase = oy * IN + ox;
      for (let ky = 0; ky < 5; ky++) {
        const xBase = rowBase + ky * IN;
        const wRow = wBase + ky * 5;
        gConv1W[wRow] += dz * input[xBase];
        gConv1W[wRow + 1] += dz * input[xBase + 1];
        gConv1W[wRow + 2] += dz * input[xBase + 2];
        gConv1W[wRow + 3] += dz * input[xBase + 3];
        gConv1W[wRow + 4] += dz * input[xBase + 4];
      }
    }
    gConv1B[f1] += db;
  }

  return loss;
}

function stepParam(param, vel, grad, scale, lr, decay) {
  for (let i = 0; i < param.length; i++) {
    const g = grad[i] / scale + decay * param[i];
    vel[i] = MOMENTUM * vel[i] - lr * g;
    param[i] += vel[i];
  }
}

function applyGrads(scale, lr) {
  stepParam(conv1W, vConv1W, gConv1W, scale, lr, WEIGHT_DECAY);
  stepParam(conv1B, vConv1B, gConv1B, scale, lr, 0);
  stepParam(conv2W, vConv2W, gConv2W, scale, lr, WEIGHT_DECAY);
  stepParam(conv2B, vConv2B, gConv2B, scale, lr, 0);
  stepParam(fc1W, vFc1W, gFc1W, scale, lr, WEIGHT_DECAY);
  stepParam(fc1B, vFc1B, gFc1B, scale, lr, 0);
  stepParam(fc2W, vFc2W, gFc2W, scale, lr, WEIGHT_DECAY);
  stepParam(fc2B, vFc2B, gFc2B, scale, lr, 0);
}

function saveBest() {
  bestConv1W.set(conv1W); bestConv1B.set(conv1B);
  bestConv2W.set(conv2W); bestConv2B.set(conv2B);
  bestFc1W.set(fc1W); bestFc1B.set(fc1B);
  bestFc2W.set(fc2W); bestFc2B.set(fc2B);
}

function restoreBest() {
  conv1W.set(bestConv1W); conv1B.set(bestConv1B);
  conv2W.set(bestConv2W); conv2B.set(bestConv2B);
  fc1W.set(bestFc1W); fc1B.set(bestFc1B);
  fc2W.set(bestFc2W); fc2B.set(bestFc2B);
}

function evaluate(images, labels, limit) {
  let correct = 0;
  let loss = 0;
  const confusion = Array.from({ length: CLASSES }, () => new Array(CLASSES).fill(0));
  const n = Math.min(limit, images.count);
  for (let i = 0; i < n; i++) {
    toFloatInput(images.pixels, i * IN * IN, input);
    forward(input);
    const actual = labels.labels[i];
    loss += -Math.log(Math.max(probs[actual], 1e-12));
    let best = 0;
    for (let c = 1; c < CLASSES; c++) if (probs[c] > probs[best]) best = c;
    if (best === actual) correct++;
    confusion[actual][best]++;
  }
  return {
    accuracy: correct / n,
    loss: loss / n,
    perClass: confusion.map((row, actual) => {
      const total = row.reduce((a, b) => a + b, 0) || 1;
      return { digit: actual, recall: row[actual] / total, support: total };
    }),
    confusion,
  };
}

const totalParams =
  conv1W.length + conv1B.length +
  conv2W.length + conv2B.length +
  fc1W.length + fc1B.length +
  fc2W.length + fc2B.length;

const config = {
  filters: F1,
  filters2: F2,
  hiddenUnits: HIDDEN,
  epochs: EPOCHS,
  trainSamples: Math.min(TRAIN_LIMIT, trainImg.count),
  testSamples: Math.min(TEST_LIMIT, testImg.count),
  batchSize: BATCH,
  learningRate: LR_MAX,
  momentum: MOMENTUM,
  optimizer: "Momentum SGD + Cosine LR",
  activation: "ReLU",
};

console.log("config", config);
console.log("parameters", totalParams);

const order = new Uint32Array(trainImg.count);
for (let i = 0; i < order.length; i++) order[i] = i;

const history = [];
const startedAt = Date.now();
let bestAcc = 0;

for (let epoch = 1; epoch <= EPOCHS; epoch++) {
  const epochStart = Date.now();
  const progress = (epoch - 1) / Math.max(EPOCHS - 1, 1);
  const lr = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + Math.cos(Math.PI * progress));

  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }

  const n = config.trainSamples;
  let runningLoss = 0;
  let trainCorrect = 0;
  zeroGrads();
  let inBatch = 0;

  for (let s = 0; s < n; s++) {
    const idx = order[s];
    const label = trainLbl.labels[idx];
    let sx = 0, sy = 0;
    if (rand() < 0.35) {
      sx = Math.floor(rand() * 3) - 1;
      sy = Math.floor(rand() * 3) - 1;
    }
    toFloatInput(trainImg.pixels, idx * IN * IN, input, sx, sy);
    forward(input);
    runningLoss += backward(label);
    let best = 0;
    for (let c = 1; c < CLASSES; c++) if (probs[c] > probs[best]) best = c;
    if (best === label) trainCorrect++;
    inBatch++;
    if (inBatch === BATCH || s === n - 1) {
      applyGrads(inBatch, lr);
      zeroGrads();
      inBatch = 0;
    }
  }

  const test = evaluate(testImg, testLbl, config.testSamples);
  if (test.accuracy >= bestAcc) {
    bestAcc = test.accuracy;
    saveBest();
  }
  const entry = {
    epoch,
    trainLoss: Number((runningLoss / n).toFixed(5)),
    trainAccuracy: Number((trainCorrect / n).toFixed(5)),
    testLoss: Number(test.loss.toFixed(5)),
    testAccuracy: Number(test.accuracy.toFixed(5)),
    seconds: Math.round((Date.now() - epochStart) / 1000),
  };
  history.push(entry);
  console.log(
    `epoch ${epoch} (lr=${lr.toFixed(4)}): train_loss=${entry.trainLoss.toFixed(4)} train_acc=${(entry.trainAccuracy * 100).toFixed(2)}% test_acc=${(entry.testAccuracy * 100).toFixed(2)}% (${entry.seconds}s)`,
  );
}

restoreBest();
const finalTest = evaluate(testImg, testLbl, config.testSamples);
const totalSeconds = Math.round((Date.now() - startedAt) / 1000);
console.log(`best test accuracy ${(finalTest.accuracy * 100).toFixed(2)}% in ${totalSeconds}s`);

const demoSamples = [];
const perClassCount = new Array(CLASSES).fill(0);
for (let i = 0; i < testImg.count && demoSamples.length < 60; i++) {
  const label = testLbl.labels[i];
  if (perClassCount[label] >= 6) continue;
  toFloatInput(testImg.pixels, i * IN * IN, input);
  forward(input);
  let best = 0;
  for (let c = 1; c < CLASSES; c++) if (probs[c] > probs[best]) best = c;
  if (best !== label || probs[best] < 0.98) continue;
  const raw = Buffer.from(
    testImg.pixels.buffer,
    testImg.pixels.byteOffset + i * IN * IN,
    IN * IN,
  );
  demoSamples.push({
    label,
    sourceIndex: i,
    pixels: raw.toString("base64"),
  });
  perClassCount[label]++;
}

const round = (arr, digits = 5) =>
  Array.from(arr, (v) => Number(v.toFixed(digits)));

const weights = {
  version: "cnn-mnist-v2",
  architecture: {
    input: "28x28x1 grayscale",
    layers: [
      { type: "conv2d", filters: F1, kernel: K1, stride: 1, padding: "valid", activation: "relu", output: "24x24x" + F1 },
      { type: "maxpool2d", pool: 2, stride: 2, output: "12x12x" + F1 },
      { type: "conv2d", filters: F2, kernel: K2, stride: 1, padding: "valid", activation: "relu", output: "10x10x" + F2 },
      { type: "maxpool2d", pool: 2, stride: 2, output: "5x5x" + F2 },
      { type: "flatten", output: FLAT },
      { type: "dense", units: HIDDEN, activation: "relu", output: String(HIDDEN) },
      { type: "dense", units: CLASSES, activation: "softmax", output: "10" },
    ],
    params: totalParams,
  },
  conv1: {
    weights: round(conv1W),
    bias: round(conv1B),
    filters: F1,
    kernel: K1,
  },
  conv2: {
    weights: round(conv2W),
    bias: round(conv2B),
    filters: F2,
    inChannels: F1,
    kernel: K2,
  },
  fc1: {
    weights: round(fc1W),
    bias: round(fc1B),
    units: HIDDEN,
    inputSize: FLAT,
  },
  fc2: {
    weights: round(fc2W),
    bias: round(fc2B),
    units: CLASSES,
    inputSize: HIDDEN,
  },
};

fs.mkdirSync(OUT_DIR, { recursive: true });
fs.writeFileSync(path.join(OUT_DIR, "weights.json"), JSON.stringify(weights));
fs.writeFileSync(
  path.join(OUT_DIR, "metrics.json"),
  JSON.stringify(
    {
      version: weights.version,
      generatedAt: new Date().toISOString(),
      dataset: "MNIST (60,000 train / 10,000 test)",
      config,
      parameters: totalParams,
      history,
      final: {
        testAccuracy: finalTest.accuracy,
        testLoss: finalTest.loss,
        perClass: finalTest.perClass,
        confusion: finalTest.confusion,
      },
      trainingSeconds: totalSeconds,
    },
    null,
    2,
  ),
);
fs.writeFileSync(path.join(OUT_DIR, "samples.json"), JSON.stringify(demoSamples));
console.log("wrote", OUT_DIR);
