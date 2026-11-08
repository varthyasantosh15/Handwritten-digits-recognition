export type PredictionSource = "draw" | "upload" | "sample";

export type PredictionDto = {
  id: number;
  source: PredictionSource;
  predictedDigit: number;
  confidence: number;
  probabilities: number[];
  pixels: string;
  actualDigit: number | null;
  isCorrect: boolean | null;
  inferenceMs: number;
  preprocessMs: number;
  createdAt: string;
};

export type PredictionListItem = Omit<PredictionDto, "probabilities"> & {
  probabilities?: number[];
};

export type StatsDto = {
  total: number;
  avgConfidence: number;
  avgInferenceMs: number;
  feedbackTotal: number;
  feedbackCorrect: number;
  feedbackAccuracy: number | null;
  sampleTotal: number;
  sampleCorrect: number;
  sampleAccuracy: number | null;
  perDigit: Array<{ digit: number; count: number }>;
  sources: Array<{ source: string; count: number }>;
};

export type SampleDto = {
  id: number;
  label: number;
  pixels: string;
};

export type ModelDto = {
  id: number;
  version: string;
  dataset: string;
  architecture: string;
  parameters: number;
  filters: number;
  epochs: number;
  learningRate: number;
  batchSize: number;
  trainSamples: number;
  testSamples: number;
  testAccuracy: number;
  testLoss: number;
  trainingSeconds: number;
  history: Array<{
    epoch: number;
    trainLoss: number;
    trainAccuracy: number;
    testLoss: number;
    testAccuracy: number;
    seconds: number;
  }>;
  perClass: Array<{ digit: number; recall: number; support: number }>;
  confusion: number[][];
};

export type PredictResponse = {
  prediction: PredictionDto;
  preprocess: {
    inverted: boolean;
    threshold: number;
    inkPixels: number;
    bbox: { x: number; y: number; width: number; height: number };
    centerOfMass: { x: number; y: number };
    scaledSize: { width: number; height: number };
    ms: number;
  } | null;
  inferenceMs: number;
  featureMaps: number[][];
  modelVersion: string;
};
