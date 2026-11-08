export type ModelMetrics = {
  version: string;
  generatedAt: string;
  dataset: string;
  config: {
    filters: number;
    epochs: number;
    trainSamples: number;
    testSamples: number;
    batchSize: number;
    learningRate: number;
    momentum: number;
    optimizer: string;
    activation: string;
  };
  parameters: number;
  history: Array<{
    epoch: number;
    trainLoss: number;
    trainAccuracy: number;
    testLoss: number;
    testAccuracy: number;
    seconds: number;
  }>;
  final: {
    testAccuracy: number;
    testLoss: number;
    perClass: Array<{ digit: number; recall: number; support: number }>;
    confusion: number[][];
  };
  trainingSeconds: number;
};

export type DemoRole = {
  label: number;
  sourceIndex: number;
  pixels: string;
};
