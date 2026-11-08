import { sql } from "drizzle-orm";
import { db } from "@/db";
import { datasetSamples, models } from "@/db/schema";
import metricsJson from "./model/metrics.json";
import samplesJson from "./model/samples.json";
import type { ModelMetrics, DemoRole } from "./model-types";

const metrics = metricsJson as unknown as ModelMetrics;

const ARCHITECTURE_DESC =
  "Conv2D(16 x 5x5, ReLU) -> MaxPool2D(2x2) -> Conv2D(32 x 3x3, ReLU) -> MaxPool2D(2x2) -> Dense(64, ReLU) -> Dense(10, Softmax)";

let seedPromise: Promise<void> | null = null;

export function ensureSeeded(): Promise<void> {
  if (!seedPromise) {
    seedPromise = seed().catch((error) => {
      seedPromise = null;
      throw error;
    });
  }
  return seedPromise;
}

async function seed(): Promise<void> {
  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS models (
      id serial PRIMARY KEY,
      version varchar(64) NOT NULL UNIQUE,
      dataset text NOT NULL,
      architecture text NOT NULL,
      parameters integer NOT NULL,
      filters integer NOT NULL,
      epochs integer NOT NULL,
      learning_rate real NOT NULL,
      batch_size integer NOT NULL,
      train_samples integer NOT NULL,
      test_samples integer NOT NULL,
      test_accuracy real NOT NULL,
      test_loss real NOT NULL,
      training_seconds integer NOT NULL,
      history jsonb NOT NULL,
      per_class jsonb NOT NULL,
      confusion jsonb NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now()
    )
  `);

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS dataset_samples (
      id serial PRIMARY KEY,
      sample_key varchar(64) NOT NULL UNIQUE,
      label integer NOT NULL,
      split varchar(16) NOT NULL DEFAULT 'test',
      pixels text NOT NULL
    )
  `);

  await db.execute(sql`
    CREATE TABLE IF NOT EXISTS predictions (
      id serial PRIMARY KEY,
      source varchar(16) NOT NULL DEFAULT 'draw',
      predicted_digit integer NOT NULL,
      confidence real NOT NULL,
      probabilities jsonb NOT NULL,
      pixels text NOT NULL,
      actual_digit integer,
      is_correct boolean,
      inference_ms real NOT NULL DEFAULT 0,
      preprocess_ms real NOT NULL DEFAULT 0,
      created_at timestamptz NOT NULL DEFAULT now()
    )
  `);
  await db.execute(
    sql`CREATE INDEX IF NOT EXISTS predictions_created_at_idx ON predictions (created_at)`,
  );

  await db
    .insert(models)
    .values({
      version: metrics.version,
      dataset: metrics.dataset,
      architecture: ARCHITECTURE_DESC,
      parameters: metrics.parameters,
      filters: metrics.config.filters,
      epochs: metrics.config.epochs,
      learningRate: metrics.config.learningRate,
      batchSize: metrics.config.batchSize,
      trainSamples: metrics.config.trainSamples,
      testSamples: metrics.config.testSamples,
      testAccuracy: metrics.final.testAccuracy,
      testLoss: metrics.final.testLoss,
      trainingSeconds: metrics.trainingSeconds,
      history: metrics.history,
      perClass: metrics.final.perClass,
      confusion: metrics.final.confusion,
    })
    .onConflictDoUpdate({
      target: models.version,
      set: {
        architecture: ARCHITECTURE_DESC,
        parameters: metrics.parameters,
        filters: metrics.config.filters,
        epochs: metrics.config.epochs,
        learningRate: metrics.config.learningRate,
        batchSize: metrics.config.batchSize,
        trainSamples: metrics.config.trainSamples,
        testSamples: metrics.config.testSamples,
        testAccuracy: metrics.final.testAccuracy,
        testLoss: metrics.final.testLoss,
        trainingSeconds: metrics.trainingSeconds,
        history: metrics.history,
        perClass: metrics.final.perClass,
        confusion: metrics.final.confusion,
      },
    });

  const samples = samplesJson as unknown as DemoRole[];
  if (samples.length > 0) {
    await db
      .insert(datasetSamples)
      .values(
        samples.map((sample) => ({
          sampleKey: `mnist-test-${sample.sourceIndex}`,
          label: sample.label,
          split: "test",
          pixels: sample.pixels,
        })),
      )
      .onConflictDoNothing({ target: datasetSamples.sampleKey });
  }
}
