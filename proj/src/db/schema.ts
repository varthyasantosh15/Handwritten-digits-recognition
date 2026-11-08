import {
  boolean,
  index,
  integer,
  jsonb,
  pgTable,
  real,
  serial,
  text,
  timestamp,
  varchar,
} from "drizzle-orm/pg-core";

/**
 * Every inference produced by the app. The 28x28 MNIST-format tensor is stored
 * as a base64 string so history thumbnails can be re-rendered without the
 * original upload.
 */
export const predictions = pgTable(
  "predictions",
  {
    id: serial("id").primaryKey(),
    source: varchar("source", { length: 16 }).notNull().default("draw"),
    predictedDigit: integer("predicted_digit").notNull(),
    confidence: real("confidence").notNull(),
    probabilities: jsonb("probabilities").$type<number[]>().notNull(),
    pixels: text("pixels").notNull(),
    actualDigit: integer("actual_digit"),
    isCorrect: boolean("is_correct"),
    inferenceMs: real("inference_ms").notNull().default(0),
    preprocessMs: real("preprocess_ms").notNull().default(0),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("predictions_created_at_idx").on(table.createdAt)],
);

/** Metadata for the trained CNN artifact that this deployment serves. */
export const models = pgTable("models", {
  id: serial("id").primaryKey(),
  version: varchar("version", { length: 64 }).notNull().unique(),
  dataset: text("dataset").notNull(),
  architecture: text("architecture").notNull(),
  parameters: integer("parameters").notNull(),
  filters: integer("filters").notNull(),
  epochs: integer("epochs").notNull(),
  learningRate: real("learning_rate").notNull(),
  batchSize: integer("batch_size").notNull(),
  trainSamples: integer("train_samples").notNull(),
  testSamples: integer("test_samples").notNull(),
  testAccuracy: real("test_accuracy").notNull(),
  testLoss: real("test_loss").notNull(),
  trainingSeconds: integer("training_seconds").notNull(),
  history: jsonb("history").$type<TrainingEpoch[]>().notNull(),
  perClass: jsonb("per_class").$type<PerClassMetric[]>().notNull(),
  confusion: jsonb("confusion").$type<number[][]>().notNull(),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

/**
 * A fixed gallery of unseen MNIST test digits, surfaced in the UI so anyone can
 * try the model without drawing anything.
 */
export const datasetSamples = pgTable("dataset_samples", {
  id: serial("id").primaryKey(),
  sampleKey: varchar("sample_key", { length: 64 }).notNull().unique(),
  label: integer("label").notNull(),
  split: varchar("split", { length: 16 }).notNull().default("test"),
  pixels: text("pixels").notNull(),
});

export type TrainingEpoch = {
  epoch: number;
  trainLoss: number;
  trainAccuracy: number;
  testLoss: number;
  testAccuracy: number;
  seconds: number;
};

export type PerClassMetric = {
  digit: number;
  recall: number;
  support: number;
};

export type PredictionRow = typeof predictions.$inferSelect;
export type ModelRow = typeof models.$inferSelect;
export type DatasetSampleRow = typeof datasetSamples.$inferSelect;
