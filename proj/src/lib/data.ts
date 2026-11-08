import { and, desc, eq, sql } from "drizzle-orm";
import { db } from "@/db";
import { datasetSamples, models, predictions } from "@/db/schema";
import { ensureSeeded } from "./seed";
import type { ModelDto, PredictionDto, SampleDto, StatsDto } from "./types";

const predictionColumns = {
  id: predictions.id,
  source: predictions.source,
  predictedDigit: predictions.predictedDigit,
  confidence: predictions.confidence,
  probabilities: predictions.probabilities,
  pixels: predictions.pixels,
  actualDigit: predictions.actualDigit,
  isCorrect: predictions.isCorrect,
  inferenceMs: predictions.inferenceMs,
  preprocessMs: predictions.preprocessMs,
  createdAt: predictions.createdAt,
};

export type PredictionRecord = {
  id: number;
  source: string;
  predictedDigit: number;
  confidence: number;
  probabilities: number[];
  pixels: string;
  actualDigit: number | null;
  isCorrect: boolean | null;
  inferenceMs: number;
  preprocessMs: number;
  createdAt: Date;
};

export function toPredictionDto(row: PredictionRecord): PredictionDto {
  return {
    id: row.id,
    source: (row.source as PredictionDto["source"]) ?? "draw",
    predictedDigit: row.predictedDigit,
    confidence: row.confidence,
    probabilities: row.probabilities,
    pixels: row.pixels,
    actualDigit: row.actualDigit,
    isCorrect: row.isCorrect,
    inferenceMs: row.inferenceMs,
    preprocessMs: row.preprocessMs,
    createdAt: row.createdAt.toISOString(),
  };
}

export type PredictionFilters = {
  limit?: number;
  source?: string;
  digit?: number;
  onlyMismatch?: boolean;
};

export async function getPredictions(filters: PredictionFilters = {}): Promise<PredictionDto[]> {
  await ensureSeeded();
  const limit = Math.min(Math.max(filters.limit ?? 20, 1), 300);
  const where = [];
  if (filters.source && filters.source !== "all") {
    where.push(eq(predictions.source, filters.source));
  }
  if (typeof filters.digit === "number" && !Number.isNaN(filters.digit)) {
    where.push(eq(predictions.predictedDigit, filters.digit));
  }
  if (filters.onlyMismatch) {
    where.push(eq(predictions.isCorrect, false));
  }

  const rows = await db
    .select(predictionColumns)
    .from(predictions)
    .where(where.length ? and(...where) : undefined)
    .orderBy(desc(predictions.id))
    .limit(limit);

  return rows.map((row) => toPredictionDto(row as PredictionRecord));
}

export async function getPredictionById(id: number): Promise<PredictionDto | null> {
  await ensureSeeded();
  const rows = await db
    .select(predictionColumns)
    .from(predictions)
    .where(eq(predictions.id, id))
    .limit(1);
  if (!rows.length) return null;
  return toPredictionDto(rows[0] as PredictionRecord);
}

export async function getStats(): Promise<StatsDto> {
  await ensureSeeded();

  const [totals] = await db
    .select({
      total: sql<number>`count(*)::int`,
      avgConfidence: sql<number>`coalesce(avg(${predictions.confidence}), 0)::float`,
      avgInferenceMs: sql<number>`coalesce(avg(${predictions.inferenceMs}), 0)::float`,
      feedbackTotal: sql<number>`count(*) filter (where ${predictions.source} <> 'sample' and ${predictions.isCorrect} is not null)::int`,
      feedbackCorrect: sql<number>`count(*) filter (where ${predictions.source} <> 'sample' and ${predictions.isCorrect} = true)::int`,
      sampleTotal: sql<number>`count(*) filter (where ${predictions.source} = 'sample' and ${predictions.isCorrect} is not null)::int`,
      sampleCorrect: sql<number>`count(*) filter (where ${predictions.source} = 'sample' and ${predictions.isCorrect} = true)::int`,
    })
    .from(predictions);

  const perDigitRows = await db
    .select({
      digit: predictions.predictedDigit,
      count: sql<number>`count(*)::int`,
    })
    .from(predictions)
    .groupBy(predictions.predictedDigit);

  const sourceRows = await db
    .select({
      source: predictions.source,
      count: sql<number>`count(*)::int`,
    })
    .from(predictions)
    .groupBy(predictions.source);

  const perDigitMap = new Map(perDigitRows.map((row) => [row.digit, row.count]));

  return {
    total: totals?.total ?? 0,
    avgConfidence: totals?.avgConfidence ?? 0,
    avgInferenceMs: totals?.avgInferenceMs ?? 0,
    feedbackTotal: totals?.feedbackTotal ?? 0,
    feedbackCorrect: totals?.feedbackCorrect ?? 0,
    feedbackAccuracy:
      totals && totals.feedbackTotal > 0 ? totals.feedbackCorrect / totals.feedbackTotal : null,
    sampleTotal: totals?.sampleTotal ?? 0,
    sampleCorrect: totals?.sampleCorrect ?? 0,
    sampleAccuracy:
      totals && totals.sampleTotal > 0 ? totals.sampleCorrect / totals.sampleTotal : null,
    perDigit: Array.from({ length: 10 }, (_, digit) => ({
      digit,
      count: perDigitMap.get(digit) ?? 0,
    })),
    sources: sourceRows.map((row) => ({ source: row.source, count: row.count })),
  };
}

export async function getModel(): Promise<ModelDto | null> {
  await ensureSeeded();
  const rows = await db.select().from(models).orderBy(desc(models.id)).limit(1);
  if (!rows.length) return null;
  const row = rows[0];
  return {
    id: row.id,
    version: row.version,
    dataset: row.dataset,
    architecture: row.architecture,
    parameters: row.parameters,
    filters: row.filters,
    epochs: row.epochs,
    learningRate: row.learningRate,
    batchSize: row.batchSize,
    trainSamples: row.trainSamples,
    testSamples: row.testSamples,
    testAccuracy: row.testAccuracy,
    testLoss: row.testLoss,
    trainingSeconds: row.trainingSeconds,
    history: row.history,
    perClass: row.perClass,
    confusion: row.confusion,
  };
}

export async function getSamples(limit = 60): Promise<SampleDto[]> {
  await ensureSeeded();
  const rows = await db
    .select({ id: datasetSamples.id, label: datasetSamples.label, pixels: datasetSamples.pixels })
    .from(datasetSamples)
    .orderBy(datasetSamples.id)
    .limit(Math.min(Math.max(limit, 1), 120));
  return rows;
}
