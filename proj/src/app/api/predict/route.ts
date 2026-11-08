import { eq } from "drizzle-orm";
import { db } from "@/db";
import { datasetSamples, predictions } from "@/db/schema";
import { predict } from "@/lib/cnn";
import { toPredictionDto, type PredictionRecord } from "@/lib/data";
import {
  FRAME_SIZE,
  PreprocessError,
  base64ToBytes,
  bytesToBase64,
  preprocessGray,
} from "@/lib/preprocess";
import { ensureSeeded } from "@/lib/seed";
import type { PredictionSource, PredictResponse } from "@/lib/types";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type PredictBody = {
  sampleId?: number;
  pixels28?: number[];
  image?: string;
  width?: number;
  height?: number;
  source?: string;
  actualDigit?: number | null;
};

const VALID_SOURCES: PredictionSource[] = ["draw", "upload", "sample"];

function asDigit(value: unknown): number | null {
  return typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 9
    ? value
    : null;
}

export async function POST(request: Request) {
  try {
    await ensureSeeded();
    const body = (await request.json()) as PredictBody;

    let pixels: Uint8Array;
    let preprocessPayload: PredictResponse["preprocess"] = null;
    let source: PredictionSource = VALID_SOURCES.includes(body.source as PredictionSource)
      ? (body.source as PredictionSource)
      : "draw";
    let actualDigit = asDigit(body.actualDigit);

    if (typeof body.sampleId === "number") {
      const rows = await db
        .select({ label: datasetSamples.label, pixels: datasetSamples.pixels })
        .from(datasetSamples)
        .where(eq(datasetSamples.id, body.sampleId))
        .limit(1);
      if (!rows.length) {
        return Response.json({ error: "Sample not found." }, { status: 404 });
      }
      pixels = base64ToBytes(rows[0].pixels);
      if (pixels.length !== FRAME_SIZE * FRAME_SIZE) {
        return Response.json({ error: "Sample is corrupted." }, { status: 500 });
      }
      source = "sample";
      actualDigit = rows[0].label;
    } else if (Array.isArray(body.pixels28)) {
      if (body.pixels28.length !== FRAME_SIZE * FRAME_SIZE) {
        return Response.json({ error: "Expected 784 pixel values." }, { status: 400 });
      }
      pixels = Uint8Array.from(
        body.pixels28.map((value) => Math.max(0, Math.min(255, Math.round(Number(value) || 0)))),
      );
    } else if (typeof body.image === "string" && body.image.length > 0) {
      const gray = base64ToBytes(body.image);
      const width = Math.round(Number(body.width));
      const height = Math.round(Number(body.height));
      if (!Number.isFinite(width) || !Number.isFinite(height) || width < 8 || height < 8) {
        return Response.json({ error: "Image dimensions are invalid." }, { status: 400 });
      }
      if (width * height !== gray.length) {
        return Response.json(
          { error: "Image buffer does not match the supplied dimensions." },
          { status: 400 },
        );
      }

      const preprocessStart = performance.now();
      let result;
      try {
        result = preprocessGray(gray, width, height);
      } catch (error) {
        if (error instanceof PreprocessError) {
          return Response.json({ error: error.message }, { status: 422 });
        }
        throw error;
      }
      const preprocessMs = Number((performance.now() - preprocessStart).toFixed(2));
      pixels = result.pixels;
      preprocessPayload = {
        inverted: result.inverted,
        threshold: result.threshold,
        inkPixels: result.inkPixels,
        bbox: result.bbox,
        centerOfMass: result.centerOfMass,
        scaledSize: result.scaledSize,
        ms: preprocessMs,
      };
    } else {
      return Response.json(
        { error: "Provide an image, 784 pixel values, or a sampleId." },
        { status: 400 },
      );
    }

    const result = predict(pixels, { tta: source !== "sample" });
    const isCorrect = actualDigit === null ? null : actualDigit === result.digit;

    const [inserted] = await db
      .insert(predictions)
      .values({
        source,
        predictedDigit: result.digit,
        confidence: result.confidence,
        probabilities: result.probabilities,
        pixels: bytesToBase64(pixels),
        actualDigit,
        isCorrect,
        inferenceMs: result.inferenceMs,
        preprocessMs: preprocessPayload?.ms ?? 0,
      })
      .returning();

    const response: PredictResponse = {
      prediction: toPredictionDto(inserted as PredictionRecord),
      preprocess: preprocessPayload,
      inferenceMs: result.inferenceMs,
      featureMaps: result.featureMaps,
      modelVersion: result.modelVersion,
    };

    return Response.json(response, { status: 201 });
  } catch (error) {
    console.error("predict failed", error);
    return Response.json({ error: "Could not run inference. Please try again." }, { status: 500 });
  }
}
