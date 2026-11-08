import { eq } from "drizzle-orm";
import { db } from "@/db";
import { predictions } from "@/db/schema";
import { getPredictionById, toPredictionDto, type PredictionRecord } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type RouteContext = { params: Promise<{ id: string }> };

export async function PATCH(request: Request, context: RouteContext) {
  try {
    const { id } = await context.params;
    const predictionId = Number(id);
    if (!Number.isInteger(predictionId) || predictionId <= 0) {
      return Response.json({ error: "Invalid prediction id." }, { status: 400 });
    }

    const existing = await getPredictionById(predictionId);
    if (!existing) {
      return Response.json({ error: "Prediction not found." }, { status: 404 });
    }

    const body = (await request.json()) as { actualDigit?: number | null };
    const actualDigit =
      typeof body.actualDigit === "number" &&
      Number.isInteger(body.actualDigit) &&
      body.actualDigit >= 0 &&
      body.actualDigit <= 9
        ? body.actualDigit
        : null;

    const [updated] = await db
      .update(predictions)
      .set({
        actualDigit,
        isCorrect: actualDigit === null ? null : actualDigit === existing.predictedDigit,
      })
      .where(eq(predictions.id, predictionId))
      .returning();

    return Response.json({ prediction: toPredictionDto(updated as PredictionRecord) });
  } catch (error) {
    console.error("feedback failed", error);
    return Response.json({ error: "Could not save feedback." }, { status: 500 });
  }
}

export async function DELETE(_request: Request, context: RouteContext) {
  try {
    const { id } = await context.params;
    const predictionId = Number(id);
    if (!Number.isInteger(predictionId) || predictionId <= 0) {
      return Response.json({ error: "Invalid prediction id." }, { status: 400 });
    }
    await db.delete(predictions).where(eq(predictions.id, predictionId));
    return Response.json({ ok: true });
  } catch (error) {
    console.error("delete prediction failed", error);
    return Response.json({ error: "Could not delete prediction." }, { status: 500 });
  }
}
