import { db } from "@/db";
import { predictions } from "@/db/schema";
import { getPredictions } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const limitParam = Number(searchParams.get("limit"));
    const digitParam = searchParams.get("digit");
    const items = await getPredictions({
      limit: Number.isFinite(limitParam) && limitParam > 0 ? limitParam : 30,
      source: searchParams.get("source") ?? undefined,
      digit: digitParam !== null && digitParam !== "" ? Number(digitParam) : undefined,
      onlyMismatch: searchParams.get("mismatch") === "1",
    });
    return Response.json({ items });
  } catch (error) {
    console.error("list predictions failed", error);
    return Response.json({ error: "Could not load predictions." }, { status: 500 });
  }
}

export async function DELETE() {
  try {
    await db.delete(predictions);
    return Response.json({ ok: true });
  } catch (error) {
    console.error("clear predictions failed", error);
    return Response.json({ error: "Could not clear history." }, { status: 500 });
  }
}
