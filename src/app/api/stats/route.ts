import { getStats } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return Response.json(await getStats());
  } catch (error) {
    console.error("stats failed", error);
    return Response.json({ error: "Could not load stats." }, { status: 500 });
  }
}
