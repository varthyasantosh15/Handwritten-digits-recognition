import { getModel } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const model = await getModel();
    if (!model) {
      return Response.json({ error: "No model registered." }, { status: 404 });
    }
    return Response.json({ model });
  } catch (error) {
    console.error("model failed", error);
    return Response.json({ error: "Could not load model metadata." }, { status: 500 });
  }
}
