import { getSamples } from "@/lib/data";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  try {
    return Response.json({ samples: await getSamples(60) });
  } catch (error) {
    console.error("samples failed", error);
    return Response.json({ error: "Could not load samples." }, { status: 500 });
  }
}
