import { Studio } from "@/components/Studio";
import { getModel, getPredictions, getSamples, getStats } from "@/lib/data";
import type { ModelDto, PredictionDto, SampleDto, StatsDto } from "@/lib/types";

export const dynamic = "force-dynamic";

const emptyStats: StatsDto = {
  total: 0,
  avgConfidence: 0,
  avgInferenceMs: 0,
  feedbackTotal: 0,
  feedbackCorrect: 0,
  feedbackAccuracy: null,
  sampleTotal: 0,
  sampleCorrect: 0,
  sampleAccuracy: null,
  perDigit: Array.from({ length: 10 }, (_, digit) => ({ digit, count: 0 })),
  sources: [],
};

export default async function HomePage() {
  const [model, stats, recent, samples] = await Promise.all([
    getModel().catch((): ModelDto | null => null),
    getStats().catch(() => emptyStats),
    getPredictions({ limit: 8 }).catch((): PredictionDto[] => []),
    getSamples(60).catch((): SampleDto[] => []),
  ]);

  return (
    <Studio
      initialRecent={recent}
      initialStats={stats}
      initialSamples={samples}
      model={model}
    />
  );
}
