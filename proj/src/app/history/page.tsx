import { HistoryClient } from "@/components/HistoryClient";
import { getPredictions, getStats } from "@/lib/data";
import type { PredictionDto, StatsDto } from "@/lib/types";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Prediction history — DigitLab",
  description: "Every handwritten digit classified by the CNN, stored in PostgreSQL.",
};

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

export default async function HistoryPage() {
  const [items, stats] = await Promise.all([
    getPredictions({ limit: 150 }).catch((): PredictionDto[] => []),
    getStats().catch(() => emptyStats),
  ]);

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-3 pb-1 md:flex-row md:items-end md:justify-between">
        <div>
          <p className="eyebrow">/ prediction log</p>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight sm:text-4xl">
            Every read, stored.
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
            Each inference is a row in the <code className="font-mono text-xs">predictions</code>{" "}
            table: the 28×28 tensor, the softmax vector, latency and your ground-truth labels. Use it
            to audit the model or export a dataset.
          </p>
        </div>
        <div className="flex gap-2">
          <div className="rounded-xl border border-line bg-white/70 px-3 py-2">
            <p className="eyebrow">rows</p>
            <p className="font-mono text-lg font-semibold">{stats.total.toLocaleString()}</p>
          </div>
          <div className="rounded-xl border border-line bg-white/70 px-3 py-2">
            <p className="eyebrow">labelled acc.</p>
            <p className="font-mono text-lg font-semibold">
              {stats.feedbackAccuracy === null ? "—" : `${(stats.feedbackAccuracy * 100).toFixed(1)}%`}
            </p>
          </div>
        </div>
      </header>

      <HistoryClient initialItems={items} initialStats={stats} />
    </div>
  );
}
