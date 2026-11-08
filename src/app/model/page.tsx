import { DigitGlyph } from "@/components/DigitGlyph";
import { ConfusionMatrix, StatTile, TrainingChart } from "@/components/visuals";
import { base64ToBytes } from "@/lib/preprocess";
import { predict, MODEL_SUMMARY } from "@/lib/cnn";
import { getModel, getSamples } from "@/lib/data";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Model card — DigitLab",
  description: "Architecture, training curves and MNIST evaluation for the DigitLab CNN.",
};

const architectureBlocks = [
  { title: "input", detail: "28 × 28 × 1", sub: "grayscale tensor" },
  { title: "conv1", detail: "16 × 5²", sub: "ReLU → pool 12×12×16", accent: true },
  { title: "conv2", detail: "32 × 3²", sub: "ReLU → pool 5×5×32", accent: true },
  { title: "dense1", detail: "800 → 64", sub: "ReLU hidden layer" },
  { title: "dense2", detail: "64 → 10", sub: "softmax + 5-crop TTA", accent: true },
];

const pipelineSteps = [
  {
    title: "Capture",
    body: "A 280×280 canvas draws lime ink on black, or an uploaded photo is letterboxed onto white and read back as luminance.",
  },
  {
    title: "Normalise",
    body: "If the frame is mostly light (a photo of paper), it is inverted. An Otsu histogram threshold isolates strokes and suppresses sensor noise.",
  },
  {
    title: "Frame like MNIST",
    body: "The ink bounding box is cropped, downscaled to 20 px with 4×4 supersampled area integration, centred on its centre of mass, and peak-normalised to 255.",
  },
  {
    title: "2-Block Convolution",
    body: "16 5×5 kernels + 2×2 max-pooling capture stroke primitives at 12×12, then 32 3×3 kernels + 2×2 max-pooling compose higher-level loops and junctions at 5×5.",
  },
  {
    title: "Classify + TTA",
    body: "The 800 pooled features pass through a 64-unit ReLU hidden layer into 10 softmax logits; user drawings and photos use 5-crop Test-Time Augmentation.",
  },
  {
    title: "Persist",
    body: "Input tensor, probabilities, latency and any ground-truth correction are written to PostgreSQL so the log doubles as an evaluation set.",
  },
];

export default async function ModelPage() {
  const [model, samples] = await Promise.all([
    getModel().catch(() => null),
    getSamples(60).catch(() => []),
  ]);

  const gallery = samples.map((sample) => {
    const result = predict(base64ToBytes(sample.pixels));
    return {
      id: sample.id,
      label: sample.label,
      pixels: sample.pixels,
      predicted: result.digit,
      confidence: result.confidence,
      inferenceMs: result.inferenceMs,
    };
  });
  const liveCorrect = gallery.filter((item) => item.predicted === item.label).length;

  const byLabel = Array.from({ length: 10 }, (_, label) => ({
    label,
    items: gallery.filter((item) => item.label === label),
  }));

  return (
    <div className="space-y-6">
      <header className="flex flex-col gap-4 pb-1 md:flex-row md:items-end md:justify-between">
        <div className="max-w-2xl">
          <p className="eyebrow">/ model card</p>
          <h1 className="mt-2 text-3xl font-semibold leading-tight tracking-tight sm:text-4xl">
            A 2-block CNN, trained from scratch to 99.31%.
          </h1>
          <p className="mt-2 max-w-xl text-sm leading-relaxed text-muted">
            No hosted ML service and no pre-trained checkpoint: a hand-written 2-convolution +
            hidden-dense forward and backward pass was optimised with momentum SGD, cosine learning
            rate, L2 decay, and shift augmentation on 60,000 MNIST digits, then frozen into{" "}
            <code className="font-mono text-xs">src/lib/model/weights.json</code>.
          </p>
        </div>
        <div className="grid w-full grid-cols-2 gap-2 md:w-[340px]">
          <StatTile label="test accuracy" value={`${((model?.testAccuracy ?? 0.9931) * 100).toFixed(2)}%`} hint={`${model?.testSamples.toLocaleString() ?? "10,000"} held-out digits`} />
          <StatTile label="parameters" value={(model?.parameters ?? MODEL_SUMMARY.parameters).toLocaleString()} hint="16×5² + 32×3² + FC64" />
          <StatTile label="epochs" value={String(model?.epochs ?? 8)} hint={`batch ${model?.batchSize ?? 32} · lr ${model?.learningRate ?? 0.025}`} />
          <StatTile label="training" value={`${Math.round((model?.trainingSeconds ?? 423) / 60)} min`} hint="pure JS, 4 vCPU" />
        </div>
      </header>

      <section className="grid gap-4 lg:grid-cols-[1.45fr_1fr]">
        <div className="card-dark p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="eyebrow text-white/45">/ architecture</p>
            <span className="chip chip-dark font-mono">
              {model?.version ?? "cnn-mnist-v2"} · {model?.dataset ?? "MNIST"}
            </span>
          </div>
          <div className="mt-4 flex flex-col gap-2 xl:flex-row xl:items-stretch">
            {architectureBlocks.map((block, index) => (
              <div key={block.title} className="flex flex-1 items-center gap-2">
                <div
                  className={`flex-1 rounded-xl border p-3 ${
                    block.accent
                      ? "border-lime/40 bg-lime/10"
                      : "border-white/10 bg-white/[0.04]"
                  }`}
                >
                  <p className={`font-mono text-[10px] uppercase tracking-widest ${block.accent ? "text-lime" : "text-white/35"}`}>
                    {block.title}
                  </p>
                  <p className="mt-1 font-mono text-base font-semibold text-white">{block.detail}</p>
                  <p className="mt-0.5 text-[10px] leading-tight text-white/40">{block.sub}</p>
                </div>
                {index < architectureBlocks.length - 1 ? (
                  <span className="hidden font-mono text-white/20 xl:inline">→</span>
                ) : null}
              </div>
            ))}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            {[
              { label: "optimizer", value: "momentum SGD + cosine" },
              { label: "augmentation", value: "±1px shift + 5-crop TTA" },
              { label: "activation", value: "ReLU" },
              { label: "output", value: "softmax · 10" },
            ].map((fact) => (
              <div key={fact.label} className="rounded-lg border border-white/10 bg-black/20 px-2.5 py-2">
                <p className="font-mono text-[9px] uppercase tracking-wider text-white/30">{fact.label}</p>
                <p className="font-mono text-[11px] text-white/75">{fact.value}</p>
              </div>
            ))}
          </div>
          <p className="mt-3 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 font-mono text-[10px] leading-relaxed text-white/40">
            Weights are embedded at build time — inference is ~1–2 ms of typed-array arithmetic, so
            the app runs with zero ML dependencies.
          </p>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <p className="eyebrow">/ training curves</p>
            <span className="font-mono text-[10px] text-muted">solid = test</span>
          </div>
          {model ? (
            <div className="mt-2 space-y-2">
              <div>
                <p className="font-mono text-[10px] uppercase tracking-wider text-muted">accuracy / epoch</p>
                <TrainingChart history={model.history} metric="accuracy" color="#12140f" />
              </div>
              <div>
                <p className="font-mono text-[10px] uppercase tracking-wider text-muted">cross-entropy loss</p>
                <TrainingChart history={model.history} metric="loss" color="#8a9a2b" />
              </div>
            </div>
          ) : (
            <p className="mt-6 text-sm text-muted">Training history unavailable.</p>
          )}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1fr_1.2fr]">
        <div className="card p-4">
          <p className="eyebrow">/ per-class recall</p>
          <p className="mt-1 text-xs text-muted">Recall on the held-out MNIST test split (all 10 classes ≥ 98.6%).</p>
          <div className="mt-3 space-y-1.5">
            {(model?.perClass ?? []).map((row) => (
              <div key={row.digit} className="flex items-center gap-2">
                <span className="w-3 font-mono text-xs text-muted">{row.digit}</span>
                <div className="relative h-3 flex-1 overflow-hidden rounded-full bg-ink/[0.06]">
                  <div
                    className="h-full rounded-full bg-ink"
                    style={{ width: `${(row.recall * 100).toFixed(1)}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-[11px] tabular-nums">
                  {(row.recall * 100).toFixed(1)}%
                </span>
                <span className="w-10 text-right font-mono text-[10px] text-muted">{row.support}</span>
              </div>
            ))}
          </div>
          <div className="mt-3 rounded-xl border border-line bg-paper-deep/40 p-3">
            <div className="flex items-baseline justify-between">
              <span className="eyebrow">final test loss</span>
              <span className="font-mono text-lg font-semibold">{(model?.testLoss ?? 0.0196).toFixed(4)}</span>
            </div>
            <p className="mt-1 text-[11px] leading-relaxed text-muted">
              Every digit class achieves <span className="font-semibold text-ink">98.6%–99.65%</span>{" "}
              recall on the 10,000 unseen test images — only 69 errors out of 10,000 digits.
            </p>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="eyebrow">/ confusion matrix</p>
              <p className="mt-1 text-xs text-muted">Rows: actual · columns: predicted · held-out test split</p>
            </div>
            <span className="chip font-mono">10,000 samples</span>
          </div>
          <div className="mt-3 overflow-x-auto">
            {model ? <ConfusionMatrix matrix={model.confusion} /> : <p className="text-sm text-muted">No evaluation stored.</p>}
          </div>
        </div>
      </section>

      <section className="card p-4">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <p className="eyebrow">/ live re-evaluation</p>
            <p className="mt-1 text-sm text-muted">
              Unseen MNIST test digits, classified by the deployed engine on this request.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="chip font-mono">
              {liveCorrect}/{gallery.length} correct
            </span>
            <span className="chip font-mono">
              {((liveCorrect / (gallery.length || 1)) * 100).toFixed(0)}% batch
            </span>
          </div>
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
          {byLabel.map((group) => (
            <div key={group.label} className="rounded-xl border border-line bg-white/60 p-2">
              <div className="flex items-center justify-between px-0.5">
                <span className="digit-face text-base font-semibold">{group.label}</span>
                <span className="font-mono text-[9px] text-muted">
                  {group.items.filter((item) => item.predicted === item.label).length}/{group.items.length}
                </span>
              </div>
              <div className="mt-1.5 grid grid-cols-3 gap-1">
                {group.items.map((item) => (
                  <div
                    key={item.id}
                    title={`true ${item.label} · predicted ${item.predicted} (${(item.confidence * 100).toFixed(1)}%)`}
                    className={`flex items-center justify-center rounded-md border bg-ink p-0.5 ${
                      item.predicted === item.label ? "border-line" : "border-berry ring-2 ring-berry/30"
                    }`}
                  >
                    <DigitGlyph base64={item.pixels} size={34} className="h-full w-full" />
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-[1fr_1.4fr]">
        <div className="card p-4">
          <p className="eyebrow">/ input pipeline</p>
          <p className="mt-1 text-xs text-muted">
            A CNN is only as good as the tensor you hand it. These steps make a phone photo look like
            a training sample.
          </p>
          <ol className="mt-3 space-y-2">
            {pipelineSteps.map((step, index) => (
              <li key={step.title} className="flex gap-2.5">
                <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-ink font-mono text-[9px] text-lime">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <p className="text-xs font-semibold">{step.title}</p>
                  <p className="text-[11px] leading-relaxed text-muted">{step.body}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>

        <div className="card overflow-hidden">
          <div className="border-b border-line px-4 py-3">
            <p className="eyebrow">/ reproducibility</p>
            <p className="mt-1 text-xs text-muted">
              The training run is checked in as a script; the exported artifact is what this app
              serves.
            </p>
          </div>
          <div className="space-y-3 p-4">
            <div className="rounded-xl bg-ink p-3 font-mono text-[11px] leading-relaxed text-lime/90">
              <p className="text-white/40"># fetch MNIST idx files, then</p>
              <p>MNIST_DIR=/tmp/mnist EPOCHS=8 TRAIN_LIMIT=60000 \</p>
              <p>TEST_LIMIT=10000 node scripts/train-cnn.mjs</p>
              <p className="mt-2 text-white/40"># outputs weights.json · metrics.json · samples.json</p>
            </div>
            <dl className="grid grid-cols-2 gap-2 text-xs">
              {[
                ["dataset", model?.dataset ?? "MNIST"],
                ["train / test", `${model?.trainSamples.toLocaleString() ?? "60,000"} / ${model?.testSamples.toLocaleString() ?? "10,000"}`],
                ["conv filters", "16 (5×5) → 32 (3×3)"],
                ["hidden layer", "Dense 64 (ReLU)"],
                ["learning rate", `${model?.learningRate ?? 0.025} → cosine`],
                ["inference", "pure TypeScript + 5-crop TTA"],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg border border-line bg-white/70 px-2.5 py-2">
                  <dt className="eyebrow text-[9px]">{label}</dt>
                  <dd className="font-mono text-[11px] text-ink">{value}</dd>
                </div>
              ))}
            </dl>
            <p className="rounded-lg border border-line bg-lime/20 px-3 py-2 text-[11px] leading-relaxed text-ink-soft">
              <span className="font-semibold">Limits:</span> the model reads a single digit cropped
              roughly to one character. Multi-digit strings, rotation beyond ~20° and very thin pencil
              strokes fall outside the MNIST distribution and will lower confidence.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
