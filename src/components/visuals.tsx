import type { ReactNode } from "react";

export function StatTile({
  label,
  value,
  hint,
  tone = "light",
}: {
  label: string;
  value: ReactNode;
  hint?: string;
  tone?: "light" | "dark";
}) {
  const dark = tone === "dark";
  return (
    <div
      className={`rounded-xl border px-3.5 py-3 ${
        dark ? "border-white/10 bg-white/[0.04]" : "border-line bg-white/70"
      }`}
    >
      <p className={`eyebrow ${dark ? "text-white/45" : ""}`}>{label}</p>
      <p
        className={`mt-1 font-mono text-lg font-semibold tabular-nums tracking-tight ${
          dark ? "text-lime" : "text-ink"
        }`}
      >
        {value}
      </p>
      {hint ? (
        <p className={`mt-0.5 text-[11px] ${dark ? "text-white/40" : "text-muted"}`}>{hint}</p>
      ) : null}
    </div>
  );
}

export function ProbabilityBars({
  probabilities,
  predictedDigit,
  compact = false,
}: {
  probabilities: number[];
  predictedDigit: number;
  compact?: boolean;
}) {
  const max = Math.max(...probabilities, 0.0001);
  return (
    <div className={compact ? "space-y-1" : "space-y-1.5"}>
      {probabilities.map((probability, digit) => {
        const winner = digit === predictedDigit;
        const width = Math.max((probability / max) * 100, probability > 0 ? 1.5 : 0);
        return (
          <div key={digit} className="flex items-center gap-2.5">
            <span
              className={`w-4 font-mono text-xs tabular-nums ${
                winner ? "font-bold text-ink" : "text-muted"
              }`}
            >
              {digit}
            </span>
            <div className="relative h-2.5 flex-1 overflow-hidden rounded-full bg-ink/[0.06]">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  winner ? "bg-ink" : probability > 0.05 ? "bg-ink/35" : "bg-ink/15"
                }`}
                style={{ width: `${width}%` }}
              />
            </div>
            <span
              className={`w-12 text-right font-mono text-[11px] tabular-nums ${
                winner ? "font-semibold text-ink" : "text-muted"
              }`}
            >
              {(probability * 100).toFixed(probability >= 0.1 ? 1 : 2)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function SourceBadge({ source }: { source: string }) {
  const map: Record<string, { label: string; className: string }> = {
    draw: { label: "draw", className: "bg-ink text-paper" },
    upload: { label: "upload", className: "bg-sun/20 text-[#8a5a10] border border-sun/40" },
    sample: { label: "mnist", className: "bg-lime/50 text-ink border border-lime" },
  };
  const item = map[source] ?? map.draw;
  return (
    <span
      className={`inline-flex items-center rounded-md px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wider ${item.className}`}
    >
      {item.label}
    </span>
  );
}

export function TrainingChart({
  history,
  metric,
  color = "#12140f",
}: {
  history: Array<{ epoch: number; trainAccuracy: number; testAccuracy: number; trainLoss: number; testLoss: number }>;
  metric: "accuracy" | "loss";
  color?: string;
}) {
  const width = 460;
  const height = 170;
  const padding = { top: 14, right: 12, bottom: 24, left: 34 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const trainKey = metric === "accuracy" ? "trainAccuracy" : "trainLoss";
  const testKey = metric === "accuracy" ? "testAccuracy" : "testLoss";
  const all = history.flatMap((row) => [row[trainKey], row[testKey]]);
  const maxValue = metric === "accuracy" ? 1 : Math.max(...all) * 1.1;
  const minValue = metric === "accuracy" ? Math.min(...all) - 0.02 : 0;
  const span = Math.max(maxValue - minValue, 1e-6);

  const x = (index: number) =>
    padding.left + (history.length === 1 ? innerW / 2 : (index / (history.length - 1)) * innerW);
  const y = (value: number) => padding.top + innerH - ((value - minValue) / span) * innerH;

  const path = (key: "trainAccuracy" | "testAccuracy" | "trainLoss" | "testLoss") =>
    history.map((row, index) => `${index === 0 ? "M" : "L"}${x(index)},${y(row[key])}`).join(" ");

  const ticks = metric === "accuracy" ? [minValue, (minValue + maxValue) / 2, maxValue] : [0, maxValue / 2, maxValue];
  const format = (value: number) =>
    metric === "accuracy" ? `${(value * 100).toFixed(0)}%` : value.toFixed(2);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={`${metric} per epoch`}>
      {ticks.map((tick) => (
        <g key={tick}>
          <line
            x1={padding.left}
            x2={width - padding.right}
            y1={y(tick)}
            y2={y(tick)}
            stroke="#12140f"
            strokeOpacity={0.1}
            strokeDasharray="2 4"
          />
          <text x={padding.left - 6} y={y(tick) + 3} textAnchor="end" fontSize={8} fill="#6d7265" fontFamily="monospace">
            {format(tick)}
          </text>
        </g>
      ))}
      <path d={path(trainKey)} fill="none" stroke="#6d7265" strokeWidth={1.5} strokeDasharray="4 3" />
      <path d={path(testKey)} fill="none" stroke={color} strokeWidth={2.2} strokeLinecap="round" />
      {history.map((row, index) => (
        <circle key={row.epoch} cx={x(index)} cy={y(row[testKey])} r={2.5} fill={color} />
      ))}
      {history.map((row, index) => (
        <text
          key={`x-${row.epoch}`}
          x={x(index)}
          y={height - 7}
          textAnchor="middle"
          fontSize={8}
          fill="#6d7265"
          fontFamily="monospace"
        >
          {row.epoch}
        </text>
      ))}
    </svg>
  );
}

export function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(...matrix.flat(), 1);
  return (
    <div className="inline-block">
      <div className="flex">
        <div className="w-6" />
        {Array.from({ length: 10 }, (_, digit) => (
          <div key={digit} className="w-7 text-center font-mono text-[9px] text-muted">
            {digit}
          </div>
        ))}
      </div>
      {matrix.map((row, actual) => (
        <div key={actual} className="flex items-center">
          <div className="w-6 font-mono text-[9px] text-muted">{actual}</div>
          {row.map((value, predicted) => {
            const intensity = value === 0 ? 0 : 0.12 + (value / max) * 0.88;
            const correct = actual === predicted;
            return (
              <div
                key={predicted}
                title={`actual ${actual} → predicted ${predicted}: ${value}`}
                className={`m-0.5 flex h-7 w-7 items-center justify-center rounded font-mono text-[8px] ${
                  correct ? "text-ink" : intensity > 0.4 ? "text-paper" : "text-muted"
                }`}
                style={{
                  backgroundColor: correct
                    ? `rgba(203, 235, 78, ${0.35 + intensity * 0.65})`
                    : `rgba(18, 20, 15, ${intensity})`,
                }}
              >
                {value === 0 ? "" : value}
              </div>
            );
          })}
        </div>
      ))}
      <div className="mt-1 flex pl-6 text-[9px] text-muted">
        <span className="w-[280px] text-center font-mono">predicted →</span>
      </div>
    </div>
  );
}
