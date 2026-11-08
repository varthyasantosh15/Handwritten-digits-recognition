"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { DigitGlyph } from "./DigitGlyph";
import { SourceBadge } from "./visuals";
import type { PredictionDto, StatsDto } from "@/lib/types";

type HistoryClientProps = {
  initialItems: PredictionDto[];
  initialStats: StatsDto;
};

const sourceOptions = [
  { value: "all", label: "All sources" },
  { value: "draw", label: "Drawn" },
  { value: "upload", label: "Uploaded" },
  { value: "sample", label: "MNIST sample" },
];

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    timeZone: "UTC",
  });
}

function csvEscape(value: string | number | boolean | null): string {
  const text = value === null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function HistoryClient({ initialItems, initialStats }: HistoryClientProps) {
  const [items, setItems] = useState<PredictionDto[]>(initialItems);
  const [stats, setStats] = useState<StatsDto>(initialStats);
  const [source, setSource] = useState("all");
  const [digit, setDigit] = useState("all");
  const [mismatch, setMismatch] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [confirmClear, setConfirmClear] = useState(false);
  const [busyId, setBusyId] = useState<number | null>(null);
  const firstRender = useRef(true);

  const loadStats = useCallback(async () => {
    try {
      const response = await fetch("/api/stats", { cache: "no-store" });
      if (response.ok) setStats((await response.json()) as StatsDto);
    } catch {
      // best effort
    }
  }, []);

  const loadItems = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const params = new URLSearchParams({ limit: "150" });
      if (source !== "all") params.set("source", source);
      if (digit !== "all") params.set("digit", digit);
      if (mismatch) params.set("mismatch", "1");
      const response = await fetch(`/api/predictions?${params.toString()}`, { cache: "no-store" });
      const data = (await response.json()) as { items?: PredictionDto[]; error?: string };
      if (!response.ok || !data.items) {
        setError(data.error ?? "Could not load the prediction log.");
        return;
      }
      setItems(data.items);
    } catch {
      setError("Network error while loading the log.");
    } finally {
      setLoading(false);
    }
  }, [digit, mismatch, source]);

  useEffect(() => {
    if (firstRender.current) {
      firstRender.current = false;
      return;
    }
    void loadItems();
  }, [loadItems]);

  const updateGroundTruth = async (id: number, actualDigit: number | null) => {
    setBusyId(id);
    setError(null);
    try {
      const response = await fetch(`/api/predictions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actualDigit }),
      });
      const data = (await response.json()) as { prediction?: PredictionDto; error?: string };
      if (!response.ok || !data.prediction) {
        setError(data.error ?? "Could not save that label.");
        return;
      }
      const updated = data.prediction;
      setItems((current) => current.map((item) => (item.id === id ? updated : item)));
      void loadStats();
    } catch {
      setError("Network error while saving the label.");
    } finally {
      setBusyId(null);
    }
  };

  const deleteRow = async (id: number) => {
    setBusyId(id);
    setError(null);
    try {
      const response = await fetch(`/api/predictions/${id}`, { method: "DELETE" });
      if (!response.ok) {
        setError("Could not delete that row.");
        return;
      }
      setItems((current) => current.filter((item) => item.id !== id));
      void loadStats();
    } catch {
      setError("Network error while deleting the row.");
    } finally {
      setBusyId(null);
    }
  };

  const clearAll = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch("/api/predictions", { method: "DELETE" });
      if (!response.ok) {
        setError("Could not clear the log.");
        return;
      }
      setItems([]);
      setConfirmClear(false);
      await loadStats();
    } catch {
      setError("Network error while clearing the log.");
    } finally {
      setLoading(false);
    }
  };

  const exportCsv = () => {
    const header = [
      "id",
      "created_at_utc",
      "source",
      "predicted_digit",
      "confidence",
      "actual_digit",
      "is_correct",
      "inference_ms",
      "preprocess_ms",
    ];
    const rows = items.map((item) =>
      [
        item.id,
        item.createdAt,
        item.source,
        item.predictedDigit,
        item.confidence.toFixed(6),
        item.actualDigit,
        item.isCorrect,
        item.inferenceMs,
        item.preprocessMs,
      ]
        .map(csvEscape)
        .join(","),
    );
    const blob = new Blob([[header.join(","), ...rows].join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `digitlab-predictions-${new Date().toISOString().slice(0, 10)}.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const mismatches = items.filter((item) => item.isCorrect === false).length;

  return (
    <div className="space-y-4">
      <div className="card flex flex-wrap items-end gap-3 p-3.5">
        <label className="flex flex-col gap-1">
          <span className="eyebrow">source</span>
          <select className="field min-w-[130px]" value={source} onChange={(event) => setSource(event.target.value)}>
            {sourceOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="eyebrow">predicted</span>
          <select className="field min-w-[110px]" value={digit} onChange={(event) => setDigit(event.target.value)}>
            <option value="all">Any digit</option>
            {Array.from({ length: 10 }, (_, index) => (
              <option key={index} value={String(index)}>
                Digit {index}
              </option>
            ))}
          </select>
        </label>
        <label className="flex h-[34px] cursor-pointer items-center gap-2 rounded-lg border border-line bg-white/70 px-2.5 text-xs text-muted">
          <input
            type="checkbox"
            checked={mismatch}
            onChange={(event) => setMismatch(event.target.checked)}
            className="h-3.5 w-3.5 accent-ink"
          />
          mismatches only
        </label>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <button type="button" className="btn" onClick={() => void loadItems()} disabled={loading}>
            {loading ? "Loading…" : "↻ Refresh"}
          </button>
          <button type="button" className="btn" onClick={exportCsv} disabled={items.length === 0}>
            ↓ Export CSV
          </button>
          {confirmClear ? (
            <>
              <button type="button" className="btn border-berry/40 bg-berry/10 text-berry" onClick={() => void clearAll()}>
                Confirm delete all
              </button>
              <button type="button" className="btn" onClick={() => setConfirmClear(false)}>
                Cancel
              </button>
            </>
          ) : (
            <button
              type="button"
              className="btn"
              onClick={() => setConfirmClear(true)}
              disabled={stats.total === 0}
            >
              Clear log
            </button>
          )}
        </div>
      </div>

      {error ? (
        <p className="rounded-lg border border-berry/30 bg-berry/5 px-3 py-2 text-xs text-berry">{error}</p>
      ) : null}

      <div className="flex flex-wrap items-center gap-2 px-1 text-xs text-muted">
        <span className="chip font-mono">{items.length} shown</span>
        <span className="chip font-mono">{stats.total.toLocaleString()} total rows</span>
        <span className="chip font-mono">{mismatches} mismatches</span>
        <span className="chip font-mono">
          avg {(items.reduce((sum, item) => sum + item.inferenceMs, 0) / (items.length || 1)).toFixed(2)} ms
        </span>
        <span className="ml-auto font-mono text-[10px]">set ground truth inline · saved via PATCH</span>
      </div>

      <div className="card overflow-hidden">
        <div className="hidden grid-cols-[56px_1fr_120px_130px_150px_120px_44px] gap-3 border-b border-line bg-paper-deep/50 px-4 py-2 lg:grid">
          {["input", "prediction", "confidence", "source", "ground truth", "timestamp", ""].map((label) => (
            <span key={label} className="eyebrow text-[10px]">
              {label}
            </span>
          ))}
        </div>

        {items.length === 0 ? (
          <div className="flex flex-col items-center gap-1 px-4 py-16 text-center">
            <span className="text-2xl">🗂️</span>
            <p className="text-sm font-medium">No predictions match these filters.</p>
            <p className="text-xs text-muted">Draw a digit in the studio and it will show up here instantly.</p>
          </div>
        ) : (
          <ul className="divide-y divide-line/70">
            {items.map((item) => (
              <li
                key={item.id}
                className="grid grid-cols-[48px_1fr_44px] items-center gap-3 px-4 py-2.5 transition hover:bg-paper-deep/30 lg:grid-cols-[56px_1fr_120px_130px_150px_120px_44px]"
              >
                <div className="flex h-11 w-11 items-center justify-center overflow-hidden rounded-lg border border-line bg-ink">
                  <DigitGlyph base64={item.pixels} size={44} className="h-full w-full" />
                </div>

                <div className="min-w-0">
                  <div className="flex items-center gap-1.5">
                    <span className="digit-face text-lg font-semibold">{item.predictedDigit}</span>
                    <span className="font-mono text-[10px] text-muted">#{item.id}</span>
                    {item.isCorrect === true ? (
                      <span className="rounded bg-emerald-100 px-1 font-mono text-[10px] text-emerald-800">✓ match</span>
                    ) : null}
                    {item.isCorrect === false ? (
                      <span className="rounded bg-berry/10 px-1 font-mono text-[10px] text-berry">
                        misread as {item.actualDigit}
                      </span>
                    ) : null}
                  </div>
                  <p className="truncate font-mono text-[10px] text-muted lg:hidden">
                    {formatTimestamp(item.createdAt)} UTC · {item.source}
                  </p>
                </div>

                <div className="hidden lg:block">
                  <div className="flex items-center gap-1.5">
                    <div className="h-1.5 w-14 overflow-hidden rounded-full bg-ink/10">
                      <div className="h-full rounded-full bg-ink" style={{ width: `${item.confidence * 100}%` }} />
                    </div>
                    <span className="font-mono text-[10px] tabular-nums text-muted">
                      {(item.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  <p className="mt-1 font-mono text-[9px] text-muted">{item.inferenceMs.toFixed(2)} ms</p>
                </div>

                <div className="hidden lg:block">
                  <SourceBadge source={item.source} />
                </div>

                <div className="hidden lg:block">
                  <select
                    value={item.actualDigit === null ? "" : String(item.actualDigit)}
                    onChange={(event) =>
                      void updateGroundTruth(
                        item.id,
                        event.target.value === "" ? null : Number(event.target.value),
                      )
                    }
                    disabled={busyId === item.id}
                    className="field py-1 text-xs"
                    aria-label={`Ground truth for prediction ${item.id}`}
                  >
                    <option value="">Unlabelled</option>
                    {Array.from({ length: 10 }, (_, index) => (
                      <option key={index} value={String(index)}>
                        Actual {index}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="hidden font-mono text-[10px] leading-tight text-muted lg:block">
                  {formatTimestamp(item.createdAt)}
                  <span className="block text-[9px] opacity-60">UTC</span>
                </div>

                <button
                  type="button"
                  onClick={() => void deleteRow(item.id)}
                  disabled={busyId === item.id}
                  title="Delete this prediction"
                  className="flex h-8 w-8 items-center justify-center rounded-lg border border-line text-muted transition hover:border-berry/40 hover:bg-berry/5 hover:text-berry disabled:opacity-40"
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
