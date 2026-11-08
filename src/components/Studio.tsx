"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { DigitGlyph } from "./DigitGlyph";
import { FeatureMap } from "./FeatureMap";
import { SourceBadge, StatTile } from "./visuals";
import type { ModelDto, PredictionDto, PredictResponse, SampleDto, StatsDto } from "@/lib/types";

const CANVAS_SIZE = 280;

type StudioProps = {
  initialRecent: PredictionDto[];
  initialStats: StatsDto;
  initialSamples: SampleDto[];
  model: ModelDto | null;
};

type Mode = "draw" | "upload";

function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const chunk = 4096;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function grayPayload(canvas: HTMLCanvasElement, background: "#000000" | "#ffffff", invert = false) {
  const temp = document.createElement("canvas");
  temp.width = canvas.width;
  temp.height = canvas.height;
  const ctx = temp.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("Canvas is not supported in this browser.");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, temp.width, temp.height);
  ctx.drawImage(canvas, 0, 0);
  const { data } = ctx.getImageData(0, 0, temp.width, temp.height);
  const gray = new Uint8Array(temp.width * temp.height);
  for (let i = 0; i < gray.length; i++) {
    const luminance = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
    gray[i] = Math.round(invert ? 255 - luminance : luminance);
  }
  return { image: bytesToBase64(gray), width: temp.width, height: temp.height };
}

function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleString("en-GB", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  });
}

function confidenceLabel(confidence: number): { text: string; className: string } {
  if (confidence >= 0.9) return { text: "high confidence", className: "text-lime" };
  if (confidence >= 0.6) return { text: "probable match", className: "text-lime/80" };
  return { text: "low confidence — re-draw", className: "text-sun" };
}

export function Studio({ initialRecent, initialStats, initialSamples, model }: StudioProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const undoStackRef = useRef<ImageData[]>([]);
  const autoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const uploadImageRef = useRef<HTMLImageElement | null>(null);

  const [mode, setMode] = useState<Mode>("draw");
  const [brushSize, setBrushSize] = useState(30);
  const [autoRead, setAutoRead] = useState(true);
  const [hasInk, setHasInk] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [recent, setRecent] = useState<PredictionDto[]>(initialRecent);
  const [stats, setStats] = useState<StatsDto>(initialStats);
  const [samples, setSamples] = useState<SampleDto[]>(initialSamples);
  const [busySampleId, setBusySampleId] = useState<number | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [uploadReady, setUploadReady] = useState(false);
  const [pickingTruth, setPickingTruth] = useState(false);
  const [feedbackBusy, setFeedbackBusy] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const gallerySamples = useMemo(() => {
    const seen = new Set<number>();
    return samples.filter((sample) => {
      if (seen.has(sample.label)) return false;
      seen.add(sample.label);
      return true;
    }).slice(0, 10);
  }, [samples]);

  const refreshAggregates = useCallback(async () => {
    try {
      const [recentResponse, statsResponse] = await Promise.all([
        fetch("/api/predictions?limit=8", { cache: "no-store" }),
        fetch("/api/stats", { cache: "no-store" }),
      ]);
      if (recentResponse.ok) {
        const data = (await recentResponse.json()) as { items: PredictionDto[] };
        setRecent(data.items);
      }
      if (statsResponse.ok) {
        setStats((await statsResponse.json()) as StatsDto);
      }
    } catch {
      // history refresh is best-effort
    }
  }, []);

  // ------------------------------------------------------------------ drawing
  const setUpCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = "#e6fa8c";
    ctx.lineWidth = brushSize;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    undoStackRef.current = [];
    setHasInk(false);
    setResult(null);
    setError(null);
  }, [brushSize]);

  useEffect(() => {
    setUpCanvas();
    // only reset on mount / brush change, not on every keystroke
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx) {
      ctx.strokeStyle = "#e6fa8c";
      ctx.lineWidth = brushSize;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
    }
  }, [brushSize]);

  const pointFromEvent = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const scale = canvas.width / rect.width;
    return {
      x: (event.clientX - rect.left) * scale,
      y: (event.clientY - rect.top) * scale,
    };
  };

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (mode !== "draw" || loading) return;
    event.preventDefault();
    const canvas = event.currentTarget;
    canvas.setPointerCapture(event.pointerId);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    if (undoStackRef.current.length > 7) undoStackRef.current.shift();
    undoStackRef.current.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
    drawingRef.current = true;
    const point = pointFromEvent(event);
    lastPointRef.current = point;
    ctx.beginPath();
    ctx.moveTo(point.x, point.y);
    ctx.lineTo(point.x + 0.01, point.y + 0.01);
    ctx.stroke();
    setHasInk(true);
    setResult(null);
    setError(null);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    const ctx = event.currentTarget.getContext("2d");
    if (!ctx || !lastPointRef.current) return;
    const point = pointFromEvent(event);
    ctx.beginPath();
    ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    lastPointRef.current = point;
  };

  const finishStroke = () => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastPointRef.current = null;
    setHasInk(true);
    if (autoRead) {
      if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
      autoTimerRef.current = setTimeout(() => {
        void predictDrawing();
      }, 360);
    }
  };

  // ------------------------------------------------------------------- upload
  const loadUploadedFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        setError("Please choose an image file (PNG, JPG, WEBP or GIF).");
        return;
      }
      if (file.size > 12 * 1024 * 1024) {
        setError("That image is larger than 12 MB. Try a smaller photo.");
        return;
      }
      setError(null);
      const reader = new FileReader();
      reader.onload = () => {
        const image = new Image();
        image.onload = () => {
          uploadImageRef.current = image;
          setUploadPreview(typeof reader.result === "string" ? reader.result : null);
          setUploadReady(true);
          setMode("upload");
          setResult(null);
          // Classify straight away — the deliberate "Read digit" button stays available.
          if (autoRead) {
            window.setTimeout(() => {
              void predictUpload();
            }, 30);
          }
        };
        image.onerror = () => setError("That image could not be decoded.");
        image.src = String(reader.result ?? "");
      };
      reader.onerror = () => setError("Could not read that file.");
      reader.readAsDataURL(file);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [autoRead],
  );

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) loadUploadedFile(file);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragActive(false);
    const file = event.dataTransfer.files?.[0];
    if (file) loadUploadedFile(file);
  };

  // ---------------------------------------------------------------- inference
  const postPrediction = useCallback(
    async (payload: Record<string, unknown>) => {
      setLoading(true);
      setError(null);
      setPickingTruth(false);
      try {
        const response = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = (await response.json()) as PredictResponse & { error?: string };
        if (!response.ok) {
          setError(data.error ?? "The model could not read that. Try again.");
          return;
        }
        setResult(data);
        void refreshAggregates();
      } catch {
        setError("Network error — could not reach the inference API.");
      } finally {
        setLoading(false);
      }
    },
    [refreshAggregates],
  );

  async function predictDrawing() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const payload = grayPayload(canvas, "#000000");
    await postPrediction({ ...payload, source: "draw" });
  }

  async function predictUpload() {
    const image = uploadImageRef.current;
    if (!image) return;
    const canvas = document.createElement("canvas");
    canvas.width = CANVAS_SIZE;
    canvas.height = CANVAS_SIZE;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    const scale = Math.min(CANVAS_SIZE / image.width, CANVAS_SIZE / image.height);
    const drawWidth = image.width * scale;
    const drawHeight = image.height * scale;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(
      image,
      (CANVAS_SIZE - drawWidth) / 2,
      (CANVAS_SIZE - drawHeight) / 2,
      drawWidth,
      drawHeight,
    );
    const payload = grayPayload(canvas, "#ffffff");
    await postPrediction({ ...payload, source: "upload" });
  }

  async function predictSample(sampleId: number) {
    setBusySampleId(sampleId);
    try {
      await postPrediction({ sampleId });
    } finally {
      setBusySampleId(null);
    }
  }

  const handlePrimaryAction = async () => {
    setPickingTruth(false);
    if (mode === "draw") {
      await predictDrawing();
    } else {
      await predictUpload();
    }
  };

  const clearCanvas = () => {
    if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
    setUpCanvas();
  };

  const undo = () => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const snapshot = undoStackRef.current.pop();
    if (!snapshot) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      setHasInk(false);
      setResult(null);
      return;
    }
    ctx.putImageData(snapshot, 0, 0);
    setHasInk(true);
    setResult(null);
  };

  const removeUpload = () => {
    uploadImageRef.current = null;
    setUploadPreview(null);
    setUploadReady(false);
    setResult(null);
    setError(null);
  };

  const submitFeedback = async (actualDigit: number | null) => {
    if (!result) return;
    setFeedbackBusy(true);
    try {
      const response = await fetch(`/api/predictions/${result.prediction.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ actualDigit }),
      });
      const data = (await response.json()) as { prediction?: PredictionDto; error?: string };
      if (!response.ok || !data.prediction) {
        setError(data.error ?? "Could not save that feedback.");
        return;
      }
      setResult({ ...result, prediction: data.prediction });
      setPickingTruth(false);
      void refreshAggregates();
    } catch {
      setError("Network error while saving feedback.");
    } finally {
      setFeedbackBusy(false);
    }
  };

  useEffect(() => {
    return () => {
      if (autoTimerRef.current) clearTimeout(autoTimerRef.current);
    };
  }, []);

  const activePrediction = result?.prediction ?? null;
  const confidence = activePrediction ? confidenceLabel(activePrediction.confidence) : null;

  return (
    <div className="space-y-6">
      {/* --------------------------------------------------------------- hero */}
      <section className="flex flex-col gap-5 pb-1 md:flex-row md:items-end md:justify-between">
        <div className="max-w-2xl">
          <p className="eyebrow flex items-center gap-2">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-lime ring-2 ring-ink/10" />
            2-block CNN + 5-crop TTA · MNIST · {model?.parameters.toLocaleString() ?? "56,970"} parameters
          </p>
          <h1 className="mt-2 text-3xl font-semibold leading-[1.05] tracking-tight sm:text-4xl md:text-[2.75rem]">
            Draw a digit. Watch a CNN <span className="relative whitespace-nowrap">read it.<span className="absolute -bottom-1 left-0 h-[3px] w-full rounded-full bg-lime" /></span>
          </h1>
          <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted">
            Every stroke is box-cropped, scaled to 20 px, centred on its own mass and pushed through
            a trained convolution stack. Predictions, probabilities and latencies are persisted to
            PostgreSQL.
          </p>
        </div>
        <div className="grid grid-cols-3 gap-2 md:w-[330px]">
          <StatTile label="test acc" value={`${((model?.testAccuracy ?? 0.9931) * 100).toFixed(2)}%`} hint={`${model?.testSamples.toLocaleString() ?? "10,000"} held out`} />
          <StatTile label="reads saved" value={stats.total.toLocaleString()} hint="row per inference" />
          <StatTile label="live p50" value={`${(stats.avgInferenceMs || 0).toFixed(1)}ms`} hint="conv → softmax" />
        </div>
      </section>

      {/* ------------------------------------------------------------ workshop */}
      <section className="grid gap-4 lg:grid-cols-[1.05fr_1fr]">
        <div className="card flex flex-col overflow-hidden">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="eyebrow">/ input</span>
              <div className="flex rounded-lg border border-line bg-paper-deep/60 p-0.5">
                {(["draw", "upload"] as Mode[]).map((option) => (
                  <button
                    key={option}
                    type="button"
                    onClick={() => {
                      setMode(option);
                      setError(null);
                    }}
                    className={`rounded-md px-2.5 py-1 text-xs font-semibold capitalize transition ${
                      mode === option ? "bg-white text-ink shadow-sm" : "text-muted hover:text-ink"
                    }`}
                  >
                    {option === "draw" ? "Draw" : "Upload photo"}
                  </button>
                ))}
              </div>
            </div>
            <label className="flex cursor-pointer items-center gap-2 text-[11px] text-muted">
              <input
                type="checkbox"
                checked={autoRead}
                onChange={(event) => setAutoRead(event.target.checked)}
                className="h-3.5 w-3.5 accent-ink"
              />
              auto-read
            </label>
          </div>

          <div className="p-4">
            {mode === "draw" ? (
              <div className="relative overflow-hidden rounded-xl bg-[#0f120c] ring-1 ring-inset ring-white/10">
                <div className="pointer-events-none absolute inset-0 opacity-30">
                  <div className="absolute left-1/2 top-0 h-full w-px bg-gradient-to-b from-transparent via-white/30 to-transparent" />
                  <div className="absolute top-1/2 left-0 h-px w-full bg-gradient-to-r from-transparent via-white/30 to-transparent" />
                  <div className="absolute inset-[14%] rounded-full border border-dashed border-white/15" />
                </div>
                {loading ? (
                  <div className="scanline pointer-events-none absolute inset-x-0 top-0 z-10 h-10 bg-gradient-to-b from-transparent via-lime/40 to-transparent" />
                ) : null}
                <canvas
                  ref={canvasRef}
                  width={CANVAS_SIZE}
                  height={CANVAS_SIZE}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={finishStroke}
                  onPointerCancel={finishStroke}
                  onPointerLeave={finishStroke}
                  className="relative z-[5] block aspect-square w-full cursor-crosshair touch-none select-none"
                  aria-label="Drawing canvas for a handwritten digit"
                />
                <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-center justify-between px-3 py-2 font-mono text-[10px] text-white/35">
                  <span>{hasInk ? "ink detected" : "280 × 280 frame"}</span>
                  <span>brush {brushSize}px</span>
                </div>
              </div>
            ) : (
              <div
                onDragOver={(event) => {
                  event.preventDefault();
                  setDragActive(true);
                }}
                onDragLeave={() => setDragActive(false)}
                onDrop={handleDrop}
                className={`relative flex aspect-square w-full flex-col items-center justify-center gap-3 overflow-hidden rounded-xl border-2 border-dashed p-4 text-center transition ${
                  dragActive ? "border-ink bg-lime/20" : "border-line bg-paper-deep/40"
                }`}
              >
                {uploadPreview ? (
                  <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={uploadPreview}
                      alt="Uploaded digit"
                      className="max-h-[82%] max-w-full rounded-lg object-contain shadow-sm"
                    />
                    <div className="absolute right-2 top-2 flex gap-1.5">
                      <button type="button" className="btn px-2 py-1 text-[11px]" onClick={() => fileInputRef.current?.click()}>
                        Replace
                      </button>
                      <button type="button" className="btn px-2 py-1 text-[11px]" onClick={removeUpload}>
                        Remove
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <span className="flex h-12 w-12 items-center justify-center rounded-xl border border-line bg-white text-xl">
                      ⬆
                    </span>
                    <div>
                      <p className="text-sm font-semibold">Drop a photo of a digit here</p>
                      <p className="mt-1 text-xs text-muted">
                        PNG, JPG or WEBP · dark ink on light paper works best
                      </p>
                    </div>
                    <button type="button" className="btn" onClick={() => fileInputRef.current?.click()}>
                      Choose an image
                    </button>
                  </>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </div>
            )}

            <div className="mt-3 flex flex-wrap items-center gap-2">
              {mode === "draw" ? (
                <>
                  <div className="flex items-center gap-2 rounded-lg border border-line bg-white/70 px-2.5 py-1.5">
                    <span className="font-mono text-[10px] uppercase tracking-wider text-muted">brush</span>
                    <input
                      type="range"
                      min={12}
                      max={48}
                      step={2}
                      value={brushSize}
                      onChange={(event) => setBrushSize(Number(event.target.value))}
                      className="h-1 w-20 accent-ink"
                      aria-label="Brush size"
                    />
                    <span className="font-mono text-[10px] text-muted">{brushSize}</span>
                  </div>
                  <button type="button" className="btn" onClick={undo} disabled={loading}>
                    ↶ Undo
                  </button>
                  <button type="button" className="btn" onClick={clearCanvas} disabled={loading}>
                    Clear
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="btn"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={loading}
                >
                  Choose another image
                </button>
              )}

              <button
                type="button"
                onClick={handlePrimaryAction}
                disabled={loading || (mode === "draw" ? !hasInk : !uploadReady)}
                className="btn btn-primary ml-auto min-w-[136px]"
              >
                {loading ? (
                  <>
                    <span className="h-3 w-3 animate-spin rounded-full border-2 border-lime/40 border-t-lime" />
                    Reading…
                  </>
                ) : (
                  <>Read digit →</>
                )}
              </button>
            </div>

            {error ? (
              <p className="mt-3 rounded-lg border border-berry/30 bg-berry/5 px-3 py-2 text-xs text-berry">
                {error}
              </p>
            ) : null}

            <div className="mt-4 border-t border-dashed border-line pt-3">
              <div className="flex items-center justify-between">
                <p className="eyebrow">/ try an unseen MNIST digit</p>
                <button
                  type="button"
                  className="font-mono text-[10px] text-muted underline decoration-dotted underline-offset-2 hover:text-ink"
                  onClick={() => {
                    const random = samples[Math.floor(Math.random() * samples.length)];
                    if (random) void predictSample(random.id);
                  }}
                  disabled={loading}
                >
                  random sample
                </button>
              </div>
              <div className="mt-2 grid grid-cols-5 gap-1.5 sm:grid-cols-10">
                {gallerySamples.map((sample) => (
                  <button
                    key={sample.id}
                    type="button"
                    onClick={() => void predictSample(sample.id)}
                    disabled={loading}
                    title="Classify this held-out MNIST test digit"
                    className={`group relative flex aspect-square items-center justify-center overflow-hidden rounded-lg border bg-white p-0.5 transition hover:-translate-y-0.5 hover:border-ink/40 hover:shadow-sm disabled:opacity-50 ${
                      busySampleId === sample.id ? "border-ink ring-2 ring-lime" : "border-line"
                    }`}
                  >
                    <DigitGlyph base64={sample.pixels} size={34} className="h-full w-full" />
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* --------------------------------------------------------- verdict */}
        <div className="card-dark flex flex-col overflow-hidden">
          <div className="flex items-center justify-between gap-2 border-b border-white/10 px-4 py-3">
            <span className="eyebrow text-white/45">/ prediction</span>
            <div className="flex items-center gap-1.5">
              <span className="chip chip-dark font-mono">{result?.modelVersion ?? model?.version ?? "cnn-mnist-v2"}</span>
              <span className="chip chip-dark">
                <span className={`h-1.5 w-1.5 rounded-full ${loading ? "bg-lime" : "bg-emerald-400"}`} />
                {loading ? "inferring" : "ready"}
              </span>
            </div>
          </div>

          <div className="flex flex-1 flex-col gap-4 p-4">
            <div className="flex items-center gap-4">
              <div className="relative flex h-[124px] w-[124px] shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-white/[0.03]">
                <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-lime/40 to-transparent" />
                {loading ? (
                  <div className="h-10 w-10 animate-spin rounded-full border-2 border-lime/20 border-t-lime" />
                ) : activePrediction ? (
                  <span key={activePrediction.id} className="digit-face animate-rise text-[92px] font-semibold text-lime">
                    {activePrediction.predictedDigit}
                  </span>
                ) : (
                  <span className="digit-face text-5xl text-white/20">?</span>
                )}
                <span className="absolute bottom-1.5 right-2 font-mono text-[9px] uppercase tracking-widest text-white/30">
                  predicted
                </span>
              </div>

              <div className="min-w-0 flex-1">
                {activePrediction ? (
                  <>
                    <div className="flex items-baseline gap-1.5">
                      <span className="font-mono text-3xl font-semibold tabular-nums text-white">
                        {(activePrediction.confidence * 100).toFixed(1)}
                      </span>
                      <span className="font-mono text-sm text-white/50">%</span>
                      <span className="ml-1 text-[11px] text-white/40">confidence</span>
                    </div>
                    <p className={`mt-0.5 font-mono text-[11px] ${confidence?.className}`}>{confidence?.text}</p>
                    <div className="mt-2 flex flex-wrap items-center gap-1.5">
                      <SourceBadge source={activePrediction.source} />
                      <span className="chip chip-dark font-mono">
                        {result?.inferenceMs.toFixed(2)} ms
                      </span>
                      {result?.preprocess ? (
                        <span className="chip chip-dark font-mono">
                          pre {result.preprocess.ms.toFixed(1)} ms
                        </span>
                      ) : null}
                    </div>
                    <p className="mt-2 font-mono text-[10px] text-white/35">
                      read #{activePrediction.id} · {formatTimestamp(activePrediction.createdAt)} UTC
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-sm font-medium text-white/85">No read yet</p>
                    <p className="mt-1 text-xs leading-relaxed text-white/45">
                      Draw a single digit or upload a photo. The CNN returns a full softmax
                      distribution over 0–9 in a few milliseconds.
                    </p>
                    <div className="mt-2 flex gap-1.5">
                      {[0, 1, 2, 3].map((digit) => (
                        <span
                          key={digit}
                          className="flex h-6 w-5 items-center justify-center rounded border border-white/10 font-mono text-[10px] text-white/25"
                        >
                          {digit}
                        </span>
                      ))}
                      <span className="font-mono text-[10px] text-white/25">…</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div>
              <div className="mb-1.5 flex items-center justify-between">
                <p className="eyebrow text-white/40">/ softmax output · 10 classes</p>
                {activePrediction ? (
                  <span className="font-mono text-[10px] text-white/30">
                    margin{" "}
                    {(
                      (activePrediction.probabilities[activePrediction.predictedDigit] -
                        [...activePrediction.probabilities].sort((a, b) => b - a)[1]) *
                      100
                    ).toFixed(1)}
                    pts
                  </span>
                ) : null}
              </div>
              <div className="rounded-xl border border-white/10 bg-black/20 p-3">
                {activePrediction ? (
                  <ProbabilityBarsOnDark
                    probabilities={activePrediction.probabilities}
                    predictedDigit={activePrediction.predictedDigit}
                  />
                ) : (
                  <p className="py-6 text-center font-mono text-[11px] text-white/25">
                    awaiting inference…
                  </p>
                )}
              </div>
            </div>

            <div className="mt-auto">
              {activePrediction ? (
                activePrediction.actualDigit !== null ? (
                  <div
                    className={`flex items-center justify-between gap-2 rounded-xl border px-3 py-2 text-xs ${
                      activePrediction.isCorrect
                        ? "border-lime/30 bg-lime/10 text-lime"
                        : "border-berry/40 bg-berry/10 text-[#ffb4ae]"
                    }`}
                  >
                    <span>
                      Ground truth is <span className="font-mono font-bold">{activePrediction.actualDigit}</span> ·{" "}
                      {activePrediction.isCorrect ? "correct" : "misread"}
                    </span>
                    <button
                      type="button"
                      className="font-mono text-[10px] text-white/50 underline decoration-dotted hover:text-white"
                      onClick={() => void submitFeedback(null)}
                      disabled={feedbackBusy || activePrediction.source === "sample"}
                    >
                      reset
                    </button>
                  </div>
                ) : (
                  <div className="rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] text-white/60">Was this read correct?</span>
                      {!pickingTruth ? (
                        <div className="flex gap-1.5">
                          <button
                            type="button"
                            className="btn btn-lime px-2.5 py-1 text-[11px]"
                            onClick={() => void submitFeedback(activePrediction.predictedDigit)}
                            disabled={feedbackBusy}
                          >
                            ✓ Yes
                          </button>
                          <button
                            type="button"
                            className="btn border-white/20 bg-white/10 px-2.5 py-1 text-[11px] text-white hover:bg-white/20"
                            onClick={() => setPickingTruth(true)}
                            disabled={feedbackBusy}
                          >
                            ✕ No
                          </button>
                        </div>
                      ) : null}
                    </div>
                    {pickingTruth ? (
                      <div className="mt-2">
                        <p className="font-mono text-[10px] text-white/40">
                          Pick the correct digit → saved to PostgreSQL
                        </p>
                        <div className="mt-1.5 grid grid-cols-10 gap-1">
                          {Array.from({ length: 10 }, (_, digit) => (
                            <button
                              key={digit}
                              type="button"
                              onClick={() => void submitFeedback(digit)}
                              disabled={feedbackBusy}
                              className="rounded-md border border-white/15 bg-white/5 py-1 font-mono text-xs text-white/80 transition hover:border-lime hover:bg-lime hover:text-ink disabled:opacity-50"
                            >
                              {digit}
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : null}
                  </div>
                )
              ) : (
                <div className="grid grid-cols-3 gap-2">
                  <MiniFact label="input" value="28×28" />
                  <MiniFact label="conv" value="16×5² · 32×3²" />
                  <MiniFact label="dense" value="64 → 10" />
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      {/* --------------------------------------------------- inside the network */}
      <section className="card-dark overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/10 px-4 py-3">
          <div className="flex items-center gap-2">
            <span className="eyebrow text-white/45">/ inside the network</span>
            <span className="chip chip-dark font-mono">conv1(5×5) → pool → conv2(3×3) → pool → fc(64) → softmax</span>
          </div>
          <span className="font-mono text-[10px] text-white/35">
            {result ? `forward pass ${result.inferenceMs.toFixed(2)} ms` : "run a prediction to inspect activations"}
          </span>
        </div>
        <div className="grid gap-4 p-4 lg:grid-cols-[170px_1fr_230px]">
          <div className="space-y-2">
            <p className="font-mono text-[10px] uppercase tracking-widest text-white/35">model input</p>
            <div className="flex h-[150px] w-[150px] items-center justify-center rounded-xl border border-white/10 bg-black/30 p-2">
              {activePrediction ? (
                <DigitGlyph
                  base64={activePrediction.pixels}
                  size={128}
                  className="rounded-md ring-1 ring-white/10"
                  title="Preprocessed 28×28 input"
                />
              ) : (
                <span className="font-mono text-[10px] text-white/20">28 × 28 px</span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <MiniFact label="ink px" value={result?.preprocess ? String(result.preprocess.inkPixels) : "—"} />
              <MiniFact label="threshold" value={result?.preprocess ? String(result.preprocess.threshold) : "—"} />
              <MiniFact label="bbox" value={result?.preprocess ? `${result.preprocess.bbox.width}×${result.preprocess.bbox.height}` : "—"} />
              <MiniFact label="inverted" value={result?.preprocess ? (result.preprocess.inverted ? "yes" : "no") : "—"} />
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <p className="font-mono text-[10px] uppercase tracking-widest text-white/35">
                16 conv1 filters · 12×12 pooled activations
              </p>
              <span className="font-mono text-[10px] text-white/25">bright = strong match</span>
            </div>
            <div className="grid grid-cols-4 gap-2 rounded-xl border border-white/10 bg-black/20 p-3 sm:grid-cols-8">
              {result
                ? result.featureMaps.map((map, index) => (
                    <FeatureMap key={index} values={map} label={String(index)} />
                  ))
                : Array.from({ length: 16 }, (_, index) => (
                    <div
                      key={index}
                      className="aspect-square w-full rounded-md border border-dashed border-white/10 bg-white/[0.02]"
                    />
                  ))}
            </div>
          </div>

          <div className="space-y-2">
            <p className="font-mono text-[10px] uppercase tracking-widest text-white/35">pipeline</p>
            <ol className="space-y-1.5">
              {[
                "Luminance + Otsu threshold",
                "4×4 supersampled 20 px scale",
                "Centre of mass + peak contrast",
                "Conv1 5×5 ×16 + MaxPool → 12×12",
                "Conv2 3×3 ×32 + MaxPool → 5×5",
                "Dense 800 → 64 ReLU",
                "Dense 64 → 10 Softmax + 5-crop TTA",
              ].map((step, index) => (
                <li key={step} className="flex items-start gap-2 text-[11px] leading-snug text-white/55">
                  <span className="mt-0.5 font-mono text-[9px] text-lime/70">{String(index + 1).padStart(2, "0")}</span>
                  {step}
                </li>
              ))}
            </ol>
            <div className="rounded-lg border border-white/10 bg-white/[0.04] p-2 font-mono text-[10px] leading-relaxed text-white/45">
              {result
                ? `${result.inferenceMs.toFixed(2)} ms inference · preprocess ${
                    result.preprocess ? result.preprocess.ms.toFixed(1) + " ms" : "skipped"
                  }`
                : "All inference runs server-side in TypeScript — no TensorFlow, no API keys."}
            </div>
          </div>
        </div>
      </section>

      {/* -------------------------------------------------------- recent reads */}
      <section className="grid gap-4 lg:grid-cols-[1fr_1.6fr]">
        <div className="card p-4">
          <div className="flex items-center justify-between">
            <p className="eyebrow">/ session & store</p>
            <span className="chip font-mono">{stats.total.toLocaleString()} rows</span>
          </div>
          <div className="mt-3 grid grid-cols-2 gap-2">
            <StatTile
              label="labelled accuracy"
              value={stats.feedbackAccuracy === null ? "—" : `${(stats.feedbackAccuracy * 100).toFixed(1)}%`}
              hint={`${stats.feedbackCorrect}/${stats.feedbackTotal} human labels`}
            />
            <StatTile
              label="sample accuracy"
              value={stats.sampleAccuracy === null ? "—" : `${(stats.sampleAccuracy * 100).toFixed(0)}%`}
              hint={`${stats.sampleCorrect}/${stats.sampleTotal} MNIST tests`}
            />
            <StatTile label="avg confidence" value={`${(stats.avgConfidence * 100).toFixed(1)}%`} hint="all stored reads" />
            <StatTile label="avg inference" value={`${(stats.avgInferenceMs || 0).toFixed(2)}ms`} hint="mean forward pass" />
          </div>
          <div className="mt-3 rounded-xl border border-line bg-paper-deep/40 p-3">
            <p className="font-mono text-[10px] uppercase tracking-widest text-muted">class distribution</p>
            <div className="mt-2 flex h-16 items-end gap-1.5">
              {stats.perDigit.map((item) => {
                const max = Math.max(...stats.perDigit.map((entry) => entry.count), 1);
                return (
                  <div key={item.digit} className="flex flex-1 flex-col items-center gap-1">
                    <div
                      className={`w-full rounded-t ${item.count ? "bg-ink" : "bg-ink/10"}`}
                      style={{ height: `${Math.max((item.count / max) * 48, 2)}px` }}
                      title={`${item.digit}: ${item.count} reads`}
                    />
                    <span className="font-mono text-[9px] text-muted">{item.digit}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="card p-4">
          <div className="flex items-center justify-between">
            <p className="eyebrow">/ recent reads</p>
            <Link href="/history" className="text-xs font-semibold text-ink underline decoration-lime decoration-2 underline-offset-4 hover:decoration-ink">
              Full history →
            </Link>
          </div>
          {recent.length === 0 ? (
            <div className="mt-6 flex flex-col items-center gap-1 py-8 text-center">
              <span className="text-2xl">✍️</span>
              <p className="text-sm font-medium">The store is empty.</p>
              <p className="text-xs text-muted">Your first read will be written to PostgreSQL automatically.</p>
            </div>
          ) : (
            <ul className="mt-2 divide-y divide-line/70">
              {recent.map((item) => (
                <li key={item.id} className="flex items-center gap-3 py-2">
                  <span className="flex h-10 w-10 items-center justify-center overflow-hidden rounded-lg border border-line bg-ink">
                    <DigitGlyph base64={item.pixels} size={40} className="h-full w-full" />
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="font-mono text-sm font-semibold">{item.predictedDigit}</span>
                      <span className="font-mono text-[11px] text-muted">{(item.confidence * 100).toFixed(1)}%</span>
                      <SourceBadge source={item.source} />
                      {item.isCorrect === false ? (
                        <span className="rounded bg-berry/10 px-1 font-mono text-[10px] text-berry">
                          ≠ {item.actualDigit}
                        </span>
                      ) : null}
                      {item.isCorrect === true ? (
                        <span className="font-mono text-[10px] text-emerald-700">✓</span>
                      ) : null}
                    </div>
                    <p className="font-mono text-[10px] text-muted">
                      #{item.id} · {formatTimestamp(item.createdAt)} UTC · {item.inferenceMs.toFixed(2)} ms
                    </p>
                  </div>
                  <div className="hidden w-24 sm:block">
                    <div className="h-1 overflow-hidden rounded-full bg-ink/10">
                      <div className="h-full rounded-full bg-ink" style={{ width: `${item.confidence * 100}%` }} />
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </div>
  );
}

function MiniFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.04] px-2 py-1.5">
      <p className="font-mono text-[9px] uppercase tracking-wider text-white/30">{label}</p>
      <p className="font-mono text-[11px] font-semibold text-white/80">{value}</p>
    </div>
  );
}

/** Lime-on-dark version of the probability bars, tuned for the verdict panel. */
function ProbabilityBarsOnDark({
  probabilities,
  predictedDigit,
}: {
  probabilities: number[];
  predictedDigit: number;
}) {
  const max = Math.max(...probabilities, 0.0001);
  return (
    <div className="space-y-1">
      {probabilities.map((probability, digit) => {
        const winner = digit === predictedDigit;
        const width = Math.max((probability / max) * 100, probability > 0 ? 1.5 : 0);
        return (
          <div key={digit} className="flex items-center gap-2">
            <span
              className={`w-3.5 font-mono text-[11px] tabular-nums ${
                winner ? "font-bold text-lime" : "text-white/40"
              }`}
            >
              {digit}
            </span>
            <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/10">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  winner ? "bg-lime" : probability > 0.05 ? "bg-white/40" : "bg-white/20"
                }`}
                style={{ width: `${width}%` }}
              />
            </div>
            <span
              className={`w-11 text-right font-mono text-[10px] tabular-nums ${
                winner ? "font-semibold text-white" : "text-white/35"
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
