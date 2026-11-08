"use client";

import { useEffect, useRef } from "react";

/** Visualises one pooled convolution filter activation map (12x12 or 13x13). */
export function FeatureMap({ values, label }: { values: number[]; label: string }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const side = Math.max(1, Math.round(Math.sqrt(values.length || 144)));

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const total = side * side;
    const image = ctx.createImageData(side, side);
    for (let i = 0; i < total; i++) {
      const v = Math.max(0, Math.min(1, values[i] ?? 0));
      image.data[i * 4] = Math.round(30 + v * 187);
      image.data[i * 4 + 1] = Math.round(34 + v * 213);
      image.data[i * 4 + 2] = Math.round(25 + v * 70);
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [values, side]);

  return (
    <div className="flex flex-col gap-1.5">
      <canvas
        ref={ref}
        width={side}
        height={side}
        className="pixel-canvas aspect-square w-full rounded-md ring-1 ring-white/10"
        aria-label={`Activation map ${label}`}
      />
      <span className="font-mono text-[10px] text-white/45">f{label}</span>
    </div>
  );
}
