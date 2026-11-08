"use client";

import { useEffect, useRef } from "react";

type DigitGlyphProps = {
  base64?: string | null;
  pixels?: number[] | null;
  size?: number;
  className?: string;
  title?: string;
  invert?: boolean;
};

/** Renders a 28x28 grayscale MNIST tensor without any image assets. */
export function DigitGlyph({
  base64,
  pixels,
  size = 64,
  className = "",
  title,
  invert = false,
}: DigitGlyphProps) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let values: Uint8Array | null = null;
    if (pixels && pixels.length === 784) {
      values = Uint8Array.from(pixels, (v) => Math.max(0, Math.min(255, Math.round(v))));
    } else if (base64) {
      try {
        const binary = atob(base64.replace(/^data:[^;]+;base64,/, ""));
        if (binary.length === 784) {
          values = Uint8Array.from(binary, (char) => char.charCodeAt(0));
        }
      } catch {
        values = null;
      }
    }

    ctx.clearRect(0, 0, 28, 28);
    if (!values) {
      ctx.fillStyle = invert ? "#f4f2ea" : "#12140f";
      ctx.fillRect(0, 0, 28, 28);
      return;
    }

    const image = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const v = values[i];
      image.data[i * 4] = invert ? 255 - v : v;
      image.data[i * 4 + 1] = invert ? 255 - v : v;
      image.data[i * 4 + 2] = invert ? 255 - v : v;
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }, [base64, pixels, invert]);

  return (
    <canvas
      ref={ref}
      width={28}
      height={28}
      title={title}
      aria-label={title ?? "28 by 28 digit input"}
      className={`pixel-canvas ${className}`}
      style={{ width: size, height: size }}
    />
  );
}
