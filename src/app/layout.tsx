import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Space_Grotesk, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import { SiteHeader } from "@/components/SiteHeader";
import "./globals.css";

const display = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-grotesk",
  weight: ["400", "500", "600", "700"],
});

const body = IBM_Plex_Sans({
  subsets: ["latin"],
  variable: "--font-plex-sans",
  weight: ["400", "500", "600", "700"],
});

const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  variable: "--font-plex-mono",
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "DigitLab — Handwritten Digit Recognition with a CNN",
  description:
    "Draw a digit or upload a photo and watch a convolutional neural network trained on MNIST read it, with every prediction stored in PostgreSQL.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${display.variable} ${body.variable} ${mono.variable}`}>
      <body className="min-h-screen bg-paper text-ink antialiased">
        <div className="page-grid min-h-screen">
          <SiteHeader />
          <main className="mx-auto w-full max-w-6xl px-4 pb-16 pt-6 sm:px-6">{children}</main>
          <footer className="border-t border-line/80 bg-paper-deep/60">
            <div className="mx-auto flex w-full max-w-6xl flex-col gap-2 px-4 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
              <p>
                <span className="font-semibold text-ink">DigitLab</span> · CNN inference in
                pure TypeScript · Next.js + Drizzle + PostgreSQL
              </p>
              <p className="font-mono">
                MNIST test accuracy 99.31% · 56,970 parameters · 2-block CNN + TTA
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
