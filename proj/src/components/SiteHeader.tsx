"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Studio" },
  { href: "/history", label: "History" },
  { href: "/model", label: "Model" },
];

export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-40 border-b border-line/80 bg-paper/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <Link href="/" className="group flex items-center gap-3">
          <span className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-ink text-lime shadow-sm">
            <span className="digit-face text-lg font-bold">7</span>
            <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-lime ring-2 ring-paper" />
          </span>
          <span className="leading-tight">
            <span className="block text-sm font-semibold tracking-tight">DigitLab</span>
            <span className="eyebrow block text-[10px]">CNN digit recognition</span>
          </span>
        </Link>

        <nav className="flex items-center gap-1 rounded-full border border-line bg-white/70 p-1">
          {links.map((link) => {
            const active =
              link.href === "/" ? pathname === "/" : pathname.startsWith(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                className={`rounded-full px-3 py-1.5 text-xs font-semibold transition ${
                  active
                    ? "bg-ink text-paper shadow-sm"
                    : "text-muted hover:bg-paper-deep hover:text-ink"
                }`}
              >
                {link.label}
              </Link>
            );
          })}
        </nav>

        <div className="hidden items-center gap-2 sm:flex">
          <span className="chip">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            model live
          </span>
          <span className="chip font-mono">99.3% test</span>
        </div>
      </div>
    </header>
  );
}
