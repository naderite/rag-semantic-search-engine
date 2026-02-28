import { useState } from "react";

type ResultCardProps = {
  rank: number;
  score: number;
  text: string;
  delayMs: number;
};

function asPct(value?: number): string {
  const safe = Number.isFinite(value) ? Math.max(0, Math.min(1, value as number)) : 0;
  return `${Math.round(safe * 100)}%`;
}

export default function ResultCard({ rank, score, text, delayMs }: ResultCardProps) {
  const [expanded, setExpanded] = useState(true);

  const copyText = async () => {
    if (!text) return;
    await navigator.clipboard.writeText(text);
  };

  return (
    <article
      className="result-card-food group rounded-3xl border border-line/80 bg-white/85 p-4 sm:p-5 shadow-[0_12px_28px_rgba(84,43,23,.1)] transition hover:-translate-y-0.5 hover:border-teal/65 hover:shadow-glow motion-safe:animate-floatIn"
      style={{ animationDelay: `${delayMs}ms` }}
    >
      <div className="flex flex-col items-start justify-between gap-2 border-b border-line/50 pb-3 sm:flex-row sm:items-center">
        <div className="flex items-center gap-2">
          <span className="rounded-full border border-line bg-[#fff4f1] px-3 py-1.5 font-mono text-sm font-semibold text-teal">#{rank}</span>
          <p className="text-sm uppercase tracking-wide text-muted">Match</p>
        </div>
        <p className="inline-flex items-center rounded-full border border-line/70 bg-[#fff8f0] px-3.5 py-1.5 font-mono text-sm text-muted">
          Score: {asPct(score)}
        </p>
      </div>

      <p className={`readable-copy mt-3 text-[1.12rem] text-text ${expanded ? "" : "max-h-36 overflow-hidden"}`}>
        {text || "Aucun fragment retourné."}
      </p>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="rounded-lg border border-line px-3 py-2 text-base text-muted transition hover:border-violet/80 hover:text-text focus:outline-none focus:ring-2 focus:ring-violet/30 sm:min-w-[110px]"
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
        <button
          type="button"
          onClick={copyText}
          disabled={!text}
          className="rounded-lg border border-line px-3 py-2 text-base text-muted transition hover:border-teal/80 hover:text-text disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-teal/30 sm:min-w-[90px]"
        >
          Copy
        </button>
      </div>
    </article>
  );
}
