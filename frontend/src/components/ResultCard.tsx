import { useState } from "react";
import ScoreViz from "./ScoreViz";

type ResultCardProps = {
  rank: number;
  score: number;
  text: string;
  delayMs: number;
};

export default function ResultCard({ rank, score, text, delayMs }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false);

  const copyText = async () => {
    if (!text) return;
    await navigator.clipboard.writeText(text);
  };

  return (
    <article
      className="result-card-food group rounded-2xl border border-line bg-surface/95 p-4 shadow-[0_12px_24px_rgba(74,35,19,.1)] transition hover:-translate-y-0.5 hover:border-teal/65 hover:shadow-glow motion-safe:animate-floatIn"
      style={{ animationDelay: `${delayMs}ms` }}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="rounded-lg border border-line bg-[#fff4f1] px-2 py-1 font-mono text-sm text-teal">#{rank}</span>
          <p className="text-sm text-muted">Baking signal</p>
        </div>
        <ScoreViz score={score} />
      </div>

      <p className={`mt-3 text-base leading-7 text-text ${expanded ? "" : "line-clamp-4"}`}>{text || "Aucun fragment retourné."}</p>

      <div className="mt-3 flex items-center gap-2">
        <button
          type="button"
          onClick={() => setExpanded((v) => !v)}
          className="rounded-md border border-line px-2 py-1 text-sm text-muted transition hover:border-violet/80 hover:text-text focus:outline-none focus:ring-2 focus:ring-violet/30"
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
        <button
          type="button"
          onClick={copyText}
          disabled={!text}
          className="rounded-md border border-line px-2 py-1 text-sm text-muted transition hover:border-teal/80 hover:text-text disabled:cursor-not-allowed disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-teal/30"
        >
          Copy
        </button>
      </div>
    </article>
  );
}
