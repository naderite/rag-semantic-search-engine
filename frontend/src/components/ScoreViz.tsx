import { formatScore } from "../lib/format";

type ScoreVizProps = {
  score: number;
};

export default function ScoreViz({ score }: ScoreVizProps) {
  const safeScore = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
  const activeBars = Math.max(1, Math.round(safeScore * 10));

  return (
    <div className="flex items-center gap-2 rounded-full border border-line/60 bg-[#fff7ef] px-2.5 py-1" aria-label={`Similarity score ${formatScore(safeScore)}`}>
      <div className="flex h-7 items-end gap-1">
        {Array.from({ length: 10 }, (_, idx) => {
          const isActive = idx < activeBars;
          return (
            <span
              key={idx}
              className={`w-1 rounded-sm transition ${isActive ? "bg-gradient-to-t from-teal to-violet shadow-[0_0_8px_rgba(192,30,64,.35)]" : "bg-[#e7d9cb]"}`}
              style={{ height: `${18 + ((idx % 5) * 14)}%` }}
            />
          );
        })}
      </div>
      <span className="font-mono text-xs text-text">{formatScore(safeScore)}</span>
    </div>
  );
}
