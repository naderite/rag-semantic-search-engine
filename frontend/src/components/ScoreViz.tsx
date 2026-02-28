import { formatScore } from "../lib/format";

type ScoreVizProps = {
  score: number;
};

export default function ScoreViz({ score }: ScoreVizProps) {
  const safeScore = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
  const activeBars = Math.max(1, Math.round(safeScore * 8));

  return (
    <div className="flex items-center gap-2" aria-label={`Similarity score ${formatScore(safeScore)}`}>
      <div className="flex h-8 items-end gap-1">
        {Array.from({ length: 8 }, (_, idx) => {
          const isActive = idx < activeBars;
          return (
            <span
              key={idx}
              className={`w-1.5 rounded-sm transition ${isActive ? "bg-gradient-to-t from-teal to-violet shadow-[0_0_8px_rgba(192,30,64,.35)]" : "bg-[#dccfc2]"}`}
              style={{ height: `${22 + ((idx % 4) * 12)}%` }}
            />
          );
        })}
      </div>
      <span className="font-mono text-sm text-text">{formatScore(safeScore)}</span>
    </div>
  );
}
