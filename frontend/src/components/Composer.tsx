import { useEffect, useMemo, useRef } from "react";

const EXAMPLES = [
  "Could you compare the dosage ranges for TG MAX63, TG MAX64, and TG883?",
  "Pour HCB708, quel materiau actif est mentionne et quelle activite est declaree ?",
  "For HCF MAX X alone, can you list the standardization, optimum, and bread-improvement dosage values?",
  "Je verifie la conformite de l'acide ascorbique : quel dosage maximum legal est mentionne ?",
  "When I scan these enzyme TDS files, what shelf life and storage temperature keep repeating?",
];

type ComposerProps = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
};

export default function Composer({ value, onChange, onSubmit, loading }: ComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const node = textareaRef.current;
    if (!node) return;
    node.style.height = "auto";
    node.style.height = `${Math.min(node.scrollHeight, 260)}px`;
  }, [value]);

  const hint = useMemo(() => "Cmd/Ctrl + Enter pour lancer la recherche", []);

  return (
    <section className="food-panel rounded-2xl border border-line bg-surface/95 p-4 sm:p-5 shadow-glow backdrop-blur">
      <label className="mb-3 block text-lg text-muted" htmlFor="question-composer">
        Question
      </label>
      <textarea
        id="question-composer"
        ref={textareaRef}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
            e.preventDefault();
            onSubmit();
          }
        }}
        aria-label="Question"
        placeholder="Ex: dosage pain de mie, conservation farine, activite enzymatique..."
        className="min-h-[140px] w-full resize-none rounded-xl border border-line bg-[#fffdfa] px-4 py-3 text-[1.1rem] leading-9 text-text outline-none transition focus:border-teal/80 focus:ring-2 focus:ring-teal/20"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {EXAMPLES.map((example, idx) => (
          <button
            key={example}
            type="button"
            onClick={() => onChange(example)}
            className={`max-w-full rounded-full border px-3 py-2 text-left text-base leading-6 text-muted transition focus:outline-none focus:ring-2 ${
              idx % 2 === 0
                ? "border-[rgba(192,30,64,.22)] bg-[#fff8ee] hover:border-teal/70 hover:text-text focus:ring-violet/35"
                : "border-[rgba(248,180,63,.45)] bg-[#fff9e8] hover:border-violet/80 hover:text-text focus:ring-teal/30"
            }`}
          >
            {example}
          </button>
        ))}
      </div>
      <div className="mt-4 flex flex-col items-start justify-between gap-3 sm:flex-row sm:items-center">
        <p className="text-base text-muted">{hint}</p>
        <button
          type="button"
          onClick={onSubmit}
          disabled={loading}
          aria-label="Search"
          className="w-full rounded-lg bg-gradient-to-r from-teal to-violet px-6 py-2.5 text-lg font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-teal/40 sm:w-auto"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </section>
  );
}
