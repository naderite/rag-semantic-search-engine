import { useEffect, useMemo, useRef } from "react";

const EXAMPLES = [
  "I need better loaf volume. For the HCF line, what dosage range do bakers usually start with?",
  "I am checking compliance for ascorbic acid; what legal maximum dose is mentioned in these docs?",
  "When I scan these enzyme TDS files, what shelf life and storage temperature keep repeating?",
  "Could you compare the dosage ranges for TG MAX63, TG MAX64, and TG883?",
  "For HCB708, what effective material is listed, and what activity value is declared?",
  "For HCF MAX X alone, can you list the standardization, optimum, and bread-improvement dosage values?",
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
    <section className="food-panel rounded-2xl border border-line bg-surface/95 p-5 shadow-glow backdrop-blur">
      <label className="mb-3 block text-base text-muted" htmlFor="question-composer">
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
        className="min-h-[120px] w-full resize-none rounded-xl border border-line bg-[#fffdfa] px-4 py-3 text-base text-text outline-none transition focus:border-teal/80 focus:ring-2 focus:ring-teal/20"
      />
      <div className="mt-3 flex flex-wrap items-center gap-2">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => onChange(example)}
            className="rounded-full border border-line bg-[#fff8ee] px-3 py-1.5 text-sm text-muted transition hover:border-teal/70 hover:text-text focus:outline-none focus:ring-2 focus:ring-violet/35"
          >
            {example}
          </button>
        ))}
      </div>
      <div className="mt-4 flex items-center justify-between gap-2">
        <p className="text-sm text-muted">{hint}</p>
        <button
          type="button"
          onClick={onSubmit}
          disabled={loading}
          aria-label="Search"
          className="rounded-lg bg-gradient-to-r from-teal to-violet px-5 py-2 text-base font-medium text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-teal/40"
        >
          {loading ? "Searching..." : "Search"}
        </button>
      </div>
    </section>
  );
}
