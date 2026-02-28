import { useMemo, useState } from "react";
import Composer from "../components/Composer";
import Header from "../components/Header";
import ResultCard from "../components/ResultCard";
import Skeleton from "../components/Skeleton";
import { searchQuestion, type SearchResult } from "../lib/api";
import { formatMs } from "../lib/format";

type SearchError = {
  message: string;
  details?: string;
};

const EMPTY_RESULTS: SearchResult[] = [
  { rank: 1, score: 0, text: "" },
  { rank: 2, score: 0, text: "" },
  { rank: 3, score: 0, text: "" },
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [latencyMs, setLatencyMs] = useState<number | undefined>(undefined);
  const [error, setError] = useState<SearchError | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const visibleResults = useMemo(() => {
    if (results.length >= 3) return results.slice(0, 3);
    if (results.length === 0) return EMPTY_RESULTS;
    return [...results, ...EMPTY_RESULTS.slice(results.length, 3)].map((item, idx) => ({ ...item, rank: idx + 1 }));
  }, [results]);

  const runSearch = async () => {
    const trimmed = question.trim();
    if (!trimmed) return;

    setLoading(true);
    setError(null);

    try {
      const data = await searchQuestion(trimmed);
      setResults(data.results.slice(0, 3).map((item, idx) => ({ ...item, rank: idx + 1 })));
      setLatencyMs(data.meta.time_ms);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError({
        message: "Recherche indisponible.",
        details: message,
      });
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-4 pb-10 pt-6 sm:px-8 sm:pt-10">
      <div className="atlas-bg pointer-events-none fixed inset-0 -z-10" aria-hidden="true" />
      <Header lastLatencyMs={latencyMs} />

      <div className="mt-6 grid gap-5 lg:grid-cols-[1.2fr_1fr]">
        <Composer value={question} onChange={setQuestion} onSubmit={runSearch} loading={loading} />

        <section className="food-panel rounded-2xl border border-line bg-surface/90 p-5 backdrop-blur" aria-live="polite">
          <div className="mb-4 flex items-center justify-between">
            <h2 className="text-base text-muted">Top Matches</h2>
            {latencyMs !== undefined ? <p className="font-mono text-sm text-muted">{formatMs(latencyMs)}</p> : null}
          </div>

          {!loading && !error && results.length === 0 ? (
            <div className="food-panel flex min-h-[280px] flex-col items-center justify-center rounded-xl border border-dashed border-line bg-[#fff8f0] p-5 text-center">
              <svg className="h-24 w-24" viewBox="0 0 120 120" fill="none" aria-hidden="true">
                <defs>
                  <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="var(--teal)" />
                    <stop offset="100%" stopColor="var(--violet)" />
                  </linearGradient>
                </defs>
                <path d="M24 68c0-14 14-24 32-24s32 10 32 24v10H24V68Z" fill="#f2d9bb" stroke="url(#g)" strokeWidth="2" />
                <path d="M42 62v12M60 58v16M78 62v12" stroke="#c01e40" strokeWidth="2" strokeLinecap="round" />
                <path d="M20 84h80" stroke="url(#g)" strokeWidth="2" strokeLinecap="round" />
                <path d="M27 42c4-3 6-7 6-13m0 13c-4-3-6-7-6-13m60 13c4-3 6-7 6-13m0 13c-4-3-6-7-6-13" stroke="#b4803a" strokeWidth="2" strokeLinecap="round" />
              </svg>
              <p className="mt-3 text-base text-muted">Posez une question pour retrouver les passages clés de vos fiches techniques.</p>
            </div>
          ) : null}

          {loading ? (
            <div className="space-y-3">
              <Skeleton />
              <Skeleton />
              <Skeleton />
            </div>
          ) : null}

          {error ? (
            <div className="rounded-xl border border-red-400/30 bg-red-50 p-4 text-base text-red-700">
              <p>{error.message}</p>
              <div className="mt-3 flex items-center gap-2">
                <button
                  type="button"
                  onClick={runSearch}
                  className="rounded-md border border-red-300/60 px-3 py-1 text-sm transition hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-300/40"
                >
                  Retry
                </button>
                <button
                  type="button"
                  onClick={() => setShowDetails((v) => !v)}
                  className="rounded-md border border-red-300/60 px-3 py-1 text-sm transition hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-red-300/40"
                >
                  Voir détails
                </button>
              </div>
              {showDetails && error.details ? <pre className="mt-3 overflow-x-auto whitespace-pre-wrap text-sm">{error.details}</pre> : null}
            </div>
          ) : null}

          {!loading && !error && results.length > 0 ? (
            <div className="space-y-3">
              {visibleResults.map((item, idx) => (
                <ResultCard key={`${item.rank}-${idx}`} rank={item.rank} score={item.score} text={item.text} delayMs={idx * 70} />
              ))}
            </div>
          ) : null}
        </section>
      </div>
    </main>
  );
}
