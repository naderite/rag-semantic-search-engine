type HeaderProps = {
  lastLatencyMs?: number;
};

export default function Header({ lastLatencyMs }: HeaderProps) {
  return (
    <header className="relative z-10 flex flex-col gap-5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <p className="inline-flex items-center gap-2 rounded-full border border-line/70 bg-surface/70 px-3 py-1 text-sm text-muted">
          <svg className="h-4 w-4 text-teal" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M12 4v16M12 7c-2 0-3.2 1.3-4 3m4-1c2 0 3.2 1.3 4 3m-4 1c-1.8 0-2.8 1.1-3.6 2.6m3.6-1.6c1.8 0 2.8 1.1 3.6 2.6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
          </svg>
          Semantic Atlas
        </p>
        <h1 className="mt-3 font-title text-4xl font-semibold tracking-tight text-text sm:text-5xl">Knowledge Retrieval Console</h1>
        <p className="mt-2 text-base text-muted">Retrouvez instantanément les passages les plus pertinents pour vos produits céréaliers.</p>
      </div>

      <div className="inline-flex items-center gap-3 rounded-full border border-line bg-surface/90 px-4 py-2 text-sm text-muted backdrop-blur">
        <span className="text-teal">Local RAG</span>
        <span className="text-violet">•</span>
        <span>FastAPI</span>
        {typeof lastLatencyMs === "number" ? <span className="font-mono text-text">{lastLatencyMs} ms</span> : null}
      </div>
    </header>
  );
}
