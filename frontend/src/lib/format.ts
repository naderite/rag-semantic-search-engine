export function formatScore(score: number): string {
  const clamped = Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
  return clamped.toFixed(3);
}

export function formatMs(value: number): string {
  const safe = Number.isFinite(value) ? Math.max(0, value) : 0;
  return `${Math.round(safe)} ms`;
}
