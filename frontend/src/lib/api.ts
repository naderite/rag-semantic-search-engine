export type SearchResult = {
  text: string;
  score: number;
  rank: number;
};

export type SearchResponse = {
  question: string;
  results: SearchResult[];
  meta: {
    k: number;
    time_ms: number;
  };
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function searchQuestion(question: string, timeoutMs = 12000): Promise<SearchResponse> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch(`${API_BASE_URL}/search`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const fallback = `Request failed with status ${res.status}`;
      let detail = fallback;
      try {
        const payload = await res.json();
        detail = payload?.detail || fallback;
      } catch {
        detail = fallback;
      }
      throw new Error(detail);
    }

    return (await res.json()) as SearchResponse;
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new Error("Request timed out. Please retry.");
    }
    throw err;
  } finally {
    window.clearTimeout(timer);
  }
}
