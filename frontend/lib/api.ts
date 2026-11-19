// Typed client for the TaxLens API: auth token storage, JSON calls, and the
// server-sent-events answer stream.

export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Citation = {
  n: number;
  record_id: number;
  doc_id: string | null;
  title: string | null;
  tax_year: number | null;
  page: number | null;
  boxes: { page: number; x0: number; top: number; x1: number; bottom: number }[];
  line_label: string | null;
  heading_path: string | null;
  snippet: string | null;
};

export type AnswerSentence = { text: string; citations: number[] };

export type Status = "answered" | "abstained" | "needs_year" | "out_of_scope" | "year_unavailable";

export type Result = {
  status: Status;
  question: string;
  tax_year: number | null;
  year_source: string | null;
  intent: string | null;
  sentences: AnswerSentence[];
  citations: Citation[];
  message: string | null;
  mode: string | null;
  compare_years: number[] | null;
  disclaimer: string;
  query_id: number | null;
  cache_hit: boolean;
  latency_ms: number;
};

export type StreamEvent =
  | { type: "meta"; status: Status; tax_year: number | null; year_source: string | null;
      intent: string | null; message: string | null; compare_years: number[] | null;
      available_years: number[] }
  | { type: "sentence"; sentence: AnswerSentence; citations: Citation[] }
  | { type: "abstain"; message: string; reason: string }
  | { type: "error"; error: string; code: string }
  | { type: "done"; result: Result };

export type Meta = { index_loaded: boolean; error: string | null; years: number[]; llm: string;
                     allow_dev_tokens: boolean };

export type HistoryItem = { query_id: number; question: string; tax_year: number | null;
                            intent_type: string | null; abstained: boolean; created_at: string };

const TOKEN_KEY = "taxlens.token";
const USER_KEY = "taxlens.user";

export function getToken(): string | null {
  return typeof window === "undefined" ? null : sessionStorage.getItem(TOKEN_KEY);
}

export function getUser(): string | null {
  return typeof window === "undefined" ? null : sessionStorage.getItem(USER_KEY);
}

export function signOut() {
  sessionStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(USER_KEY);
}

export class ApiError extends Error {
  constructor(public status: number, message: string, public code?: string) {
    super(message);
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  if (init.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, `Can't reach the TaxLens API at ${API_URL}. Is it running?`);
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.error ?? res.statusText, body.code);
  }
  return res.json() as Promise<T>;
}

export async function signIn(username: string): Promise<void> {
  const data = await request<{ access_token: string; user: string }>("/v1/auth/dev-token", {
    method: "POST",
    body: JSON.stringify({ username }),
  });
  sessionStorage.setItem(TOKEN_KEY, data.access_token);
  sessionStorage.setItem(USER_KEY, data.user);
}

export const fetchMeta = () => request<Meta>("/v1/meta");

export const fetchHistory = () =>
  request<{ items: HistoryItem[]; has_more: boolean }>("/v1/history?limit=30");

export const sendFeedback = (query_id: number, rating: 1 | -1, comment?: string) =>
  request<{ status: string }>("/v1/feedback", {
    method: "POST",
    body: JSON.stringify({ query_id, rating, comment: comment || null }),
  });

/** Fetch a highlighted page image (needs the auth header, so it can't be a plain <img src>). */
export async function fetchPageImage(c: Citation): Promise<string> {
  if (!c.doc_id || !c.page) throw new ApiError(404, "This citation has no page to show.");
  const boxes = c.boxes
    .filter((b) => b.page === c.page)
    .map((b) => [b.x0, b.top, b.x1, b.bottom].map((v) => v.toFixed(1)).join(","))
    .join(";");
  const res = await fetch(
    `${API_URL}/v1/documents/${encodeURIComponent(c.doc_id)}/pages/${c.page}?boxes=${encodeURIComponent(boxes)}`,
    { headers: { Authorization: `Bearer ${getToken() ?? ""}` } },
  );
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.error ?? "Couldn't load the page.");
  }
  return URL.createObjectURL(await res.blob());
}

/** POST /v1/ask/stream and call onEvent for every server-sent event. */
export async function streamAnswer(question: string, taxYear: number | null,
                                   onEvent: (e: StreamEvent) => void, signal?: AbortSignal) {
  let res: Response;
  try {
    res = await fetch(`${API_URL}/v1/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${getToken() ?? ""}` },
      body: JSON.stringify({ question, tax_year: taxYear }),
      signal,
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") return;
    throw new ApiError(0, `Can't reach the TaxLens API at ${API_URL}. Is it running?`);
  }
  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(res.status, body.error ?? res.statusText, body.code);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      for (const line of chunk.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        const data = line.slice(6);
        if (data === "[DONE]") return;
        onEvent(JSON.parse(data) as StreamEvent);
      }
    }
  }
}
