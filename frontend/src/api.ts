import type { InspectionList, InspectionResponse, ProductInfo, Status } from "./types";

export class ApiRequestError extends Error {
  constructor(public code: string, message: string, public httpStatus: number) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    /* not JSON: handled below */
  }
  if (!res.ok) {
    const err = (body as { error?: { code?: string; message?: string } } | null)?.error;
    throw new ApiRequestError(err?.code ?? "http_error", err?.message ?? `request failed (HTTP ${res.status})`, res.status);
  }
  return body as T;
}

async function send(input: string, init?: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init);
  } catch {
    throw new ApiRequestError("network_error", "Cannot reach the inspection service. Is the backend running?", 0);
  }
}

export async function fetchProducts(): Promise<ProductInfo[]> {
  return parse<ProductInfo[]>(await send("/products"));
}

export async function inspectImage(product: string, file: File): Promise<InspectionResponse> {
  const form = new FormData();
  form.append("product", product);
  form.append("file", file);
  return parse<InspectionResponse>(await send("/inspect", { method: "POST", body: form }));
}

export interface HistoryFilter {
  product: string;
  status: Status | "";
  code: string;
  since: string; // ISO instant, inclusive; "" = no bound
  until: string; // ISO instant, exclusive; "" = no bound
  limit: number;
  offset: number;
}

export async function listInspections(f: HistoryFilter): Promise<InspectionList> {
  const q = new URLSearchParams();
  for (const [key, value] of Object.entries(f)) if (value !== "" && value !== undefined) q.set(key, String(value));
  return parse<InspectionList>(await send(`/inspections?${q.toString()}`));
}

export async function getInspection(id: string): Promise<InspectionResponse> {
  return parse<InspectionResponse>(await send(`/inspections/${encodeURIComponent(id)}`));
}

export async function deleteInspection(id: string): Promise<void> {
  const res = await send(`/inspections/${encodeURIComponent(id)}`, { method: "DELETE" });
  if (!res.ok) await parse<unknown>(res); // throws the API's error
}