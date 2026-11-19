import type { InspectionResponse, ProductInfo } from "./types";

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