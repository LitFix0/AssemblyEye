import type { Finding, Status } from "./types";

export const STATUS_TEXT: Record<Status, { title: string; hint: string }> = {
  PASS: { title: "PASS", hint: "All checked requirements are met." },
  REVIEW: { title: "REVIEW", hint: "Not certain. A person must look at this part." },
  FAIL: { title: "FAIL", hint: "A requirement is violated." },
};

/** FAIL findings first, then REVIEW; original order kept inside each group. */
export function sortFindings(findings: Finding[]): Finding[] {
  const rank = (f: Finding) => (f.status === "FAIL" ? 0 : 1);
  return [...findings].sort((a, b) => rank(a) - rank(b));
}

export function componentState(c: { expected: number; confident: number; uncertain: number }): "ok" | "check" | "wrong" {
  if (c.confident === c.expected && c.uncertain === 0) return "ok";
  if (c.confident > c.expected || c.confident + c.uncertain < c.expected) return "wrong";
  return "check";
}

export function fileProblem(file: File, maxBytes = 25 * 1024 * 1024): string | null {
  if (!/\.(png|jpe?g|bmp)$/i.test(file.name)) return "Choose a PNG, JPG or BMP image.";
  if (file.size === 0) return "That file is empty.";
  if (file.size > maxBytes) return "That file is larger than 25 MB.";
  return null;
}