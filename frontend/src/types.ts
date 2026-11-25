// Mirrors backend/app/schemas (inspection.py, api.py). If the API changes, change this file and the tests.
export type Status = "PASS" | "FAIL" | "REVIEW";

export interface Finding {
  code: string;
  status: "FAIL" | "REVIEW";
  class_name: string;
  expected: number | null;
  confident: number | null;
  uncertain: number | null;
  message: string;
  evidence: Record<string, unknown>;
}

export interface ComponentCount {
  class_name: string;
  expected: number;
  confident: number;
  uncertain: number;
}

export interface InspectionResult {
  product: string;
  spec_version: string;
  status: Status;
  image: { name: string; width: number; height: number };
  detections: { class_name: string; confidence: number }[];
  findings: Finding[];
  detector: string;
  components: ComponentCount[];
}

export interface InspectionResponse {
  inspection_id: string;
  created_at: string;
  result: InspectionResult;
  evidence: { original_url: string; annotated_url: string };
  timings_ms: Record<string, number>;
}

export interface ProductInfo {
  product: string;
  spec_version: string;
  description: string;
  weights_present: boolean;
  detector_loaded: boolean;
}

export interface InspectionSummary {
  inspection_id: string;
  created_at: string;
  product: string;
  spec_version: string;
  status: Status;
  image_name: string;
  detector: string;
  finding_codes: string[];
  total_ms: number;
}

export interface InspectionList {
  items: InspectionSummary[];
  total: number;
  limit: number;
  offset: number;
}