import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import type { InspectionResponse, ProductInfo } from "./types";

const PRODUCTS: ProductInfo[] = [
  { product: "screw_bag", spec_version: "0.2", description: "", weights_present: true, detector_loaded: false },
  { product: "splicing_connectors", spec_version: "0.2", description: "", weights_present: false, detector_loaded: false },
];

function response(status: "PASS" | "REVIEW" | "FAIL"): InspectionResponse {
  const findings =
    status === "PASS"
      ? []
      : [{ code: status === "FAIL" ? "MISSING" : "UNCERTAIN_PART", status: status as "FAIL" | "REVIEW",
           class_name: "washer", expected: 2, confident: 1, uncertain: 0, message: "washer: expected 2, found 1", evidence: { a: 1 } }];
  return {
    inspection_id: "0123456789abcdef0123456789abcdef", created_at: "2026-01-01T00:00:00Z",
    result: {
      product: "screw_bag", spec_version: "0.2", status, image: { name: "part.png", width: 400, height: 300 },
      detections: [], findings, detector: "fake", components: [{ class_name: "washer", expected: 2, confident: 1, uncertain: 0 }],
    },
    evidence: { original_url: "/inspections/x/original", annotated_url: "/inspections/x/annotated" },
    timings_ms: { load: 1, detect: 2, total: 3 },
  };
}

function mockFetch(handlers: Record<string, () => Response | Promise<Response>>) {
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    const h = handlers[url];
    if (!h) throw new Error(`unexpected fetch ${url}`);
    return Promise.resolve(h());
  }));
}
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

async function pickAndSubmit(name = "part.png") {
  await screen.findByRole("option", { name: "screw_bag" });
  fireEvent.change(screen.getByLabelText(/image/i), { target: { files: [new File(["x"], name, { type: "image/png" })] } });
  fireEvent.click(screen.getByRole("button", { name: "Inspect" }));
}

afterEach(() => vi.unstubAllGlobals());

describe("App", () => {
  it("lists products and keeps Inspect disabled until a file is chosen", async () => {
    mockFetch({ "/products": () => json(PRODUCTS) });
    render(<App />);
    expect(await screen.findByRole("option", { name: "screw_bag" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Inspect" })).toBeDisabled();
  });

  it.each(["PASS", "REVIEW", "FAIL"] as const)("shows a %s result with findings and the components table", async (status) => {
    mockFetch({ "/products": () => json(PRODUCTS), "/inspect": () => json(response(status)) });
    render(<App />);
    await pickAndSubmit();
    expect(await screen.findByRole("status")).toHaveTextContent(status);
    if (status === "PASS") expect(screen.getByText(/every check passed/i)).toBeInTheDocument();
    else expect(screen.getByText("washer: expected 2, found 1")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /annotated image of part.png/i })).toHaveAttribute("src", "/inspections/x/annotated");
    expect(screen.getByRole("table")).toHaveTextContent("washer");
  });

  it("switches between annotated and original image", async () => {
    mockFetch({ "/products": () => json(PRODUCTS), "/inspect": () => json(response("PASS")) });
    render(<App />);
    await pickAndSubmit();
    await screen.findByRole("status");
    fireEvent.click(screen.getByRole("button", { name: "Original" }));
    expect(screen.getByRole("img")).toHaveAttribute("src", "/inspections/x/original");
  });

  it("shows the API's error message and code, and no stale result", async () => {
    mockFetch({
      "/products": () => json(PRODUCTS),
      "/inspect": () => json({ error: { code: "invalid_image", message: "file could not be decoded as an image" } }, 422),
    });
    render(<App />);
    await pickAndSubmit();
    expect(await screen.findByRole("alert")).toHaveTextContent("file could not be decoded as an image (invalid_image)");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("explains a network failure instead of hanging", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) =>
      url === "/products" ? Promise.resolve(json(PRODUCTS)) : Promise.reject(new TypeError("Failed to fetch"))));
    render(<App />);
    await pickAndSubmit();
    expect(await screen.findByRole("alert")).toHaveTextContent(/cannot reach the inspection service/i);
  });

  it("rejects a non-image file in the browser before any request is sent", async () => {
    mockFetch({ "/products": () => json(PRODUCTS) });
    render(<App />);
    await pickAndSubmit("notes.txt");
    expect(await screen.findByRole("alert")).toHaveTextContent(/PNG, JPG or BMP/);
    expect(screen.getByRole("button", { name: "Inspect" })).toBeDisabled();
  });

  it("warns when the selected product has no weights installed", async () => {
    mockFetch({ "/products": () => json(PRODUCTS) });
    render(<App />);
    await screen.findByRole("option", { name: "screw_bag" });
    fireEvent.change(screen.getByLabelText(/product/i), { target: { value: "splicing_connectors" } });
    await waitFor(() => expect(screen.getByText(/weights for this product are not installed/i)).toBeInTheDocument());
  });
});