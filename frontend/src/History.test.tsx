import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import App from "./App";
import type { InspectionList, InspectionResponse, InspectionSummary, ProductInfo } from "./types";

const PRODUCTS: ProductInfo[] = [
  { product: "screw_bag", spec_version: "0.2", description: "", weights_present: true, detector_loaded: false },
];

const row = (n: number, over: Partial<InspectionSummary> = {}): InspectionSummary => ({
  inspection_id: n.toString(16).padStart(32, "0"), created_at: "2026-01-10T12:00:00Z", product: "screw_bag",
  spec_version: "0.2", status: "PASS", image_name: `img${n}.png`, detector: "fake", finding_codes: [], total_ms: 120.4, ...over,
});

const detail = (r: InspectionSummary): InspectionResponse => ({
  inspection_id: r.inspection_id, created_at: r.created_at,
  result: { product: r.product, spec_version: "0.2", status: r.status, image: { name: r.image_name, width: 4, height: 3 },
            detections: [], findings: [], detector: "fake", components: [] },
  evidence: { original_url: `/inspections/${r.inspection_id}/original`, annotated_url: `/inspections/${r.inspection_id}/annotated` },
  timings_ms: { total: 5 },
});

const list = (items: InspectionSummary[], total = items.length, offset = 0): InspectionList => ({ items, total, limit: 20, offset });
const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status });

type Handler = (url: URL, init?: RequestInit) => Response | Promise<Response>;
let calls: { url: URL; method: string }[];

function mockApi(onList: Handler, extra: Record<string, Handler> = {}) {
  calls = [];
  vi.stubGlobal("fetch", vi.fn((input: string, init?: RequestInit) => {
    const url = new URL(input, "http://x");
    calls.push({ url, method: init?.method ?? "GET" });
    if (url.pathname === "/products") return Promise.resolve(json(PRODUCTS));
    if (url.pathname === "/inspections" && (init?.method ?? "GET") === "GET") return Promise.resolve(onList(url, init));
    const h = extra[`${init?.method ?? "GET"} ${url.pathname}`];
    if (!h) throw new Error(`unexpected fetch ${init?.method ?? "GET"} ${input}`);
    return Promise.resolve(h(url, init));
  }));
}

async function openHistory() {
  render(<App />);
  await screen.findByRole("option", { name: "screw_bag" });
  fireEvent.click(screen.getByRole("button", { name: "History" }));
}
const listCalls = () => calls.filter((c) => c.url.pathname === "/inspections" && c.method === "GET");

afterEach(() => vi.unstubAllGlobals());

describe("History", () => {
  it("lists past inspections with status, image and finding codes", async () => {
    mockApi(() => json(list([row(1, { status: "FAIL", finding_codes: ["MISSING", "EXTRA"] }), row(2)])));
    await openHistory();
    const table = await screen.findByRole("table");
    expect(within(table).getByText("img1.png")).toBeInTheDocument();
    expect(within(table).getByText("MISSING, EXTRA")).toBeInTheDocument();
    expect(within(table).getByText("FAIL")).toBeInTheDocument();
    expect(screen.getByText("1–2 of 2")).toBeInTheDocument();
  });

  it("shows an empty state", async () => {
    mockApi(() => json(list([])));
    await openHistory();
    expect(await screen.findByText(/no inspections match/i)).toBeInTheDocument();
    expect(screen.getByText("0–0 of 0")).toBeInTheDocument();
  });

  it("sends the chosen filters to the API, with whole-day date bounds", async () => {
    mockApi(() => json(list([])));
    await openHistory();
    await screen.findByText(/no inspections match/i);
    fireEvent.change(screen.getByLabelText("Product"), { target: { value: "screw_bag" } });
    fireEvent.change(screen.getByLabelText("Status"), { target: { value: "FAIL" } });
    fireEvent.change(screen.getByLabelText("Finding code"), { target: { value: " MISSING " } });
    fireEvent.change(screen.getByLabelText("From"), { target: { value: "2026-01-10" } });
    fireEvent.change(screen.getByLabelText("To"), { target: { value: "2026-01-10" } });
    fireEvent.click(screen.getByRole("button", { name: "Apply" }));
    await waitFor(() => expect(listCalls().length).toBe(2));
    const q = listCalls()[1].url.searchParams;
    expect(q.get("product")).toBe("screw_bag");
    expect(q.get("status")).toBe("FAIL");
    expect(q.get("code")).toBe("MISSING");
    expect(q.get("offset")).toBe("0");
    const since = new Date(q.get("since")!), until = new Date(q.get("until")!);
    expect(since.getHours()).toBe(0);                                   // local midnight
    expect(until.getTime() - since.getTime()).toBeGreaterThanOrEqual(23 * 3600e3);  // one whole day (DST can make it 23 h)
    expect(until.getTime() - since.getTime()).toBeLessThanOrEqual(25 * 3600e3);
  });

  it("does not send empty filters", async () => {
    mockApi(() => json(list([])));
    await openHistory();
    await screen.findByText(/no inspections match/i);
    const q = listCalls()[0].url.searchParams;
    expect([...q.keys()].sort()).toEqual(["limit", "offset"]);
  });

  it("pages forward and back", async () => {
    mockApi((url) => {
      const off = Number(url.searchParams.get("offset"));
      return json(list(off === 0 ? [row(1)] : [row(2)], 21, off));
    });
    await openHistory();
    await screen.findByText("img1.png");
    expect(screen.getByRole("button", { name: "Previous" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Next" }));
    await screen.findByText("img2.png");
    expect(listCalls()[1].url.searchParams.get("offset")).toBe("20");
    expect(screen.getByRole("button", { name: "Next" })).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Previous" }));
    await screen.findByText("img1.png");
  });

  it("opens a stored inspection and can close it", async () => {
    const r = row(3, { status: "REVIEW" });
    mockApi(() => json(list([r])), { [`GET /inspections/${r.inspection_id}`]: () => json(detail(r)) });
    await openHistory();
    fireEvent.click(await screen.findByRole("button", { name: "View img3.png" }));
    expect(await screen.findByRole("status")).toHaveTextContent("REVIEW");
    expect(screen.getByRole("img")).toHaveAttribute("src", `/inspections/${r.inspection_id}/annotated`);
    fireEvent.click(screen.getByRole("button", { name: "Close" }));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("deletes only after confirmation and then refreshes the list", async () => {
    const r = row(4);
    let deleted = false;
    mockApi(() => json(list(deleted ? [] : [r])), {
      [`DELETE /inspections/${r.inspection_id}`]: () => { deleted = true; return new Response(null, { status: 204 }); },
    });
    await openHistory();
    const del = await screen.findByRole("button", { name: "Delete img4.png" });

    vi.stubGlobal("confirm", vi.fn(() => false));
    fireEvent.click(del);
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);          // declined: nothing sent

    vi.stubGlobal("confirm", vi.fn(() => true));
    fireEvent.click(del);
    expect(await screen.findByText(/no inspections match/i)).toBeInTheDocument();
    expect(calls.filter((c) => c.method === "DELETE")).toHaveLength(1);
  });

  it("shows the API's error and keeps the page usable when the list fails", async () => {
    mockApi(() => json({ error: { code: "invalid_request", message: "limit: too big" } }, 422));
    await openHistory();
    expect(await screen.findByRole("alert")).toHaveTextContent("limit: too big (invalid_request)");
    expect(screen.getByRole("button", { name: "Apply" })).toBeInTheDocument();
  });

  it("shows an error if a delete fails and keeps the row", async () => {
    const r = row(5);
    mockApi(() => json(list([r])), {
      [`DELETE /inspections/${r.inspection_id}`]: () => json({ error: { code: "not_found", message: "no such inspection" } }, 404),
    });
    vi.stubGlobal("confirm", vi.fn(() => true));
    await openHistory();
    fireEvent.click(await screen.findByRole("button", { name: "Delete img5.png" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("no such inspection (not_found)");
    expect(screen.getByText("img5.png")).toBeInTheDocument();
  });
});