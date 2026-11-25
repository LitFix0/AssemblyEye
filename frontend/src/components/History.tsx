import { useCallback, useEffect, useState } from "react";
import { ApiRequestError, deleteInspection, getInspection, listInspections } from "../api";
import { dayStartIso, nextDayStartIso } from "../format";
import type { InspectionList, InspectionResponse, ProductInfo, Status } from "../types";
import ResultView from "./ResultView";

const PAGE = 20;

interface Draft {
  product: string;
  status: Status | "";
  code: string;
  from: string; // "YYYY-MM-DD" or ""
  to: string;
}
const EMPTY: Draft = { product: "", status: "", code: "", from: "", to: "" };

const msg = (e: unknown) => (e instanceof ApiRequestError ? `${e.message} (${e.code})` : "Unexpected error.");

export default function History({ products }: { products: ProductInfo[] }) {
  const [draft, setDraft] = useState<Draft>(EMPTY);
  const [applied, setApplied] = useState<Draft>(EMPTY);
  const [offset, setOffset] = useState(0);
  const [list, setList] = useState<InspectionList | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<InspectionResponse | null>(null);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    listInspections({
      product: applied.product,
      status: applied.status,
      code: applied.code.trim(),
      since: applied.from ? dayStartIso(applied.from) : "",
      until: applied.to ? nextDayStartIso(applied.to) : "",
      limit: PAGE,
      offset,
    })
      .then((l) => live && setList(l))
      .catch((e) => live && setError(msg(e)))
      .finally(() => live && setLoading(false));
    return () => {
      live = false;
    };
  }, [applied, offset, reload]);

  function apply(e: React.FormEvent) {
    e.preventDefault();
    setOffset(0);
    setApplied(draft);
  }

  function clear() {
    setDraft(EMPTY);
    setApplied(EMPTY);
    setOffset(0);
  }

  const open = useCallback(async (id: string) => {
    setError(null);
    try {
      setDetail(await getInspection(id));
    } catch (e) {
      setError(msg(e));
    }
  }, []);

  async function remove(id: string) {
    if (!window.confirm("Delete this inspection and its images? This cannot be undone.")) return;
    try {
      await deleteInspection(id);
      if (detail?.inspection_id === id) setDetail(null);
      if (list && list.items.length === 1 && offset > 0) setOffset(offset - PAGE); // do not strand an empty page
      else setReload((n) => n + 1);
    } catch (e) {
      setError(msg(e));
    }
  }

  const first = list && list.total > 0 ? offset + 1 : 0;
  const last = list ? offset + list.items.length : 0;

  return (
    <section aria-label="Inspection history">
      <form onSubmit={apply} className="panel">
        <label>
          Product
          <select value={draft.product} onChange={(e) => setDraft({ ...draft, product: e.target.value })}>
            <option value="">All</option>
            {products.map((p) => (
              <option key={p.product} value={p.product}>{p.product}</option>
            ))}
          </select>
        </label>
        <label>
          Status
          <select value={draft.status} onChange={(e) => setDraft({ ...draft, status: e.target.value as Status | "" })}>
            <option value="">All</option>
            <option value="PASS">PASS</option>
            <option value="REVIEW">REVIEW</option>
            <option value="FAIL">FAIL</option>
          </select>
        </label>
        <label>
          Finding code
          <input type="text" placeholder="e.g. MISSING" value={draft.code} onChange={(e) => setDraft({ ...draft, code: e.target.value })} />
        </label>
        <label>
          From
          <input type="date" value={draft.from} onChange={(e) => setDraft({ ...draft, from: e.target.value })} />
        </label>
        <label>
          To
          <input type="date" value={draft.to} onChange={(e) => setDraft({ ...draft, to: e.target.value })} />
        </label>
        <button type="submit">Apply</button>
        <button type="button" className="secondary" onClick={clear}>Clear</button>
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {loading && !list && <p className="muted">Loading…</p>}

      {list && (
        <>
          {list.items.length === 0 ? (
            <p className="muted">No inspections match.</p>
          ) : (
            <table className="history">
              <thead>
                <tr>
                  <th>Time</th><th>Product</th><th>Status</th><th>Image</th><th>Findings</th><th>ms</th><th></th>
                </tr>
              </thead>
              <tbody>
                {list.items.map((r) => (
                  <tr key={r.inspection_id}>
                    <td>{new Date(r.created_at).toLocaleString()}</td>
                    <td>{r.product}</td>
                    <td><span className={`chip chip-${r.status.toLowerCase()}`}>{r.status}</span></td>
                    <td>{r.image_name}</td>
                    <td>{r.finding_codes.length ? r.finding_codes.join(", ") : "—"}</td>
                    <td>{Math.round(r.total_ms)}</td>
                    <td className="actions">
                      <button type="button" onClick={() => open(r.inspection_id)} aria-label={`View ${r.image_name}`}>View</button>
                      <button type="button" className="secondary" onClick={() => remove(r.inspection_id)} aria-label={`Delete ${r.image_name}`}>Delete</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <div className="pager">
            <span className="muted">{first}–{last} of {list.total}</span>
            <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
            <button type="button" disabled={last >= list.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
          </div>
        </>
      )}

      {detail && (
        <div className="detail">
          <div className="detail-head">
            <h2>Inspection {detail.inspection_id.slice(0, 8)}</h2>
            <button type="button" className="secondary" onClick={() => setDetail(null)}>Close</button>
          </div>
          <ResultView response={detail} />
        </div>
      )}
    </section>
  );
}