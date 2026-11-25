import { useEffect, useState } from "react";
import { ApiRequestError, fetchProducts, inspectImage } from "./api";
import History from "./components/History";
import ResultView from "./components/ResultView";
import { fileProblem } from "./format";
import type { InspectionResponse, ProductInfo } from "./types";

export default function App() {
  const [products, setProducts] = useState<ProductInfo[]>([]);
  const [product, setProduct] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<InspectionResponse | null>(null);
  const [tab, setTab] = useState<"inspect" | "history">("inspect");

  useEffect(() => {
    fetchProducts()
      .then((p) => {
        setProducts(p);
        if (p.length > 0) setProduct(p[0].product);
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  const chosen = products.find((p) => p.product === product);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!file || !product) return;
    setBusy(true);
    setError(null);
    setResponse(null);
    try {
      setResponse(await inspectImage(product, file));
    } catch (err) {
      setError(err instanceof ApiRequestError ? `${err.message} (${err.code})` : "Unexpected error.");
    } finally {
      setBusy(false);
    }
  }

  function onFile(f: File | null) {
    setFile(null);
    setError(null);
    if (!f) return;
    const problem = fileProblem(f);
    if (problem) setError(problem);
    else setFile(f);
  }

  return (
    <main>
      <h1>Assembly verification</h1>
      <p className="muted">Research prototype on public data. Not validated for production use.</p>

      <nav className="tabs" aria-label="Sections">
        <button type="button" className={tab === "inspect" ? "on" : ""} aria-pressed={tab === "inspect"} onClick={() => setTab("inspect")}>New inspection</button>
        <button type="button" className={tab === "history" ? "on" : ""} aria-pressed={tab === "history"} onClick={() => setTab("history")}>History</button>
      </nav>

      {tab === "history" ? (
        <History products={products} />
      ) : (
        <>
      <form onSubmit={submit} className="panel">
        <label>
          Product
          <select value={product} onChange={(e) => setProduct(e.target.value)} disabled={busy || products.length === 0}>
            {products.map((p) => (
              <option key={p.product} value={p.product}>{p.product}</option>
            ))}
          </select>
        </label>
        {chosen && !chosen.weights_present && (
          <p className="warn">The model weights for this product are not installed on the server.</p>
        )}
        <label>
          Image
          <input type="file" accept=".png,.jpg,.jpeg,.bmp" onChange={(e) => onFile(e.target.files?.[0] ?? null)} disabled={busy} />
        </label>
        <button type="submit" disabled={!file || !product || busy}>{busy ? "Inspecting…" : "Inspect"}</button>
        {busy && <p className="muted" role="progressbar" aria-label="Inspecting">The first request loads the model and can take 10–20 seconds.</p>}
      </form>

      {error && <p className="error" role="alert">{error}</p>}
      {response && <ResultView response={response} />}
        </>
      )}
    </main>
  );
}