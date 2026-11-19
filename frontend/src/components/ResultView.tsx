import { useState } from "react";
import type { InspectionResponse } from "../types";
import ComponentTable from "./ComponentTable";
import FindingsList from "./FindingsList";
import StatusBanner from "./StatusBanner";

export default function ResultView({ response }: { response: InspectionResponse }) {
  const { result, evidence, timings_ms } = response;
  const [view, setView] = useState<"annotated" | "original">("annotated");
  const src = view === "annotated" ? evidence.annotated_url : evidence.original_url;
  return (
    <section className="result" aria-label="Inspection result">
      <StatusBanner status={result.status} />
      <div className="result-grid">
        <div>
          <div className="toggle" role="group" aria-label="Image view">
            <button className={view === "annotated" ? "on" : ""} onClick={() => setView("annotated")}>Annotated</button>
            <button className={view === "original" ? "on" : ""} onClick={() => setView("original")}>Original</button>
          </div>
          <img className="evidence" src={src} alt={`${view} image of ${result.image.name}`} />
        </div>
        <div>
          <h2>Findings</h2>
          <FindingsList findings={result.findings} />
          <h2>Components</h2>
          <ComponentTable components={result.components} />
          <p className="meta muted">
            {result.product} spec v{result.spec_version} · {result.image.width}×{result.image.height} · inspection{" "}
            {response.inspection_id.slice(0, 8)} · {Math.round(timings_ms.total)} ms
            <br />
            detector: {result.detector}
          </p>
        </div>
      </div>
    </section>
  );
}