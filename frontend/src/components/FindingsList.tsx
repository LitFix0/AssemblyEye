import { sortFindings } from "../format";
import type { Finding } from "../types";

export default function FindingsList({ findings }: { findings: Finding[] }) {
  if (findings.length === 0) {
    return <p className="muted">No findings: every check passed.</p>;
  }
  return (
    <ul className="findings">
      {sortFindings(findings).map((f, i) => (
        <li key={`${f.code}-${i}`} className={`finding finding-${f.status.toLowerCase()}`}>
          <div>
            <span className="tag">{f.status}</span> <strong>{f.code}</strong> <span className="muted">{f.class_name}</span>
          </div>
          <p>{f.message}</p>
          {Object.keys(f.evidence).length > 0 && (
            <details>
              <summary>Evidence</summary>
              <pre>{JSON.stringify(f.evidence, null, 2)}</pre>
            </details>
          )}
        </li>
      ))}
    </ul>
  );
}