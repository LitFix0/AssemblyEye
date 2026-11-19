import { componentState } from "../format";
import type { ComponentCount } from "../types";

const LABEL = { ok: "as expected", check: "needs a look", wrong: "count differs" } as const;

export default function ComponentTable({ components }: { components: ComponentCount[] }) {
  return (
    <table className="components">
      <caption>Expected vs detected (confidence bands from the spec)</caption>
      <thead>
        <tr>
          <th>Component</th>
          <th>Expected</th>
          <th>Confident</th>
          <th>Low confidence</th>
          <th>State</th>
        </tr>
      </thead>
      <tbody>
        {components.map((c) => {
          const state = componentState(c);
          return (
            <tr key={c.class_name} className={`row-${state}`}>
              <td>{c.class_name}</td>
              <td>{c.expected}</td>
              <td>{c.confident}</td>
              <td>{c.uncertain}</td>
              <td>{LABEL[state]}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}