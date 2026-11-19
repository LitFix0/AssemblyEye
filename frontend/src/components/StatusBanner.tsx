import { STATUS_TEXT } from "../format";
import type { Status } from "../types";

const ICON: Record<Status, string> = { PASS: "✔", REVIEW: "?", FAIL: "✖" };

export default function StatusBanner({ status }: { status: Status }) {
  return (
    <div className={`banner banner-${status.toLowerCase()}`} role="status">
      <span className="banner-icon" aria-hidden="true">{ICON[status]}</span>
      <span className="banner-title">{STATUS_TEXT[status].title}</span>
      <span className="banner-hint">{STATUS_TEXT[status].hint}</span>
    </div>
  );
}