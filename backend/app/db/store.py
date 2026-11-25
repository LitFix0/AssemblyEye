"""SQLite inspection history.

What goes where
  - this database : one row per inspection (queryable columns + the full result as JSON) and one row per
                    finding, so history can be filtered by product, status, finding code and date.
  - evidence files: the original and annotated images stay on disk in <data_dir>/<inspection_id>/.
                    Images are never stored in the database (no duplicated bytes, small DB).
The database is the single source of truth for results; there is no second copy of the JSON on disk.

Timestamps are stored as UTC text 'YYYY-MM-DDTHH:MM:SS.ffffffZ', which sorts and compares correctly as text.
A connection is opened per operation: SQLite connections are cheap, and the API serves requests from a
thread pool, so no connection is ever shared between threads.
"""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.schemas.api import InspectionList, InspectionResponse, InspectionSummary

TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

SCHEMA = """
CREATE TABLE IF NOT EXISTS inspections (
    id           TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    product      TEXT NOT NULL,
    spec_version TEXT NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('PASS', 'FAIL', 'REVIEW')),
    image_name   TEXT NOT NULL,
    detector     TEXT NOT NULL,
    total_ms     REAL NOT NULL,
    result_json  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    inspection_id TEXT NOT NULL REFERENCES inspections(id) ON DELETE CASCADE,
    code          TEXT NOT NULL,
    status        TEXT NOT NULL,
    class_name    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at);
CREATE INDEX IF NOT EXISTS idx_inspections_product_status ON inspections(product, status);
CREATE INDEX IF NOT EXISTS idx_findings_code ON findings(code);
CREATE INDEX IF NOT EXISTS idx_findings_inspection ON findings(inspection_id);
"""


def to_ts(dt: datetime) -> str:
    """Naive datetimes are taken as UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(TS_FORMAT)


def from_ts(text: str) -> datetime:
    return datetime.strptime(text, TS_FORMAT).replace(tzinfo=timezone.utc)


class InspectionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
            conn.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def add(self, r: InspectionResponse) -> None:
        res = r.result
        with closing(self._connect()) as conn, conn:  # inner `conn` = one transaction
            conn.execute(
                "INSERT INTO inspections (id, created_at, product, spec_version, status, image_name, detector,"
                " total_ms, result_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r.inspection_id, to_ts(r.created_at), res.product, res.spec_version, res.status.value,
                 res.image.name, res.detector, r.timings_ms.get("total", 0.0), r.model_dump_json()))
            conn.executemany(
                "INSERT INTO findings (inspection_id, code, status, class_name) VALUES (?, ?, ?, ?)",
                [(r.inspection_id, f.code, f.status.value, f.class_name) for f in res.findings])

    def get(self, inspection_id: str) -> InspectionResponse | None:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT result_json FROM inspections WHERE id = ?", (inspection_id,)).fetchone()
        return InspectionResponse.model_validate_json(row["result_json"]) if row else None

    def exists(self, inspection_id: str) -> bool:
        with closing(self._connect()) as conn:
            return conn.execute("SELECT 1 FROM inspections WHERE id = ?", (inspection_id,)).fetchone() is not None

    def search(self, *, product: str | None = None, status: str | None = None, code: str | None = None,
               since: datetime | None = None, until: datetime | None = None,
               limit: int = 50, offset: int = 0) -> InspectionList:
        """Newest first. `since` is inclusive, `until` is exclusive."""
        where: list[str] = []
        args: list = []
        if product:
            where.append("i.product = ?")
            args.append(product)
        if status:
            where.append("i.status = ?")
            args.append(status)
        if code:
            where.append("EXISTS (SELECT 1 FROM findings f WHERE f.inspection_id = i.id AND f.code = ?)")
            args.append(code)
        if since:
            where.append("i.created_at >= ?")
            args.append(to_ts(since))
        if until:
            where.append("i.created_at < ?")
            args.append(to_ts(until))
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        with closing(self._connect()) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM inspections i {clause}", args).fetchone()[0]
            rows = conn.execute(
                f"SELECT i.id, i.created_at, i.product, i.spec_version, i.status, i.image_name, i.detector,"
                f" i.total_ms, (SELECT group_concat(DISTINCT f.code) FROM findings f WHERE f.inspection_id = i.id)"
                f" AS codes FROM inspections i {clause} ORDER BY i.created_at DESC, i.id DESC LIMIT ? OFFSET ?",
                [*args, limit, offset]).fetchall()
        items = [InspectionSummary(
            inspection_id=r["id"], created_at=from_ts(r["created_at"]), product=r["product"],
            spec_version=r["spec_version"], status=r["status"], image_name=r["image_name"],
            detector=r["detector"], finding_codes=sorted(r["codes"].split(",")) if r["codes"] else [],
            total_ms=r["total_ms"]) for r in rows]
        return InspectionList(items=items, total=total, limit=limit, offset=offset)

    def delete(self, inspection_id: str) -> bool:
        with closing(self._connect()) as conn, conn:
            return conn.execute("DELETE FROM inspections WHERE id = ?", (inspection_id,)).rowcount > 0

    def ids_older_than(self, cutoff: datetime) -> list[str]:
        with closing(self._connect()) as conn:
            rows = conn.execute("SELECT id FROM inspections WHERE created_at < ? ORDER BY created_at",
                                (to_ts(cutoff),)).fetchall()
        return [r["id"] for r in rows]