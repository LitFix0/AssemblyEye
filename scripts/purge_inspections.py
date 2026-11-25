"""Delete inspections older than N days: the database rows and their image folders.

  python scripts/purge_inspections.py --days 90 --dry-run     # only list what would go
  python scripts/purge_inspections.py --days 90               # delete

Retention is a policy decision, not an automatic one: nothing in the app deletes records on its own."""
import argparse
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.settings import load_settings  # noqa: E402
from app.db.store import InspectionStore  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, required=True, help="delete inspections older than this many days")
    ap.add_argument("--dry-run", action="store_true", help="list what would be deleted and change nothing")
    ap.add_argument("--db", help="SQLite file (default: from settings)")
    ap.add_argument("--data-dir", help="image folder root (default: from settings)")
    args = ap.parse_args(argv)
    if args.days < 0:
        print("ERROR: --days must be 0 or more", file=sys.stderr)
        return 2
    settings = load_settings(data_dir=args.data_dir, db_path=args.db)
    store = InspectionStore(settings.db_path)
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)
    ids = store.ids_older_than(cutoff)
    print(f"{len(ids)} inspection(s) older than {args.days} day(s) (before {cutoff:%Y-%m-%d %H:%M} UTC)")
    if args.dry_run or not ids:
        if args.dry_run and ids:
            print("dry run: nothing deleted")
        return 0
    removed = 0
    for iid in ids:
        store.delete(iid)
        folder = settings.data_dir / iid
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
        removed += 1
    print(f"deleted {removed} record(s) and their image folders")
    return 0


if __name__ == "__main__":
    sys.exit(main())