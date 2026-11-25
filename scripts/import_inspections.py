"""One-off: load inspections saved by the ST-05 API (a result.json inside each image folder) into the
SQLite history. Safe to run more than once: ids already in the database are skipped.

  python scripts/import_inspections.py"""
import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.core.settings import load_settings  # noqa: E402
from app.db.store import InspectionStore  # noqa: E402
from app.schemas.api import InspectionResponse  # noqa: E402
from app.services.inspection_service import ID_RE  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="SQLite file (default: from settings)")
    ap.add_argument("--data-dir", help="image folder root (default: from settings)")
    args = ap.parse_args(argv)
    settings = load_settings(data_dir=args.data_dir, db_path=args.db)
    store = InspectionStore(settings.db_path)
    added = skipped = bad = 0
    folders = sorted(p for p in settings.data_dir.iterdir() if p.is_dir() and ID_RE.match(p.name)) \
        if settings.data_dir.is_dir() else []
    for folder in folders:
        file = folder / "result.json"
        if store.exists(folder.name):
            skipped += 1
            continue
        if not file.is_file():
            continue  # a folder written by the new API already has its row; one without a file is not importable
        try:
            response = InspectionResponse.model_validate_json(file.read_text(encoding="utf-8"))
            if response.inspection_id != folder.name:
                raise ValueError("id in result.json does not match the folder name")
            store.add(response)
            added += 1
        except (ValidationError, ValueError, OSError) as e:
            bad += 1
            print(f"skipped {folder.name}: {e}", file=sys.stderr)
    print(f"imported {added}, already present {skipped}, unreadable {bad} (of {len(folders)} folder(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())