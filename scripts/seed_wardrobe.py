"""Bulk-import wardrobe items from a JSON file.

Usage:
    python scripts/seed_wardrobe.py scripts/wardrobe_seed.example.json
    python scripts/seed_wardrobe.py data/my_wardrobe.json --dry-run

Requires Ollama running with EMBED_MODEL (config.py) already pulled --
every item gets embedded on insert, so this makes one /api/embed call per
row.

Not idempotent: re-running against the same file inserts duplicates.
There's no dedupe-on-name, since "two black tees" is legitimate wardrobe
data. If you need a clean re-seed, delete data/midas.db first, or use
--dry-run to check the file before committing to a real run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Running this file directly (not as `python -m scripts.seed_wardrobe`)
# puts only scripts/ on sys.path, not the project root -- add it so
# `config`, `core`, and `tools` are importable either way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from core.db import init_db  # noqa: E402
from tools.wardrobe import add_wardrobe_item  # noqa: E402


def load_items(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("items", [])
    return data


def validate(item: dict) -> str | None:
    """Returns an error string, or None if the item is well-formed."""
    if not item.get("name", "").strip():
        return "missing or empty 'name'"
    category = item.get("category", "")
    if category not in config.WARDROBE_CATEGORIES:
        return f"unknown category {category!r} (expected one of {config.WARDROBE_CATEGORIES})"
    for field in ("warmth_level", "formality"):
        value = item.get(field)
        if value is not None and not (isinstance(value, int) and 1 <= value <= 5):
            return f"'{field}' must be an integer 1-5 or omitted, got {value!r}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-import wardrobe items from a JSON file.")
    parser.add_argument("json_path", type=Path)
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and report without inserting."
    )
    args = parser.parse_args()

    items = load_items(args.json_path)
    print(f"Loaded {len(items)} item(s) from {args.json_path}\n")

    init_db()

    imported = skipped = failed = 0
    for item in items:
        error = validate(item)
        if error:
            print(f"  SKIP {item.get('name', '<unnamed>')!r}: {error}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  OK (dry run) {item['name']!r} ({item['category']})")
            imported += 1
            continue

        try:
            result = add_wardrobe_item(
                category=item["category"],
                name=item["name"],
                color=item.get("color"),
                warmth_level=item.get("warmth_level"),
                formality=item.get("formality"),
                tags=item.get("tags"),
            )
            print(f"  OK #{result['id']}: {result['name']} ({result['category']})")
            imported += 1
        except Exception as exc:
            print(f"  FAIL {item.get('name', '<unnamed>')!r}: {exc}")
            failed += 1

    total = len(items)
    print(f"\nImported {imported}/{total} items ({skipped} skipped, {failed} failed).")


if __name__ == "__main__":
    main()
