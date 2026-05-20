#!/usr/bin/env python3
"""
Export historicon_filter.py to the JSON format expected by OpenWebUI.

Usage:
    python openwebui/export_filter.py

Reads:  openwebui/historicon_filter.py
Writes: openwebui/historicon_filter.json  (overwrites in place)

The version number and function ID are extracted from the filter source itself
so this script never needs to be updated manually.
"""

import json
import re
import sys
import time
from pathlib import Path

FILTER_PY = Path(__file__).parent / "historicon_filter.py"
FILTER_JSON = Path(__file__).parent / "historicon_filter.json"


def extract_frontmatter(source: str) -> dict:
    """Parse simple key: value pairs from the module docstring front-matter."""
    fm = {}
    for key in ("title", "author", "version", "description"):
        m = re.search(rf"^{key}:\s*(.+)", source, re.MULTILINE)
        if m:
            fm[key] = m.group(1).strip()
    return fm


def main() -> None:
    src = FILTER_PY.read_text(encoding="utf-8")
    fm = extract_frontmatter(src)
    if not fm.get("version"):
        raise ValueError("Could not find 'version:' in historicon_filter.py docstring")

    # id = filename stem (matches what OpenWebUI uses when you create/export)
    function_id = FILTER_PY.stem  # "historicon_filter"
    now = int(time.time())

    export = [
        {
            "id": function_id,
            "name": function_id,
            "type": "filter",
            "content": src,
            "meta": {
                "description": function_id,
                "manifest": fm,
            },
            "is_active": True,
            "is_global": True,
            "updated_at": now,
            "created_at": now,
        }
    ]

    FILTER_JSON.write_text(
        json.dumps(export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Exported {FILTER_PY.name} v{fm['version']} → {FILTER_JSON.name} ({len(src):,} chars)"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
