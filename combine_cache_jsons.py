"""
Combine multiple neo4j cache JSON files into one.

Reads cache/<Author>_data.json files and merges them into a single
JSON payload that neo4j_import.py can ingest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def combine_cache_files(cache_dir: Path, paths: List[Path]) -> Dict[str, object]:
    author_data: Dict[str, object] = {}
    works_data: Dict[str, object] = {}
    author_names: List[str] = []
    source_files: List[str] = []

    for path in paths:
        payload = load_json(path)
        author_name = payload.get("author_name")
        if author_name:
            author_names.append(author_name)
        source_files.append(str(path.relative_to(cache_dir)))

        for author_id, author in (payload.get("author_data") or {}).items():
            if author_id in author_data and author_data[author_id] != author:
                print(f"Warning: author_id collision with differing data: {author_id}")
                continue
            author_data.setdefault(author_id, author)

        for work_id, work in (payload.get("works_data") or {}).items():
            if work_id in works_data and works_data[work_id] != work:
                print(f"Warning: work_id collision with differing data: {work_id}")
                continue
            works_data.setdefault(work_id, work)

    author_id_to_label = {author_id: str(i) for i, author_id in enumerate(author_data.keys())}

    return {
        "author_name": "combined",
        "author_names": author_names,
        "source_files": source_files,
        "author_data": author_data,
        "works_data": works_data,
        "author_id_to_label": author_id_to_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine neo4j cache JSON files")
    parser.add_argument(
        "--cache-dir",
        default="neo4j_and/cache",
        help="Directory containing *_data.json files",
    )
    parser.add_argument(
        "--output",
        default="neo4j_and/cache/combined_data.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--pattern",
        default="*_data.json",
        help="Glob pattern for cache files",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        raise SystemExit(f"Cache dir not found: {cache_dir}")

    paths = sorted(cache_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No cache files found in {cache_dir} (pattern {args.pattern})")

    payload = combine_cache_files(cache_dir, paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {output_path} from {len(paths)} cache files")


if __name__ == "__main__":
    main()
