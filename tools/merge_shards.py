from __future__ import annotations

import argparse
from pathlib import Path

from a5sv2_eval.common import read_jsonl, row_key, write_jsonl


def merge(manifest_path: Path, shard_paths: list[Path], output: Path) -> None:
    manifest = read_jsonl(manifest_path)
    by_key: dict[tuple[str, str, str], dict] = {}
    systems: set[str] = set()
    for path in shard_paths:
        for row in read_jsonl(path):
            key = row_key(row)
            if key in by_key:
                raise RuntimeError(f"Duplicate row across shards: {key}")
            if row.get("status") != "ok":
                raise RuntimeError(f"Non-success row in {path}: {row.get('id')}")
            by_key[key] = row
            systems.add(row["system_id"])
    if len(systems) != 1:
        raise RuntimeError(f"Expected one system across shards, found: {sorted(systems)}")
    missing = [row["id"] for row in manifest if row_key(row) not in by_key]
    extra = set(by_key) - {row_key(row) for row in manifest}
    if missing or extra:
        raise RuntimeError(f"Shard coverage mismatch: {len(missing)} missing, {len(extra)} extra")
    write_jsonl(output, [by_key[row_key(row)] for row in manifest])
    print(f"Saved {output} ({len(manifest)} rows, system={systems.pop()})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic benchmark shards")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("shards", type=Path, nargs="+")
    args = parser.parse_args()
    merge(args.manifest, args.shards, args.output)


if __name__ == "__main__":
    main()
