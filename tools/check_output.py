from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def source_to_dataset_id(group: str, source: Any) -> str:
    if group == "webvid":
        return str(source)
    return Path(str(source)).name.split(".")[0]


def flatten_rows(payload: Any) -> dict[str, list[dict[str, Any]]]:
    rows = {"webvid": [], "ss2": []}
    if not isinstance(payload, list):
        raise ValueError("top-level JSON must be a list")
    for section in payload:
        if not isinstance(section, dict):
            raise ValueError("each top-level section must be an object")
        for group in ("webvid", "ss2"):
            values = section.get(group)
            if values is not None:
                if not isinstance(values, list):
                    raise ValueError(f"{group} section must be a list")
                rows[group].extend(values)
    return rows


def expected_sources(test_json: Path | None) -> dict[tuple[str, int], str]:
    if test_json is None:
        return {}
    expected = {}
    for group, rows in flatten_rows(read_json(test_json)).items():
        for row in rows:
            expected[(group, int(row["id"]))] = source_to_dataset_id(group, row["video_source"])
    return expected


def validate(output_json: Path, test_json: Path | None, expected_top_k: int) -> dict[str, int]:
    rows = flatten_rows(read_json(output_json))
    sources = expected_sources(test_json)
    seen_keys: set[tuple[str, int]] = set()
    counts = {}
    for group, group_rows in rows.items():
        counts[group] = len(group_rows)
        for row in group_rows:
            item_id = int(row["id"])
            key = (group, item_id)
            if key in seen_keys:
                raise ValueError(f"duplicate row: {group}:{item_id}")
            seen_keys.add(key)
            targets = row.get("video_target")
            if not isinstance(targets, list):
                raise ValueError(f"{group}:{item_id} video_target must be a list")
            if len(targets) != expected_top_k:
                raise ValueError(f"{group}:{item_id} has {len(targets)} predictions, expected {expected_top_k}")
            target_strings = [str(value) for value in targets]
            if len(set(target_strings)) != len(target_strings):
                raise ValueError(f"{group}:{item_id} has duplicate predictions")
            if key in sources and sources[key] in target_strings:
                raise ValueError(f"{group}:{item_id} includes source video")
            trace = row.get("reasoning_trace")
            if not isinstance(trace, list):
                raise ValueError(f"{group}:{item_id} reasoning_trace must be a list")
    if sources:
        missing = set(sources) - seen_keys
        extra = seen_keys - set(sources)
        if missing:
            raise ValueError(f"missing rows: {len(missing)}")
        if extra:
            raise ValueError(f"extra rows: {len(extra)}")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--test-json", default="")
    parser.add_argument("--expected-top-k", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    test_json = Path(args.test_json) if args.test_json else None
    counts = validate(Path(args.output_json), test_json, args.expected_top_k)
    print(json.dumps({"ok": True, "counts": counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
