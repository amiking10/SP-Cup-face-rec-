"""Prepare public AVA ActiveSpeaker face-track CSVs for a C3 benchmark.

No network access and no video download. Only annotation metadata is processed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

SCHEMA = ["video_id", "timestamp_s", "entity_id", "x1", "y1", "x2", "y2", "label", "source_file"]


def parse_row(row: list[str], source: Path) -> dict[str, str]:
    if len(row) == 8:
        vid, ts, x1, y1, x2, y2, label, entity = row
        coords = [float(x1), float(y1), float(x2), float(y2)]
    elif len(row) == 5:
        vid, ts, bbox, label, entity = row
        coords = [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", bbox)]
        if len(coords) != 4:
            raise ValueError(f"expected four bbox coordinates, got {len(coords)}")
    else:
        raise ValueError(f"expected five or eight CSV fields, got {len(row)}")
    if not vid.strip() or not entity.strip():
        raise ValueError("missing video or entity ID")
    time = float(ts)
    if not math.isfinite(time) or time < 0:
        raise ValueError("invalid timestamp")
    x1, y1, x2, y2 = coords
    if not all(math.isfinite(v) for v in coords):
        raise ValueError("nonfinite coordinate")
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("expected normalized bbox with x1 < x2 and y1 < y2")
    return dict(zip(SCHEMA, [vid.strip(), format(time, ".6f"), entity.strip(),
                            *[format(v, ".8f") for v in coords], label.strip(), str(source)]))


def convert(folder: Path, output: Path) -> dict:
    if not folder.is_dir():
        raise FileNotFoundError(f"annotation directory not found: {folder}")
    sources = sorted(folder.rglob("*.csv"))
    if not sources:
        raise FileNotFoundError(f"no CSV annotation files under {folder}")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    rejected = []
    faces_by_frame = defaultdict(set)
    for source in sources:
        with source.open("r", newline="", encoding="utf-8-sig") as file:
            for line_no, row in enumerate(csv.reader(file), start=1):
                if not row or not any(s.strip() for s in row):
                    continue
                try:
                    record = parse_row(row, source.relative_to(folder))
                except (ValueError, TypeError) as ex:
                    rejected.append(f"{source}:{line_no}: {ex}")
                    continue
                all_rows.append(record)
                faces_by_frame[(record["video_id"], record["timestamp_s"])].add(record["entity_id"])

    with (output / "annotations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=SCHEMA)
        writer.writeheader()
        writer.writerows(all_rows)

    multiface = sorted((video, timestamp, len(ids))
                       for (video, timestamp), ids in faces_by_frame.items() if len(ids) >= 2)
    with (output / "multi_face_timestamps.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["video_id", "timestamp_s", "num_faces"])
        writer.writerows(multiface)

    summary = {
        "source_csv_files": len(sources),
        "valid_annotations": len(all_rows),
        "invalid_rows": len(rejected),
        "unique_video_ids": len({r["video_id"] for r in all_rows}),
        "multi_face_timestamps": len(multiface),
        "sample_rejections": rejected[:20],
        "note": "Track-local entity IDs are NOT guaranteed persistent biometric identities.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Extracted AVA CSV directory")
    parser.add_argument("--output", type=Path, required=True, help="Processed annotation directory")
    args = parser.parse_args()
    print(json.dumps(convert(args.input, args.output), indent=2))


if __name__ == "__main__":
    main()
