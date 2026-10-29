"""Per-frame CSV rows, including an empty-track row when no tracks exist."""
import csv
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from uuid import uuid4

FIELDS = ["frame_index", "video_timestamp_s", "track_id", "is_target", "left", "top",
          "right", "bottom", "has_detection", "confirmed", "target_visible",
          "target_lost", "target_status", "selected_target_id", "detection_confidence",
          "processing_latency_ms"]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def package_versions():
    result = {}
    for name in ("c3-tracker", "mediapipe", "opencv-contrib-python", "numpy",
                 "deep-sort-realtime", "torch", "torchvision"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "not installed"
    return result


class RunLogger:
    def __init__(self, root, metadata):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        self.directory = Path(root) / f"{stamp}_{uuid4().hex[:6]}"
        self.directory.mkdir(parents=True)
        write_json(self.directory / "run_metadata.json", {
            "created_utc": stamp, "python": platform.python_version(),
            "packages": package_versions(), "hardware": {
                "platform": platform.platform(), "processor": platform.processor()},
            **metadata})
        self.file = (self.directory / "tracking_log.csv").open("w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=FIELDS)
        self.writer.writeheader()

    def log_frame(self, index, timestamp, observations, target, latency_ms):
        for o in observations or [None]:
            row = dict(frame_index=index, video_timestamp_s=timestamp,
                       track_id=o.track_id if o else "", is_target=int(bool(o and o.track_id == target.target_id)),
                       has_detection=int(bool(o and o.observed)), confirmed=int(bool(o and o.confirmed)),
                       target_visible=int(target.visible), target_lost=int(target.status == "LOST"),
                       target_status=target.status, selected_target_id=target.target_id or "",
                       detection_confidence=o.detection_confidence if o else None,
                       processing_latency_ms=latency_ms)
            row.update(zip(("left", "top", "right", "bottom"), o.bbox if o else (None,) * 4))
            self.writer.writerow(row)
        self.file.flush()

    def close(self, summary):
        self.file.close()
        write_json(self.directory / "summary.json", summary)
