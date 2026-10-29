"""Offline runtime summary and optional independently annotated target metrics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import statistics


def evaluate(log_path, annotation_path=None):
    with Path(log_path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    frames = defaultdict(list)
    for row in rows:
        frames[int(row["frame_index"])].append(row)
    if not frames:
        raise ValueError("Tracking log has no frames")
    result = {"frames": len(frames), "latency_mean_ms": statistics.mean(
        float(rs[0]["processing_latency_ms"]) for rs in frames.values())}
    if annotation_path is None:
        result["identity_metrics"] = "Not computed: independent annotations required"
        return result
    with Path(annotation_path).open(newline="", encoding="utf-8") as f:
        annotations = list(csv.DictReader(f))
    seen = set()
    visible = correct = assigned = wrong = switches = 0
    previous_person = None
    target_person = None
    for a in sorted(annotations, key=lambda a: int(a["frame_index"])):
        index = int(a["frame_index"])
        if index in seen or index not in frames:
            raise ValueError("Annotation frames must be unique and present in the log")
        seen.add(index)
        if a["target_visible"] not in ("0", "1"):
            raise ValueError("target_visible must be 0 or 1")
        label = a["target_person_id"].strip()
        if not label or (target_person is not None and label != target_person):
            raise ValueError("Use one nonempty target_person_id per trial")
        target_person = label
        is_visible = a["target_visible"] == "1"
        # Reviewer identifies the real person inside the currently highlighted box.
        person = a["selected_person_id"].strip()
        selected = frames[index][0]["target_visible"] == "1"
        if selected and not person:
            raise ValueError("Annotate selected_person_id for every highlighted target box")
        if not selected and person:
            raise ValueError("selected_person_id must be empty when no target is highlighted")
        visible += int(is_visible)
        hit = selected and is_visible and person == target_person
        correct += int(hit)
        assigned += int(selected)
        wrong += int(selected and not hit)
        if selected:
            switches += int(previous_person is not None and previous_person != person)
            previous_person = person
    if not annotations:
        raise ValueError("Annotation file is empty")
    result.update(annotated_frames=len(seen), target_recall=correct / visible if visible else None,
                  false_target_assignment_rate=wrong / assigned if assigned else None,
                  selected_person_switches=switches,
                  scope="Annotated frames only; switches across gaps may be missed")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--annotations", type=Path)
    args = p.parse_args()
    try:
        print(json.dumps(evaluate(args.log, args.annotations), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        p.exit(1, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
