import csv
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_ava_annotations import convert, parse_row


class AVAAnnotationTests(unittest.TestCase):
    def test_8_column_row(self):
        r = parse_row(["abc", "1.000", ".1", ".2", ".5", ".6", "NOT_SPEAKING", "p1"], Path("a.csv"))
        self.assertEqual(r["entity_id"], "p1")
        self.assertEqual(r["x1"], "0.10000000")

    def test_5_column_row(self):
        r = parse_row(["abc", "1", "(0.1, 0.2, 0.5, 0.6)", "NOT_SPEAKING", "p1"], Path("a.csv"))
        self.assertEqual(r["x2"], "0.50000000")

    def test_reject_invalid_box(self):
        with self.assertRaises(ValueError):
            parse_row(["abc", "1", ".7", ".2", ".5", ".6", "x", "p1"], Path("a.csv"))

    def test_multiface_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / "raw", Path(tmp) / "out"
            src.mkdir()
            (src / "clip.csv").write_text(
                "abc,1,.1,.2,.3,.4,NOT_SPEAKING,p1\n"
                "abc,1,.5,.2,.7,.4,NOT_SPEAKING,p2\n"
                "abc,2,.1,.2,.3,.4,NOT_SPEAKING,p1\n"
                "abc,3,.9,.2,.4,.4,NOT_SPEAKING,p3\n",
                encoding="utf-8",
            )
            summary = convert(src, dst)
            self.assertEqual(summary["multi_face_timestamps"], 1)
            self.assertEqual(summary["invalid_rows"], 1)
            with (dst / "multi_face_timestamps.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["num_faces"], "2")


if __name__ == "__main__":
    unittest.main()
