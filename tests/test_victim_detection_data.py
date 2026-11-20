import tempfile
import unittest
from pathlib import Path

from victim_detection.prepare_dataset import valid_yolo_label


class VictimDetectionLabelTests(unittest.TestCase):
    def test_valid_person_boxes_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 0.5 0.5 0.2 0.4\n0 0.25 0.25 0.1 0.1\n", encoding="utf-8")
            rows = valid_yolo_label(label)
            self.assertEqual(len(rows), 2)

    def test_wrong_class_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("1 0.5 0.5 0.2 0.4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected person class 0"):
                valid_yolo_label(label)

    def test_invalid_box_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            label = Path(directory) / "sample.txt"
            label.write_text("0 1.2 0.5 0.2 0.4\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid normalized bounding box"):
                valid_yolo_label(label)


if __name__ == "__main__":
    unittest.main()
