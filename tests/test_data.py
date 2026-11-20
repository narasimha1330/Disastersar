import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.data import load_dataset, split_dataset


class DatasetTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        for label, value in (("Body", 40), ("Hand", 120), ("Leg", 210)):
            folder = root / label
            folder.mkdir(parents=True)
            for index in range(4):
                image = np.full((24, 24, 3), value + index, dtype=np.uint8)
                self.assertTrue(cv2.imwrite(str(folder / f'{index}.png'), image))

    def test_loads_three_labeled_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            X, y, encoder = load_dataset(str(root))
            self.assertEqual(X.shape, (12, 128, 128, 3))
            self.assertEqual(set(encoder.classes_), {'Body', 'Hand', 'Leg'})
            self.assertEqual(len(y), 12)

    def test_split_preserves_each_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._make_dataset(root)
            X, y, _ = load_dataset(str(root))
            X_train, X_test, y_train, y_test = split_dataset(X, y)
            self.assertEqual(len(X_train) + len(X_test), len(X))
            self.assertEqual(set(y_train), set(y_test))


if __name__ == "__main__":
    unittest.main()
