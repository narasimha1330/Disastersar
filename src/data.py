from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

IMAGE_SIZE = (128, 128)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_dataset(data_dir: str):
    """Load labeled images from data_dir/<class_name>/*.jpg."""
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {base.resolve()}. "
            "Create data/train/Body, Hand, and Leg and add labeled images."
        )

    images, labels = [], []
    class_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if len(class_dirs) < 2:
        raise ValueError("Need at least two class folders containing images.")

    for class_dir in class_dirs:
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            image = cv2.imread(str(path))
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(class_dir.name)

    if not images:
        raise ValueError(f"No readable images found in {base.resolve()}.")

    X = np.asarray(images, dtype=np.float32) / 255.0
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    counts = np.bincount(y)
    if np.min(counts) < 2:
        raise ValueError(
            "Each class needs at least two images for a stratified train/test split. "
            f"Class counts: {dict(zip(encoder.classes_, counts.tolist()))}"
        )
    return X, y, encoder

def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
