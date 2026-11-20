"""Download and split the public PDD survivor dataset into YOLO train/val/test folders."""
from __future__ import annotations
import argparse
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://codeload.github.com/HaoqianSong/Post-Disaster-Dataset/zip/refs/heads/main"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_and_extract(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "post-disaster-dataset.zip"
    if not archive.exists():
        print("Downloading the public Post-Disaster Dataset archive...")
        urllib.request.urlretrieve(ZIP_URL, archive)
    extracted = destination / "source"
    marker = extracted / "Post-Disaster-Dataset-main"
    if not marker.exists():
        extracted.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extracted)
    return marker / "879real survivor"


def valid_yolo_label(path: Path) -> list[str]:
    rows = []
    if not path.exists():
        raise ValueError(f"Missing annotation file: {path}")
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO values, got {len(fields)}")
        class_id = int(fields[0])
        coords = [float(v) for v in fields[1:]]
        if class_id != 0:
            raise ValueError(f"{path}:{line_no}: expected person class 0, got {class_id}")
        cx, cy, width, height = coords
        if not all(0 <= v <= 1 for v in coords) or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_no}: invalid normalized bounding box")
        if cx - width / 2 < -1e-6 or cy - height / 2 < -1e-6 or cx + width / 2 > 1.000001 or cy + height / 2 > 1.000001:
            raise ValueError(f"{path}:{line_no}: bounding box falls outside image")
        rows.append(" ".join(fields))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", default="victim_detection/data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = download_and_extract(Path(args.destination).resolve())
    image_dir, label_dir = root / "images", root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError("Expected source folders '879real survivor/images' and 'labels'.")
    pairs = []
    for image in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label = label_dir / f"{image.stem}.txt"
        valid_yolo_label(label)
        pairs.append((image, label))
    if len(pairs) < 10:
        raise ValueError(f"Only {len(pairs)} valid image/label pairs found; refusing to create a tiny split.")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    splits = {"train": pairs[:n_train], "val": pairs[n_train:n_train+n_val], "test": pairs[n_train+n_val:]}
    output = Path(args.destination).resolve() / "yolo"
    if output.exists():
        shutil.rmtree(output)
    for split, items in splits.items():
        for kind in ("images", "labels"):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
        for image, label in items:
            shutil.copy2(image, output / "images" / split / image.name)
            shutil.copy2(label, output / "labels" / split / label.name)
    yaml_text = (
        f"path: {output.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: person\n"
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Prepared {n} image/label pairs at {output}")
    print("Split counts:", {key: len(value) for key, value in splits.items()})
    print("Class mapping: 0 = person (survivor/person bounding boxes)")
    print("Note: random image split; visually related video frames may cross splits.")


if __name__ == "__main__":
    main()
