"""Run victim/person detection on an image or video using trained YOLO weights."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Image, video, folder, or camera index (0).")
    parser.add_argument("--weights", default="victim_detection/runs/pdd_yolov8n/weights/best.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", help="Save annotated predictions.")
    args = parser.parse_args()
    weights = Path(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(f"Trained weights not found: {weights}. Train the model first.")
    model = YOLO(str(weights))
    results = model.predict(source=args.source, conf=args.conf, save=args.save,
                            project="victim_detection/runs", name="predictions")
    for result in results:
        count = len(result.boxes) if result.boxes is not None else 0
        print(f"{Path(str(result.path)).name}: {count} person detection(s)")


if __name__ == "__main__":
    main()
