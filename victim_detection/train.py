"""Train and evaluate a YOLO victim/person detector."""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="victim_detection/data/yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model/weights.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="Examples: 0 for CUDA GPU, cpu, or blank for auto.")
    args = parser.parse_args()
    data = Path(args.data)
    if not data.is_file():
        raise FileNotFoundError("Dataset YAML not found. Run: python victim_detection/prepare_dataset.py")
    model = YOLO(args.model)
    train_args = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                      batch=args.batch, project="victim_detection/runs",
                      name="pdd_yolov8n", patience=10, seed=42, workers=2)
    if args.device:
        train_args["device"] = args.device
    result = model.train(**train_args)
    best = Path(result.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed but best weights were not found: {best}")
    print(f"Best checkpoint: {best}")
    metrics = model.val(data=str(data), split="test", imgsz=args.imgsz,
                        project="victim_detection/runs", name="test_evaluation")
    print("Held-out test metrics:")
    print(f"mAP50: {float(metrics.box.map50):.4f}")
    print(f"mAP50-95: {float(metrics.box.map):.4f}")
    print(f"Precision: {float(metrics.box.mp):.4f}")
    print(f"Recall: {float(metrics.box.mr):.4f}")


if __name__ == "__main__":
    main()
