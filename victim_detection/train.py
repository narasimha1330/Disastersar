"""Train and evaluate a YOLO victim/person detector."""
import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="victim_detection/data/yolo/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model/weights.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="Examples: 0 for CUDA GPU, cpu, or blank for auto.")
    args = parser.parse_args()
    data = Path(args.data)
    if not data.is_file():
        raise FileNotFoundError("Dataset YAML not found. Run: python victim_detection/prepare_dataset.py")
    model = YOLO(args.model)
    train_args = dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz,
                      batch=args.batch, project="victim_detection/runs",
                      name="pdd_yolov8n", patience=5, seed=42, workers=2)
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training completed but best weights were not found: {best}")
    print(f"Best checkpoint: {best}")
    metrics = model.val(data=str(data), split="test", imgsz=args.imgsz,
                        project="victim_detection/runs", name="test_evaluation")
    report = {
        "test_images": int(metrics.box.nt_per_class.sum()) if hasattr(metrics.box, "nt_per_class") else None,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "best_checkpoint": str(best),
    }
    report_path = Path("victim_detection/runs/test_metrics.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Held-out test metrics:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
