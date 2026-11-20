import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras

from src.data import load_dataset
from src.model import build_feature_extractor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN feature extraction with SVM, Random Forest, and Decision Tree."
    )
    parser.add_argument("--data", default="data/train")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 0.1 <= args.test_size <= 0.5:
        parser.error("--test-size must be between 0.1 and 0.5")

    keras.utils.set_random_seed(args.seed)
    X, y, encoder = load_dataset(args.data)
    class_counts = np.bincount(y)
    test_count = int(np.ceil(len(y) * args.test_size))
    if test_count < len(encoder.classes_):
        raise ValueError(
            f"Test split would contain only {test_count} images for "
            f"{len(encoder.classes_)} classes. Add more images or increase --test-size."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Loaded {len(X)} images; classes: {list(encoder.classes_)}")
    print(f"Class counts: {dict(zip(encoder.classes_, class_counts.tolist()))}")
    print(f"Training images: {len(X_train)} | Test images: {len(X_test)}")

    cnn, extractor = build_feature_extractor(num_classes=len(encoder.classes_))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    validation_split = 0.2 if len(X_train) >= 10 else 0.0
    fit_kwargs = {
        "x": X_train,
        "y": y_train,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "verbose": 1,
    }
    if validation_split:
        fit_kwargs["validation_split"] = validation_split
        fit_kwargs["callbacks"] = callbacks
    cnn.fit(**fit_kwargs)

    train_features = extractor.predict(X_train, verbose=0)
    test_features = extractor.predict(X_test, verbose=0)
    classifiers = {
        "svm": SVC(kernel="rbf", C=2.0, probability=True, random_state=args.seed),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=args.seed, class_weight="balanced"
        ),
        "decision_tree": DecisionTreeClassifier(
            random_state=args.seed, class_weight="balanced"
        ),
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    cnn.save(output_dir / "cnn_classifier.keras")
    extractor.save(output_dir / "cnn_feature_extractor.keras")
    joblib.dump(encoder, output_dir / "label_encoder.joblib")

    results = {}
    for name, classifier in classifiers.items():
        classifier.fit(train_features, y_train)
        predicted = classifier.predict(test_features)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision_macro": float(
                precision_score(y_test, predicted, average="macro", zero_division=0)
            ),
            "recall_macro": float(
                recall_score(y_test, predicted, average="macro", zero_division=0)
            ),
            "f1_macro": float(
                f1_score(y_test, predicted, average="macro", zero_division=0)
            ),
        }
        results[name] = metrics
        joblib.dump(classifier, output_dir / f"{name}_model.joblib")
        print(f"\n{name.upper()} metrics: {json.dumps(metrics, indent=2)}")
        print(
            classification_report(
                y_test,
                predicted,
                labels=np.arange(len(encoder.classes_)),
                target_names=encoder.classes_,
                zero_division=0,
            )
        )

        matrix = confusion_matrix(
            y_test, predicted, labels=np.arange(len(encoder.classes_))
        )
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=encoder.classes_
        )
        display.plot(xticks_rotation=30, cmap="Blues")
        plt.title(f"{name.replace('_', ' ').title()} confusion matrix")
        plt.tight_layout()
        plt.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=150)
        plt.close()

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "dataset": str(Path(args.data)),
                "image_count": int(len(X)),
                "class_counts": dict(
                    zip(encoder.classes_, class_counts.astype(int).tolist())
                ),
                "test_size": args.test_size,
                "seed": args.seed,
                "metrics": results,
            },
            file,
            indent=2,
        )
    print(f"\nTraining finished. Results and models are in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
