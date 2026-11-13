import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.data import load_dataset, split_dataset
from src.model import build_feature_extractor

def main():
    parser = argparse.ArgumentParser(description="Train DisasterSAR classifiers.")
    parser.add_argument("--data", default="data/train", help="Folder with one subfolder per class.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    X, y, encoder = load_dataset(args.data)
    X_train, X_test, y_train, y_test = split_dataset(X, y, args.test_size)
    print(f"Images: {len(X)} | Classes: {list(encoder.classes_)}")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    cnn, extractor = build_feature_extractor(num_classes=len(encoder.classes_))
    cnn.fit(
        X_train, y_train,
        validation_split=0.15 if len(X_train) >= 10 else 0.0,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    train_features = extractor.predict(X_train, verbose=0)
    test_features = extractor.predict(X_test, verbose=0)

    models = {
        "SVM": SVC(kernel="rbf", C=2.0, probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
    }

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)
    cnn.save(output_dir / "cnn_classifier.keras")
    extractor.save(output_dir / "cnn_feature_extractor.keras")
    joblib.dump(encoder, output_dir / "label_encoder.joblib")

    results = {}
    for display_name, model in models.items():
        model.fit(train_features, y_train)
        predictions = model.predict(test_features)
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        }
        results[display_name] = metrics
        filename = display_name.lower().replace(" ", "_") + "_model.joblib"
        joblib.dump(model, output_dir / filename)
        print(f"\n{display_name}: {metrics}")
        print(classification_report(
            y_test, predictions, labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_, zero_division=0
        ))
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, predictions),
            display_labels=encoder.classes_
        ).plot(xticks_rotation=30)
        plt.title(f"{display_name} confusion matrix")
        plt.tight_layout()
        plt.savefig(output_dir / f"{filename.removesuffix('_model.joblib')}_confusion_matrix.png")
        plt.close()

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nTraining complete. Actual run metrics saved to artifacts/metrics.json")

if __name__ == "__main__":
    main()
