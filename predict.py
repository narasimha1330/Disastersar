import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np
from tensorflow import keras

def main():
    parser = argparse.ArgumentParser(description="Predict Body/Hand/Leg from an image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default="artifacts/svm_model.joblib")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path.resolve()}")

    extractor_path = Path("artifacts/cnn_feature_extractor.keras")
    encoder_path = Path("artifacts/label_encoder.joblib")
    for path in (extractor_path, encoder_path, Path(args.model)):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing trained artifact: {path}. Run `python train.py` first."
            )

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128)).astype(np.float32) / 255.0
    features = keras.models.load_model(extractor_path).predict(
        np.expand_dims(image, axis=0), verbose=0
    )
    model = joblib.load(args.model)
    encoder = joblib.load(encoder_path)
    predicted_id = int(model.predict(features)[0])
    predicted_label = encoder.inverse_transform([predicted_id])[0]

    print(f"Predicted class: {predicted_label}")
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(features)[0]))
        print(f"Model confidence: {confidence:.3f} (not a safety guarantee)")

if __name__ == "__main__":
    main()
