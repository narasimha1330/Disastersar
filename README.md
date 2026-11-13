# DisasterSAR — Disaster Search & Rescue Image Classification

A reproducible project scaffold based on the college report `Machine Learning-Based Disaster Prediction and Autonomous Search Operations Platform`.

## What this code does
- Reads images from three class folders: `Body`, `Hand`, and `Leg`.
- Trains a small CNN as an image feature extractor.
- Trains SVM, Random Forest, and Decision Tree classifiers on the CNN features.
- Reports accuracy, precision, recall, F1-score, and a confusion matrix.
- Predicts the class of a new image using the saved CNN + SVM pipeline.

> Important: this repository scaffold is a clean reimplementation of the workflow described in the report. It is not a recovered copy of the original notebook. The PDF contains notebook screenshots and does not provide a complete downloadable source repository or the dataset. No accuracy result is claimed until you run training on the original/authorized dataset.

## Requirements
- Python 3.10 or 3.11 recommended
- Windows, macOS, or Linux
- CPU is sufficient for a small experiment; training can take time.

## 1. Create and activate a virtual environment

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Add your dataset

Arrange images like this:
```text
data/
  train/
    Body/
    Hand/
    Leg/
```
The dataset is not included because it was not present in the supplied PDF. Use only images you have permission to use and include their source/license in your project documentation.

## 3. Train and evaluate

```bash
python train.py --data data/train --epochs 8
```
The script saves the CNN feature extractor, SVM/Random Forest/Decision Tree models, label encoder, metrics JSON, and confusion-matrix images under `artifacts/`.

## 4. Predict an image

```bash
python predict.py --image path/to/test_image.jpg
```
The prediction is a class label (Body, Hand, or Leg), not a guarantee that a living victim has been located. This is an educational image-classification prototype, not a validated rescue or safety system.

## Project structure
```text
src/                 # Dataset loading and CNN model
train.py             # CNN feature extraction + ML classifier training predict.py           # Image prediction
requirements.txt     # Python dependencies
data/train/          # Add labeled images here
artifacts/            # Generated model files and metrics
```

## Before publishing results
1. Run training with your real dataset.
2. Test on images not used for training.
3. Record the actual metrics and dataset source/license.
4. Do not claim the report's accuracy as independently verified.
