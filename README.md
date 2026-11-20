# DisasterSAR: Disaster Search and Rescue Image Classification

This repository contains a reproducible Python implementation of the image-classification workflow described in the college project report **Machine Learning-Based Disaster Prediction and Autonomous Search Operations Platform**.

## Project workflow

1. Read disaster-scene images labelled as **Body**, **Hand**, or **Leg**.
2. Resize and normalize images.
3. Train a Convolutional Neural Network (CNN) to learn image representations.
4. Extract CNN features and train three classifiers: Support Vector Machine (SVM), Random Forest, and Decision Tree.
5. Compare accuracy, macro precision, macro recall, macro F1-score, and confusion matrices.
6. Use a saved CNN feature extractor and selected classifier to classify a new image.

> **Scope and research honesty:** This repository implements the image-classification portion of the report. It does not implement physical robot navigation, geospatial localization, or a validated live disaster-response system. The PDF supplied for this project contains notebook screenshots and does not include the original notebook source or image dataset. This is therefore a clean reimplementation, not a recovery of the original source. No accuracy figure is guaranteed; use the metrics produced by your own dataset run.

## Repository layout

```text
Disastersar/
├── data/train/{Body,Hand,Leg}/
├── src/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── tests/
│   └── test_data.py
├── artifacts/                 # Generated after training; not committed
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or 3.11 recommended
- CPU is sufficient for a small dataset; a compatible GPU can speed up CNN training
- Internet access is needed only to install packages

Install on Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
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

## Dataset preparation

Place labelled images in these folders:

```text
data/train/Body/
data/train/Hand/
data/train/Leg/
```

Supported formats: JPG, JPEG, PNG, BMP, and WEBP. Each class needs at least two readable images for a stratified train/test split; more images per class are strongly recommended. Keep separate, unseen images for an additional final evaluation when possible.

The dataset is not included in this repository because it was not supplied with the report. Use images you are authorized to use, and document their source and license. Do not upload private or restricted images.

## Run the tests

After installing requirements, run:

```bash
python -m unittest discover -s tests -v
```

The tests create a temporary sample dataset and check image loading and stratified splitting. They do not train the CNN.

## Train and evaluate

From the repository root:

```bash
python train.py --data data/train --epochs 12 --batch-size 16 --test-size 0.2
```

The training script:
- Splits images into stratified training and test sets.
- Trains a compact CNN with basic image augmentation.
- Extracts CNN features.
- Fits SVM, Random Forest, and Decision Tree classifiers.
- Prints classification reports and saves metrics and confusion matrices.

Generated files are saved in `artifacts/`:

- `cnn_classifier.keras`
- `cnn_feature_extractor.keras`
- `label_encoder.joblib`
- `svm_model.joblib`
- `random_forest_model.joblib`
- `decision_tree_model.joblib`
- `metrics.json`
- Confusion-matrix PNG files

## Predict a new image

Default SVM:

```bash
python predict.py --image path/to/test_image.jpg
```

Choose another classifier:

```bash
python predict.py --image path/to/test_image.jpg --model random-forest
python predict.py --image path/to/test_image.jpg --model decision-tree
```

Predictions are limited to the classes represented in the training dataset. A predicted body-part label is not proof that a living person is present or that a rescue location has been identified. This prototype must not be used as the sole basis for emergency decisions.

## Evaluation and reporting

Use the values written to `artifacts/metrics.json` after a successful run. Accuracy alone can hide class imbalance, so review precision, recall, F1-score, class counts, and confusion matrices. For a fair final evaluation, test on images that were not used for training or model selection.

Do not copy the accuracy stated in the report into a results section unless you can reproduce it with the same dataset, split, and evaluation procedure.

## Limitations and future work

- Obtain and document the original labelled dataset.
- Evaluate on an independent disaster-scene dataset.
- Add video/live-camera inference and a human-reviewed alert workflow.
- Integrate robot hardware, mapping, and location telemetry only after separate validation and safety testing.
- Consider stronger pretrained CNN backbones and explainability tools.

## Project reference

The implementation follows the three-class CNN feature extraction plus SVM/Random Forest/Decision Tree workflow described in the supplied college project report.

## Automated checks

A GitHub Actions workflow runs the dataset-loader unit tests on pushes and pull requests to `main`. The lightweight test dependencies are listed separately in `requirements-test.txt`, so the tests do not need TensorFlow or a GPU.

## Separate victim-detection experiment

A separate YOLO-based person/victim detection module is available in [victim_detection/README.md](victim_detection/README.md). It uses the public PDD `879real survivor` image/YOLO-label collection and does not change this repository's project title or remove the original Body/Hand/Leg classifier. Prepare its dataset with `python victim_detection/prepare_dataset.py`, then train with `python victim_detection/train.py`. The detector reports object-detection metrics (precision, recall, mAP50, mAP50-95), which are not directly comparable to the original classifier's accuracy. Dataset images and model weights are downloaded/generated locally and are not committed.

## Current completion status

- [x] CNN + SVM, Random Forest, and Decision Tree training pipeline committed.
- [x] Image loading, preprocessing, stratified split, metrics, and confusion-matrix generation implemented.
- [x] Dataset preparation instructions and public research dataset references documented.
- [x] GitHub Actions test workflow added.
- [ ] Original Body/Hand/Leg labelled image dataset obtained and verified.
- [ ] Full CNN training and evaluation run on that dataset.
- [ ] Real model artifacts and measured metrics generated.

The remaining items depend on obtaining valid class-labelled images. The report's abstract says a GitHub dataset URL should follow, but the URL is absent from the supplied PDF text. The report's testing section also mentions DARPA2000, which is not reconciled with the three Body/Hand/Leg classes described elsewhere. These source inconsistencies must be resolved before claiming a faithful reproduction.
