# Victim Detection Module (separate experiment)

This module adds a single-class **person/victim bounding-box detector** to the existing DisasterSAR repository. It does not rename the college project or replace/delete the original Body/Hand/Leg CNN + SVM/Random Forest/Decision Tree code.

## Dataset

Uses the public [Post-Disaster Dataset](https://github.com/HaoqianSong/Post-Disaster-Dataset), specifically its `879real survivor` collection: 879 post-disaster images and corresponding YOLO-format labels. The dataset repository describes these as survivor images for UAV search and rescue. Its README does not provide an explicit open-source license, so check with the dataset authors and follow the paper's usage terms before redistribution or commercial use. The restricted `155real survivor` test set is not used.

The labels are expected to use YOLO class ID 0 for person, with normalized `class x_center y_center width height` rows. The preparation script validates that assumption and stops on unexpected class IDs or invalid boxes instead of silently relabeling data.

## Install and run (from repository root)

Use Python 3.10 or 3.11 in a separate virtual environment if the original TensorFlow environment has conflicts:

```powershell
python -m venv .venv-victim
.venv-victim\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r victim_detection/requirements.txt
python victim_detection/prepare_dataset.py
python victim_detection/train.py --epochs 50 --batch 8
python victim_detection/predict.py --source path/to/disaster_image.jpg --save
```

The preparation script downloads the source archive, validates image/label pairs, and creates 70% train, 15% validation, and 15% held-out test splits. It writes `victim_detection/data/yolo/data.yaml`. Dataset images and generated runs are excluded from Git.

For a CUDA GPU, pass `--device 0`; for CPU use `--device cpu`. Training downloads the configured pretrained YOLO checkpoint if it is not already available. The first run therefore requires internet access and can take time.

## Outputs

- Best model: `victim_detection/runs/pdd_yolov8n/weights/best.pt`
- Validation and held-out test metrics: Ultralytics run output
- Annotated predictions: `victim_detection/runs/predictions/` when `--save` is used

Report mAP50, mAP50-95, precision, and recall from the actual held-out test evaluation. Do not claim a score until training and test evaluation have completed successfully.

## Important evaluation note

The default split is a seeded random image-level split, not a sequence/video-group split. Related frames may appear in different splits, so the held-out score can be optimistic. For a publication-quality result, audit near-duplicates and group related frames before splitting. This model detects visible person-shaped bounding boxes; it cannot prove that a detected person is alive, determine medical condition, or locate a victim in geographic coordinates. A trained human operator must verify every alert.

## Research reference

Song et al., *PDD: Post-Disaster Dataset for Human Detection and Performance Evaluation*, IEEE Transactions on Instrumentation and Measurement, 2023. Dataset and labels: https://github.com/HaoqianSong/Post-Disaster-Dataset
