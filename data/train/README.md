# Dataset instructions

This repository does not currently include the original labelled disaster-image dataset. The project report describes three image classes: `Body`, `Hand`, and `Leg`, but the supplied PDFs do not contain the image files or a verifiable dataset download URL.

## Required folder structure

```text
data/
└── train/
    ├── Body/
    ├── Hand/
    └── Leg/
```

Put correctly labelled JPG, JPEG, PNG, BMP, or WEBP images into the matching class folder. Use only images you have permission to use and record their source and license. Do not use unrelated images or generated placeholders as if they were the original project dataset.

Each class needs at least two readable images for the stratified split; substantially more diverse images are needed for meaningful model training. Keep a separate unseen test set if possible.

After adding images, run from the repository root:

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py --data data/train --epochs 12 --batch-size 16 --test-size 0.2
```

Training outputs are saved under `artifacts/`. Report only the metrics produced by that run. The accuracy numbers shown in the college report are not reproduced or guaranteed by this code.
