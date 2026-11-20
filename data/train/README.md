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


## Related research datasets (not drop-in replacements)

These public datasets are closer to search-and-rescue victim detection than ordinary disaster-scene classification, but their labels and image modalities do **not** match the repository's Body/Hand/Leg classes directly:

- **SARHuman**: 5,189 infrared images of people partially obscured by simulated post-disaster rubble; the Mendeley Data record states CC BY 4.0. It is an object-detection dataset, so it requires a detection-oriented pipeline or careful conversion of labels. See https://data.mendeley.com/datasets/28kr9v2tt9/1
- **Post-Disaster Dataset (PDD)**: includes 879 real survivor images with YOLO-format labels for post-disaster human detection. Its repository also notes that another 155-image test set is not publicly released due to copyright. Check the dataset repository and its cited paper before reuse: https://github.com/HaoqianSong/Post-Disaster-Dataset

Do not simply rename these datasets' labels to Body, Hand, and Leg. That would create incorrect ground truth. To keep the current three-class experiment, images must be labelled by a person according to the exact class definitions. Alternatively, redesign the task as human/victim detection and update the model, metrics, and report accordingly.
