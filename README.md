# RF-DETR-S for InsPLAD object detection

[中文说明](README_zh-CN.md)

Reproducible RF-DETR-S baselines for the InsPLAD UAV power-line asset detection
dataset, with 640 and 960 input-resolution training, COCO evaluation, per-class
metrics, prediction rendering and audited data conversion.

This repository pins the official [RF-DETR](https://github.com/roboflow/rf-detr)
1.10.0 source release. It is a baseline and research scaffold; it does not claim
a novel architecture.

## Results

Both models were trained for 150 epochs on one RTX 4090 24 GB using BF16, EMA
and seed 42. Values below are the best validation values recorded during each
single run. InsPLAD has no independent test split in this project.

| Model | Input | Batch | AP50:95 | AP50 | AP75 | mAR@100 |
|---|---:|---:|---:|---:|---:|---:|
| RF-DETR-S | 640 | 32 | 0.7432 | 0.9044 | 0.7623 | 0.8556 |
| RF-DETR-S | 960 | 16 | 0.7541 | 0.9147 | 0.7934 | 0.8633 |

These numbers use the repository's independent full-validation COCO evaluator
on each run's best checkpoint. Full metric histories, per-class results and
serialized configurations are in `release_metadata/`. Download
the best inference checkpoints from the
[`v1.0-insplad-baselines`](https://github.com/zongshun21/-RF-DETR-/releases/tag/v1.0-insplad-baselines)
release after it is published.

## Repository layout

```text
.
├── configs/                    # 512/640/960 and smoke YAML configurations
├── datasets/InsPLAD-det/       # COCO annotations, audit and data license
├── docs/                       # implementation, dataset, training and weight guides
├── release_metadata/           # exact configurations and complete metric histories
├── reports/                    # validation record and conflict previews
├── scripts/                    # setup, weight download and conflict rendering
├── third_party/rfdetr-1.10.0/  # pinned editable RF-DETR source
├── prepare_data.py
├── train.py
├── evaluate.py
└── predict.py
```

## Installation

Python 3.12 and an NVIDIA driver compatible with the PyTorch cu118 wheel were
used for validation.

```bash
git clone https://github.com/zongshun21/-RF-DETR-.git
cd -RF-DETR-
bash scripts/setup.sh
source .venv/bin/activate
python -m pip check
```

Download the official RF-DETR-S COCO initialization used for fine-tuning:

```bash
mkdir -p weights
curl -fL -C - \
  https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth \
  -o weights/rf-detr-small.pth
echo 'fb37061c1af7bace359c91b723a8d5c1  weights/rf-detr-small.pth' | md5sum -c -
```

## Dataset preparation

The COCO annotations are included under `datasets/InsPLAD-det/annotations/`.
Original JPG files are not included. Download the object-detection images from
the [official InsPLAD project](https://github.com/andreluizbvs/InsPLAD), which
licenses the data under CC BY-NC-SA 4.0 and prohibits commercial use. Arrange:

```text
/path/to/InsPLAD-det/
├── annotations/instances_train.json
├── annotations/instances_val.json
├── train/*.jpg
└── val/*.jpg
```

Then audit and prepare the RF-DETR layout:

```bash
python prepare_data.py \
  --source /path/to/InsPLAD-det \
  --output data/insplad \
  --verify-images
```

The source data is not changed. The prepared directory uses symbolic links, so
images are not duplicated.

The original train JSON has 7,981 image records and 22,635 boxes. It contains
46 duplicate filenames linked to two different image IDs and inconsistent box
sets. The default reproducible protocol excludes both ambiguous records for
each filename, producing 7,889 training records and 22,296 boxes. Full details
are in [docs/DATASET.md](docs/DATASET.md). The validation split has no `sphere`
instance, so its per-class AP is undefined rather than zero.

## Training

Check the resolved configuration:

```bash
python train.py --config configs/small_640.yaml --dry-run
```

Run 640 and 960 experiments on separate GPUs:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/small_640.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/small_960.yaml
```

Training displays a tqdm progress bar. If an output directory exists, a new
timestamped directory is created automatically. Parameters can be overridden:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/small_960.yaml \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --epochs 150 \
  --output outputs/small_960_bs4
```

Effective batch size is `batch_size × grad_accum_steps × devices`. See
[docs/TRAINING.md](docs/TRAINING.md) for DDP, checkpoint and experiment details.

## Evaluation and prediction

Download the released fine-tuned weights:

```bash
bash scripts/download_release_weights.sh
```

Evaluate with the fixed COCO protocol:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint weights/releases/rfdetr_s_insplad_640_best.pth \
  --dataset data/insplad \
  --output outputs/eval_640
```

Outputs are `predictions.coco.json`, `metrics.json` and `per_class.csv`.
Evaluation uses unfiltered scores, original-image coordinates and
`maxDets=[1,10,100]`.

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --checkpoint weights/releases/rfdetr_s_insplad_960_best.pth \
  --image /path/to/image.jpg \
  --output outputs/prediction.jpg
```

## Implementation notes

The project uses RF-DETR's official trainer and makes the pinned source editable
for architecture research. Local safeguards preserve high-resolution positional
embeddings, remove the background output from the 18-class COCO label space and
align standalone inference with the validation preprocessing. See
[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) and
[reports/validation.md](reports/validation.md).

Run tests with:

```bash
python -m pytest -q
```

## Citation and licenses

Please cite RF-DETR and the InsPLAD dataset when using this project. InsPLAD:

> A. L. B. V. e Silva et al., “InsPLAD: A Dataset and Benchmark for Power Line
> Asset Inspection in UAV Images,” International Journal of Remote Sensing,
> 44(23), 7294–7320, 2023. DOI: 10.1080/01431161.2023.2283900.

The vendored RF-DETR source is Apache-2.0 licensed; see
`third_party/rfdetr-1.10.0/LICENSE`. Dataset annotations and derivatives are
CC BY-NC-SA 4.0; see `datasets/InsPLAD-det/LICENSE-DATA.md`.
