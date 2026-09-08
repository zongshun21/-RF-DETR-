# Training and experiment management

## Environment

The verified environment is Python 3.12, PyTorch 2.7.1+cu118, torchvision
0.22.1+cu118 and RF-DETR 1.10.0. Create it with:

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m pip check
```

Download the official RF-DETR-S COCO initialization:

```bash
mkdir -p weights
curl -fL -C - \
  https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth \
  -o weights/rf-detr-small.pth
echo 'fb37061c1af7bace359c91b723a8d5c1  weights/rf-detr-small.pth' | md5sum -c -
```

Prepare the dataset as described in `DATASET.md`, then inspect a configuration:

```bash
python train.py --config configs/small_640.yaml --dry-run
```

## Published runs

The two released baselines were trained for 150 epochs on one RTX 4090 24 GB
with BF16, EMA, seed 42, fixed resolution, no multi-scale augmentation and no
scale jitter. The exact serialized configurations and complete metric histories
are under `release_metadata/`.

| Model | Input | Train batch | Accumulation | Effective batch | Best val AP50:95 | Best val AP50 |
|---|---:|---:|---:|---:|---:|---:|
| RF-DETR-S | 640 | 32 | 1 | 32 | 0.7432 | 0.9044 |
| RF-DETR-S | 960 | 16 | 1 | 16 | 0.7541 | 0.9147 |

These are independent full-validation COCO results from each run's selected
best checkpoint. They are not test-set results and do not establish statistical
significance. Training-time metric histories are retained for audit.
The table records the completed runs used for the published checkpoints. For
new runs, `configs/small_640.yaml` defaults to batch 16 as the conservative
24 GB GPU starting point.

Run each resolution on a separate GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/small_640.yaml
CUDA_VISIBLE_DEVICES=1 python train.py --config configs/small_960.yaml
```

The terminal displays a tqdm progress bar. Override common parameters without
editing YAML:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/small_960.yaml \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --epochs 150 \
  --output outputs/small_960_bs4
```

Effective batch is `batch_size × grad_accum_steps × devices`. Batch capacity is
environment-dependent; reduce the physical batch and increase accumulation if
CUDA runs out of memory.

## DDP and output behavior

Use the RF-DETR/Lightning device option directly; do not wrap it in `torchrun`:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --config configs/small_640.yaml \
  --devices 2 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --output outputs/small_640_ddp
```

When the requested output directory already contains files, `train.py` creates
a timestamped sibling. It never overwrites an earlier run. Each output contains
the final resolved configuration, dependency snapshot, metrics, TensorBoard
events, full Lightning checkpoints and lightweight inference checkpoints.

Use `last.ckpt` only when optimizer/scheduler/EMA continuation is required:

```bash
python train.py --config configs/small_640.yaml \
  --resume outputs/small_640/last.ckpt
```

Use `checkpoint_best_total.pth` for inference and evaluation.
