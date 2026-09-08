# Released weights

Release tag: `v1.0-insplad-baselines`.

Download and verify both models:

```bash
bash scripts/download_release_weights.sh
```

Expected files:

| File | Model | Input | Classes |
|---|---|---:|---:|
| `rfdetr_s_insplad_640_best.pth` | RF-DETR-S | 640 | 18 |
| `rfdetr_s_insplad_960_best.pth` | RF-DETR-S | 960 | 18 |

The release also contains `SHA256SUMS` and a metadata archive with the exact
training configuration, metric history, model configuration, runtime record and
full independent COCO evaluation outputs.

Evaluate a downloaded model:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --checkpoint weights/releases/rfdetr_s_insplad_640_best.pth \
  --dataset data/insplad \
  --output outputs/eval_640
```

Render predictions:

```bash
CUDA_VISIBLE_DEVICES=0 python predict.py \
  --checkpoint weights/releases/rfdetr_s_insplad_960_best.pth \
  --image /path/to/image.jpg \
  --output outputs/prediction.jpg
```

The weights are intended for research on the 18-class InsPLAD detection label
space. Dataset use remains subject to InsPLAD's CC BY-NC-SA 4.0 non-commercial
terms. RF-DETR source remains subject to its upstream Apache-2.0 license.
