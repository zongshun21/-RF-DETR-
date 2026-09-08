# Published weights

GitHub rejects individual files larger than 100 MB. Both best checkpoints are
therefore versioned as ordered 8 MiB chunks in `model_weights/chunks/`.
Reconstruct and verify both original files:

```bash
bash scripts/download_release_weights.sh
```

Expected files:

| File | Model | Input | Classes |
|---|---|---:|---:|
| `rfdetr_s_insplad_640_best.pth` | RF-DETR-S | 640 | 18 |
| `rfdetr_s_insplad_960_best.pth` | RF-DETR-S | 960 | 18 |

`model_weights/SHA256SUMS` records the hashes of the reconstructed checkpoints.
`release_metadata/` contains the exact training configuration, metric history,
model configuration, runtime record and full independent COCO evaluation
metrics. Do not load an individual `.part-*` file as a checkpoint.

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
