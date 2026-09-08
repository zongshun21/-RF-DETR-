# RF-DETR-S InsPLAD baselines

This release publishes the best checkpoints from the two completed RF-DETR-S
baseline runs on InsPLAD object detection.

| Asset | Input | Actual batch | AP50:95 | AP50 | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|---:|
| `rfdetr_s_insplad_640_best.pth` | 640 | 32 | 0.743162 | 0.904392 | 0.762276 | 0.855594 |
| `rfdetr_s_insplad_960_best.pth` | 960 | 16 | 0.754094 | 0.914716 | 0.793392 | 0.863309 |

The metrics come from independent COCO evaluation over all 2,626 validation
images using each run's best checkpoint. Training used BF16, EMA, seed 42 and
150 epochs. The 960 model improves AP50:95 by 0.010932 over the 640 baseline.

`SHA256SUMS` verifies both checkpoint files. The two compressed prediction
files contain the full COCO-format validation predictions used by the evaluator.
`rfdetr_insplad_metadata.tar.gz` contains the exact serialized configurations,
training histories, aggregate metrics and per-class metrics.

InsPLAD images and annotations are licensed CC BY-NC-SA 4.0. Commercial use is
prohibited. Model checkpoints are derivative research artifacts trained on that
data and should be used under compatible non-commercial terms. See
`datasets/InsPLAD-det/LICENSE-DATA.md` and the official InsPLAD project before
redistribution or reuse.
