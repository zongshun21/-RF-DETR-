# Implementation

## Scope

This repository adapts the Apache-2.0 RF-DETR 1.10.0 release to the InsPLAD
object-detection split. It provides a reproducible baseline and resolution
diagnostics. It does not claim a new architecture or a verified paper
contribution.

The pinned upstream source is vendored in `third_party/rfdetr-1.10.0/` and is
installed in editable mode. Its origin and source-distribution SHA256 are in
`third_party/SOURCE.json`.

## Model path

RF-DETR-S uses a windowed DINOv2-S encoder, a single P4 projector path, a
three-layer transformer decoder, hidden dimension 256 and 300 object queries.
The main source locations are:

- `third_party/rfdetr-1.10.0/src/rfdetr/config.py`: `RFDETRSmallConfig`;
- `third_party/rfdetr-1.10.0/src/rfdetr/models/lwdetr.py`: detector assembly;
- `third_party/rfdetr-1.10.0/src/rfdetr/models/transformer.py`: encoder/decoder;
- `third_party/rfdetr-1.10.0/src/rfdetr/models/backbone/`: DINOv2 and projector;
- `third_party/rfdetr-1.10.0/src/rfdetr/models/matcher.py`: Hungarian matching;
- `third_party/rfdetr-1.10.0/src/rfdetr/training/`: Lightning training path.

`train.py` validates the YAML through RF-DETR's Pydantic configuration, reads
the class order from the prepared dataset, sets the random seed before detector
construction, records an immutable run manifest and delegates optimization to
the official RF-DETR trainer. If an output directory already contains files, a
timestamped sibling directory is selected so an older experiment is retained.

## Local compatibility safeguards

`common.py` contains three safeguards found necessary during end-to-end tests:

1. An 18-class RF-DETR detection head can expose label 18 as background.
   `foreground_detections()` accepts labels 0–17, removes the background label
   and rejects IDs outside this space.
2. RF-DETR 1.10 `predict()` and its torchvision validation loader use different
   antialias settings. `ValidationPredictor` reuses the official validation
   transform so standalone evaluation matches the training validation protocol.
3. Lightweight checkpoints can omit geometry metadata. At the end of training,
   `stamp_checkpoint_metadata()` embeds resolution and positional-encoding
   configuration. `load_checkpoint()` reconstructs the correct geometry and
   compares every loaded tensor with the saved tensor. This prevents a 960
   checkpoint from silently reverting to the Small model's 512 geometry.

`prepare_data.py` maps the public COCO annotations into RF-DETR's expected
`train/_annotations.coco.json` and `valid/_annotations.coco.json` layout. Images
are linked rather than copied. `evaluate.py` converts contiguous prediction
labels back to the original COCO category IDs and writes overall and per-class
metrics. `predict.py` renders the same validated prediction path.

## Verification

`tests/test_data_and_metrics.py` covers sparse category mapping through the
official RF-DETR dataset class, COCO perfect/empty predictions, missing classes,
conflicting-filename quarantine, background filtering and checkpoint metadata
preservation. Runtime validation included 512/640/960 one-epoch smoke training,
two-GPU DDP, full-state resume, checkpoint reload and evaluation. Details are in
`reports/validation.md`.
