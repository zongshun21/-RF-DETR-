# InsPLAD data preparation

## License and source

InsPLAD is published under CC BY-NC-SA 4.0 and explicitly disallows commercial
use. The repository includes the object-detection COCO annotations with
attribution, but not the image files. Download the object-detection archive from
the [official project](https://github.com/andreluizbvs/InsPLAD) and cite the
[dataset paper](https://doi.org/10.1080/01431161.2023.2283900).

## Expected source layout

```text
/path/to/InsPLAD-det/
├── annotations/
│   ├── instances_train.json
│   └── instances_val.json
├── train/
│   └── *.jpg
└── val/
    └── *.jpg
```

The released annotations are also stored in
`datasets/InsPLAD-det/annotations/`. If the downloaded archive lacks those
files, copy them into its `annotations/` directory.

## Audit and conversion

```bash
python prepare_data.py \
  --source /path/to/InsPLAD-det \
  --output data/insplad \
  --verify-images
```

The command validates image IDs, category IDs, image files, decoded dimensions,
finite boxes, positive areas and image bounds. It then creates RF-DETR's
Roboflow-style COCO layout with symbolic links to the original JPG files. The
source dataset is never modified.

Original split statistics:

| Split | JSON image records | Unique files | Boxes |
|---|---:|---:|---:|
| train | 7,981 | 7,935 | 22,635 |
| val | 2,626 | 2,626 | 6,324 |

There are 18 categories in the annotations. The validation split contains no
`sphere` instance, so its per-class AP is undefined and is stored as `null`.

## Duplicate-filename conflict

The train JSON contains 46 filenames assigned to two image IDs each. The paired
records contain different bounding boxes, while only one JPG exists for each
filename. The correct record cannot be inferred safely from the available file.

The default `--duplicate-policy exclude` quarantines both records for every
conflicting filename. The prepared training set therefore has 7,889 image
records and 22,296 boxes. The complete mapping and excluded annotations are in
`datasets/InsPLAD-det/audit/train_conflicts.json`; six visual comparisons are in
`reports/conflicts/`. All model comparisons must use the same policy.

To reproduce the unmodified official JSON behavior in a separate directory:

```bash
python prepare_data.py \
  --source /path/to/InsPLAD-det \
  --output data/insplad_original \
  --duplicate-policy keep
```

Do not compare the cleaned protocol directly with published results that used
the unmodified records without disclosing the difference. There is no test
split in this project; repeated model selection on `val` should ultimately be
confirmed on an independently defined test protocol.
