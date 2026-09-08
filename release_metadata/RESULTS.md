# Published baseline results

Independent COCO evaluation on all 2,626 validation images, BF16 inference,
unfiltered scores and maxDets `[1, 10, 100]`:

| Resolution | AP50:95 | AP50 | AP75 | AP medium | AP large | AR100 |
|---:|---:|---:|---:|---:|---:|---:|
| 640 | 0.743162 | 0.904392 | 0.762276 | 0.448654 | 0.749232 | 0.855594 |
| 960 | 0.754094 | 0.914716 | 0.793392 | 0.483107 | 0.760264 | 0.863309 |

COCO AP-small is undefined because this validation annotation set contains no
instance in COCO's original-pixel small area range. `sphere` is also undefined
because validation contains no instance of that category. See each subdirectory
for `metrics.json`, `per_class.csv`, the complete training metric history and
the serialized model/training configuration.
