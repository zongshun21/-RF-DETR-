"""COCO bounding-box AP/AR on the original validation split, with per-class CSV."""
import argparse
import csv
import time
from collections import Counter

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from common import ValidationPredictor, load_checkpoint, path, read_json, write_json


def coco_metrics(coco, predictions, image_ids, output):
    if predictions:
        detected = coco.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = {"images": list(coco.imgs.values()), "categories": list(coco.cats.values()), "annotations": []}
        detected.createIndex()
    evaluator = COCOeval(coco, detected, "bbox")
    evaluator.params.imgIds = image_ids
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    metrics = {key: (float(value) if value >= 0 else None) for key, value in zip(keys, evaluator.stats)}
    image_set = set(image_ids)
    counts = Counter(a["category_id"] for a in coco.anns.values() if a["image_id"] in image_set and not a.get("iscrowd", 0))
    rows = []
    def mean_valid(values):
        values = values[values > -1]
        return float(values.mean()) if values.size else None
    for index, cid in enumerate(evaluator.params.catIds):
        precision = evaluator.eval["precision"][:, :, index, 0, -1]
        recall = evaluator.eval["recall"][:, index, 0, -1]
        rows.append({"category_id": int(cid), "name": coco.cats[cid]["name"], "instances": counts[cid],
                     "AP": mean_valid(precision), "AP50": mean_valid(precision[0]), "AR100": mean_valid(recall)})
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics.update({"images": len(image_ids), "max_dets": 100, "absent_classes": [r["name"] for r in rows if r["instances"] == 0], "per_class": rows})
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="data/insplad")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--limit", type=int, help="Debug subset only; omitted for full validation")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    output, dataset = path(args.output), path(args.dataset)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new output directory: {output}")
    mapping = read_json(dataset / "class_mapping.json")
    label_to_category = {c["label"]: c["category_id"] for c in mapping}
    coco = COCO(str(dataset / "valid/_annotations.coco.json"))
    image_ids = sorted(coco.imgs)[:args.limit]
    model = load_checkpoint(args.checkpoint, args.device)
    names = model.model.class_names
    if names != [c["name"] for c in mapping]:
        raise ValueError(f"Checkpoint class order differs from dataset: {names}")
    predictor = ValidationPredictor(model, args.precision)
    predictions = []
    start = time.perf_counter()
    for image_id in tqdm(image_ids, desc="Validation inference"):
        with Image.open(dataset / "valid" / coco.imgs[image_id]["file_name"]) as image:
            # No confidence filtering for AP. COCOeval applies maxDets=100 per image/category.
            detections = predictor(image, threshold=0.0)
        for box, score, label in zip(detections.xyxy, detections.confidence, detections.class_id):
            if int(label) not in label_to_category:
                raise ValueError(f"Unexpected predicted label: {label}")
            x1, y1, x2, y2 = map(float, box)
            predictions.append({"image_id": image_id, "category_id": label_to_category[int(label)],
                                "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score)})
    elapsed = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "predictions.coco.json", predictions)
    metrics = coco_metrics(coco, predictions, image_ids, output)
    metrics.update({"checkpoint": str(path(args.checkpoint)), "resolution": model.model_config.resolution,
                    "precision": predictor.precision, "preprocessing": "official torchvision val pipeline (antialias=True)",
                    "debug_subset": args.limit is not None, "inference_wall_seconds_including_io": elapsed,
                    "evaluation_protocol": "COCO bbox, original image pixels, maxDets=[1,10,100], scores unfiltered"})
    write_json(output / "metrics.json", metrics)


if __name__ == "__main__":
    main()
