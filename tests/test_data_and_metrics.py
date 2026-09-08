import json
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_data import prepare
from evaluate import coco_metrics
from pycocotools.coco import COCO


def test_checkpoint_metadata_preserves_weights(tmp_path):
    import torch
    from common import stamp_checkpoint_metadata
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    filename = tmp_path / "checkpoint_best_total.pth"
    torch.save({"model": {"test_weight": weight}}, filename)
    stamp_checkpoint_metadata(tmp_path, {"resolution": 960, "positional_encoding_size": 60})
    restored = torch.load(filename, weights_only=True)
    assert torch.equal(restored["model"]["test_weight"], weight)
    assert restored["model_config"]["resolution"] == 960
    assert not filename.with_suffix(".pth.tmp").exists()


def test_background_is_not_mapped_to_a_dataset_category():
    import numpy as np
    import supervision as sv
    from common import foreground_detections
    predictions = sv.Detections(xyxy=np.array([[0, 0, 10, 10]] * 3),
                               confidence=np.array([0.9, 0.8, 0.7]), class_id=np.array([0, 17, 18]))
    assert foreground_detections(predictions, 18).class_id.tolist() == [0, 17]
    predictions.class_id[2] = 19
    with pytest.raises(ValueError, match="outside"):
        foreground_detections(predictions, 18)


def fixture_data(root):
    (root / "annotations").mkdir(parents=True)
    for split in ("train", "val"):
        (root / split).mkdir()
        Image.new("RGB", (100, 80)).save(root / split / f"{split}.jpg")
        data = {"images": [{"id": 7, "file_name": f"{split}.jpg", "width": 100, "height": 80}],
                "categories": [{"id": 3, "name": "part"}, {"id": 19, "name": "rare"}],
                "annotations": [{"id": 9, "image_id": 7, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}]}
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))


def test_sparse_ids_preserved_and_idempotent(tmp_path):
    root, target = tmp_path / "source", tmp_path / "prepared"
    fixture_data(root)
    before = (root / "annotations/instances_train.json").read_bytes()
    prepare(root, target, verify_images=True)
    prepare(root, target, verify_images=True)
    assert (root / "annotations/instances_train.json").read_bytes() == before
    assert (target / "valid/val.jpg").is_symlink()
    assert not (target / "test").exists()
    assert json.loads((target / "class_mapping.json").read_text()) == [
        {"label": 0, "category_id": 3, "name": "part"}, {"label": 1, "category_id": 19, "name": "rare"}]
    from rfdetr.datasets.coco import CocoDetection
    official_dataset = CocoDetection(target / "train", target / "train/_annotations.coco.json",
                                     transforms=None, remap_category_ids=True)
    assert official_dataset.cat2label == {3: 0, 19: 1}
    assert official_dataset[0][1]["labels"].tolist() == [0]
    coco = COCO(str(target / "valid/_annotations.coco.json"))
    metrics = coco_metrics(coco, [{"image_id": 7, "category_id": 3, "bbox": [10, 10, 20, 20], "score": 0.9}], [7], tmp_path / "perfect")
    assert metrics["AP"] == pytest.approx(1.0)
    assert metrics["per_class"][1]["AP"] is None
    json.dumps(metrics, allow_nan=False)
    empty = coco_metrics(coco, [], [7], tmp_path / "empty")
    assert empty["AP"] == 0.0


def test_invalid_bbox_rejected(tmp_path):
    root = tmp_path / "source"
    fixture_data(root)
    p = root / "annotations/instances_train.json"
    data = json.loads(p.read_text())
    data["annotations"][0]["bbox"] = [90, 0, 20, 20]
    p.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match="Out-of-bounds"):
        prepare(root, tmp_path / "prepared")


def test_conflicting_filenames_quarantined_without_changing_source(tmp_path):
    root = tmp_path / "source"
    fixture_data(root)
    p = root / "annotations/instances_train.json"
    data = json.loads(p.read_text())
    data["images"].append({**data["images"][0], "id": 8})
    data["annotations"].append({**data["annotations"][0], "id": 10, "image_id": 8, "bbox": [30, 30, 20, 20]})
    p.write_text(json.dumps(data))
    before = p.read_bytes()
    report = prepare(root, tmp_path / "excluded")
    assert report["train"]["prepared_images"] == 0
    assert report["train"]["prepared_annotations"] == 0
    assert p.read_bytes() == before
    report = prepare(root, tmp_path / "original", duplicate_policy="keep")
    assert report["train"]["prepared_images"] == 2
    with pytest.raises(ValueError, match="new output"):
        prepare(root, tmp_path / "original", duplicate_policy="exclude")
