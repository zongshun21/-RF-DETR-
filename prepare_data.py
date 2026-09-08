"""Audit original COCO data and build RF-DETR's local train/valid layout."""
import argparse
import collections
import hashlib
import math
from pathlib import Path

from common import ROOT, path, read_json, write_json


def audit_split(root, split, verify_images=False):
    filename = root / "annotations" / f"instances_{split}.json"
    data = read_json(filename)
    images = {i["id"]: i for i in data["images"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}
    assert len(images) == len(data["images"]), "Duplicate image IDs"
    assert len(categories) == len(data["categories"]), "Duplicate category IDs"
    assert len(set(categories.values())) == len(categories), "Duplicate category names"
    assert len({a["id"] for a in data["annotations"]}) == len(data["annotations"]), "Duplicate annotation IDs"
    by_filename = collections.defaultdict(list)
    for i in images.values():
        by_filename[i["file_name"]].append(i["id"])
    duplicates = {name: ids for name, ids in by_filename.items() if len(ids) > 1}
    for i in images.values():
        rel = Path(i["file_name"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe filename: {rel}")
        p = root / split / rel
        if not p.is_file():
            raise FileNotFoundError(p)
        assert i["width"] > 0 and i["height"] > 0, f"Invalid dimensions: {p}"
        if verify_images:
            from PIL import Image
            with Image.open(p) as image:
                assert image.size == (i["width"], i["height"]), f"Size mismatch: {p}"
                image.verify()
    counts = collections.Counter()
    sizes = {str(r): collections.Counter() for r in (512, 640, 960)}
    for a in data["annotations"]:
        assert a["image_id"] in images and a["category_id"] in categories, f"Orphan annotation: {a['id']}"
        x, y, w, h = a["bbox"]
        i = images[a["image_id"]]
        assert all(math.isfinite(v) for v in (x, y, w, h)), f"Nonfinite bbox: {a['id']}"
        assert w > 0 and h > 0 and x >= 0 and y >= 0, f"Invalid bbox: {a['id']}"
        assert x+w <= i["width"]+1e-4 and y+h <= i["height"]+1e-4, f"Out-of-bounds bbox: {a['id']}"
        assert math.isfinite(a["area"]) and a["area"] > 0, f"Invalid area: {a['id']}"
        counts[a["category_id"]] += 1
        for r in (512, 640, 960):
            # Diagnostic only: square-resized box area, NOT COCO's original-pixel AP_s definition.
            area = w * h * r * r / (i["width"] * i["height"])
            sizes[str(r)]["small" if area < 32**2 else "medium" if area < 96**2 else "large"] += 1
    report = {
        "images": len(images), "annotations": len(data["annotations"]),
        "unique_image_files": len(by_filename), "duplicate_filenames": duplicates,
        "annotation_sha256": hashlib.sha256(filename.read_bytes()).hexdigest(),
        "class_counts": {name: counts[cid] for cid, name in sorted(categories.items())},
        "absent_classes": [name for cid, name in sorted(categories.items()) if not counts[cid]],
        "square_resized_box_sizes_diagnostic": sizes,
    }
    return data, report


def prepare(root, destination, verify_images=False, limit=None, duplicate_policy="exclude"):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("Output must be outside the original dataset")
    if duplicate_policy not in ("exclude", "keep"):
        raise ValueError("duplicate_policy must be exclude or keep")
    splits, report = {}, {"source": str(root), "test_split": None, "smoke_subset": limit is not None,
                          "duplicate_policy": duplicate_policy}
    prior = destination / "audit.json"
    if prior.exists():
        old = read_json(prior)
        if old["duplicate_policy"] != duplicate_policy or old.get("limit") != limit or old["source"] != str(root):
            raise ValueError("Use a new output directory when changing source, subset size or duplicate policy")
    report["limit"] = limit
    for split in ("train", "val"):
        splits[split], report[split] = audit_split(root, split, verify_images)
    assert splits["train"]["categories"] == splits["val"]["categories"], "Category mapping mismatch"
    overlap = {i["file_name"] for i in splits["train"]["images"]} & {i["file_name"] for i in splits["val"]["images"]}
    assert not overlap, f"Train/val filenames overlap: {sorted(overlap)[:5]}"
    report["filename_overlap"] = 0
    groups = [{i["file_name"].split("_DJI_")[0] for i in splits[s]["images"]} for s in ("train", "val")]
    report["filename_prefix_overlap_not_confirmed_tower_identity"] = sorted(groups[0] & groups[1])
    cats = sorted(splits["train"]["categories"], key=lambda c: c["id"])
    mapping = [{"label": k, "category_id": c["id"], "name": c["name"]} for k, c in enumerate(cats)]
    for split, data in splits.items():
        conflicts = report[split]["duplicate_filenames"]
        excluded_ids = {iid for ids in conflicts.values() for iid in ids} if duplicate_policy == "exclude" else set()
        excluded_annotations = [a for a in data["annotations"] if a["image_id"] in excluded_ids]
        write_json(destination / f"{split}_conflicts.json", {
            "filenames": conflicts, "excluded_image_ids": sorted(excluded_ids),
            "excluded_annotations": excluded_annotations,
        })
        data = dict(data)
        data["images"] = [i for i in data["images"] if i["id"] not in excluded_ids]
        data["annotations"] = [a for a in data["annotations"] if a["image_id"] not in excluded_ids]
        if limit:
            data = dict(data)
            data["images"] = sorted(data["images"], key=lambda i: i["id"])[:limit]
            ids = {i["id"] for i in data["images"]}
            data["annotations"] = [a for a in data["annotations"] if a["image_id"] in ids]
        target = destination / ("valid" if split == "val" else "train")
        target.mkdir(parents=True, exist_ok=True)
        for i in data["images"]:
            link, original = target / i["file_name"], root / split / i["file_name"]
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != original.resolve():
                    raise FileExistsError(f"Conflicting symlink: {link}")
            elif link.exists():
                raise FileExistsError(f"Refusing to replace: {link}")
            else:
                link.symlink_to(original)
        write_json(target / "_annotations.coco.json", data)
        report[split]["prepared_images"] = len(data["images"])
        report[split]["prepared_annotations"] = len(data["annotations"])
        counts = collections.Counter(a["category_id"] for a in data["annotations"])
        report[split]["prepared_class_counts"] = {c["name"]: counts[c["id"]] for c in cats}
    write_json(destination / "class_mapping.json", mapping)
    write_json(destination / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "InsPLAD-det")
    parser.add_argument("--output", default="data/insplad")
    parser.add_argument("--verify-images", action="store_true")
    parser.add_argument("--limit", type=int, help="Separate smoke subset only; not a scientific benchmark")
    parser.add_argument("--duplicate-policy", choices=["exclude", "keep"], default="exclude")
    args = parser.parse_args()
    if args.limit is not None and (args.limit <= 0 or args.output == "data/insplad"):
        parser.error("--limit must be positive and requires a separate --output")
    result = prepare(args.source, path(args.output), args.verify_images, args.limit, args.duplicate_policy)
    print(f"Prepared {path(args.output)}; classes=18; train={result['train']['prepared_images']}; val={result['val']['prepared_images']}")
    print(f"Validation absent classes: {result['val']['absent_classes']}")
