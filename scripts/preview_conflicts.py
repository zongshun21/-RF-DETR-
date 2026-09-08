"""Compare annotation sets for duplicate filenames; never infer which set is correct."""
import argparse
import collections
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2] / "InsPLAD-det")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "reports/conflicts")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    data = json.loads((args.source / "annotations/instances_train.json").read_text())
    files, anns = collections.defaultdict(list), collections.defaultdict(list)
    names = {c["id"]: c["name"] for c in data["categories"]}
    for record in data["images"]:
        files[record["file_name"]].append(record)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    args.output.mkdir(parents=True, exist_ok=True)
    conflicts = [(name, records) for name, records in sorted(files.items()) if len(records) > 1]
    for name, records in conflicts[:args.limit]:
        panels = []
        for record in records:
            with Image.open(args.source / "train" / name) as source:
                panel = source.convert("RGB")
            draw = ImageDraw.Draw(panel)
            for a in anns[record["id"]]:
                x, y, w, h = a["bbox"]
                draw.rectangle((x, y, x+w, y+h), outline="red", width=4)
                draw.text((x, y), names[a["category_id"]], fill="yellow", stroke_width=1, stroke_fill="black")
            panel = ImageOps.contain(panel, (960, 540))
            framed = Image.new("RGB", (960, 580), "white")
            framed.paste(panel, (0, 40))
            ImageDraw.Draw(framed).text((10, 10), f"{name} | image_id={record['id']}", fill="black")
            panels.append(framed)
        canvas = Image.new("RGB", (960*len(panels), 580), "white")
        for index, panel in enumerate(panels):
            canvas.paste(panel, (960*index, 0))
        canvas.save(args.output / name)


if __name__ == "__main__":
    main()
