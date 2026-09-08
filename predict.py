"""Render detections on a local image using a trained InsPLAD checkpoint."""
import argparse
from PIL import Image, ImageDraw
from common import ValidationPredictor, load_checkpoint, path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="outputs/prediction.jpg")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    model = load_checkpoint(args.checkpoint, args.device)
    with Image.open(path(args.image)) as source:
        image = source.convert("RGB")
    detections = ValidationPredictor(model, args.precision)(image, threshold=args.threshold)
    names = model.model.class_names
    draw = ImageDraw.Draw(image)
    rows = []
    for box, score, label in zip(detections.xyxy, detections.confidence, detections.class_id):
        box = list(map(float, box))
        label = int(label)
        name = names[label]
        color = (255, 170, 30)
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(0, box[1]-14)), f"{name} {score:.2f}", fill=color, stroke_width=1, stroke_fill="black")
        rows.append({"label": label, "name": name, "score": float(score), "xyxy": box})
    output = path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    write_json(output.with_suffix(".json"), rows)
    print(output)


if __name__ == "__main__":
    main()
