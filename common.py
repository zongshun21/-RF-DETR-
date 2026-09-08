"""Shared paths and experiment configuration. Relative paths use the project root."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def path(value):
    p = Path(value).expanduser()
    return p if p.is_absolute() else ROOT / p


def read_json(filename):
    return json.loads(Path(filename).read_text())


def write_json(filename, value):
    p = Path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_config(filename):
    import yaml
    config = yaml.safe_load(path(filename).read_text())
    if set(config) != {"model", "train"}:
        raise ValueError("Config must contain exactly model and train sections")
    config["train"]["dataset_dir"] = str(path(config["train"]["dataset_dir"]))
    config["train"]["output_dir"] = str(path(config["train"]["output_dir"]))
    if config["model"].get("pretrain_weights"):
        config["model"]["pretrain_weights"] = str(path(config["model"]["pretrain_weights"]))
    return config


def create_model(config):
    from rfdetr import RFDETRSmall, RFDETRMedium
    kwargs = dict(config)
    variant = kwargs.pop("variant")
    return {"small": RFDETRSmall, "medium": RFDETRMedium}[variant](**kwargs)


def load_checkpoint(checkpoint, device="cuda"):
    from rfdetr import RFDETR
    from rfdetr.utilities.io import _safe_torch_load
    import torch
    checkpoint = path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    saved = _safe_torch_load(str(checkpoint), trust=False)
    config = saved.get("model_config")
    if config is None and (checkpoint.parent / "training_config.json").is_file():
        config = read_json(checkpoint.parent / "training_config.json")["model_config"]
    if config is None and (checkpoint.parent / "model_config.json").is_file():
        config = read_json(checkpoint.parent / "model_config.json")
    if config is not None and not isinstance(config, dict):
        config = vars(config)
    if not config or "resolution" not in config or "positional_encoding_size" not in config:
        raise ValueError("Checkpoint lacks geometry metadata; use this project's checkpoint_best_total.pth")
    # 1.10's stripped best .pth can omit architecture metadata entirely.
    # Without these explicit kwargs a 960 checkpoint is silently interpolated back to 512.
    model = RFDETR.from_checkpoint(
        str(checkpoint), device=device, resolution=config["resolution"],
        positional_encoding_size=config["positional_encoding_size"],
    )
    if "model" in saved:
        live = model.model.model.state_dict()
        changed = [name for name, tensor in saved["model"].items()
                   if name not in live or tensor.shape != live[name].shape or not torch.equal(tensor, live[name])]
        if changed:
            raise RuntimeError(f"Checkpoint was changed while loading; inspect architecture compatibility: {changed[:5]}")
    return model


def stamp_checkpoint_metadata(output, model_config):
    """Make our locally trained lightweight checkpoints self-contained, preserving tensors."""
    import torch
    from rfdetr.utilities.io import _safe_torch_load
    output = Path(output)
    for filename in [*output.glob("checkpoint_best_*.pth"), *output.glob("last_ema.pth")]:
        checkpoint = _safe_torch_load(str(filename), trust=False)
        checkpoint["model_config"] = model_config
        temporary = filename.with_suffix(".pth.tmp")
        try:
            torch.save(checkpoint, temporary)
            temporary.replace(filename)
        finally:
            temporary.unlink(missing_ok=True)


def foreground_detections(detections, num_classes):
    """RF-DETR 1.10 predict() can expose the extra background logit at index C."""
    labels = detections.class_id
    if labels is None or ((labels < 0) | (labels > num_classes)).any():
        raise ValueError("Predicted class ID outside the foreground/background label space")
    return detections[labels < num_classes]


class ValidationPredictor:
    """Inference with the exact pinned RF-DETR torchvision validation preprocessing.

    Upstream 1.10 predict() uses antialias=False; its default validation loader uses
    antialias=True. Reuse the actual validation transform instead of mixing these protocols.
    """

    def __init__(self, model, precision="bf16"):
        import torch
        from rfdetr.datasets.coco import make_coco_transforms_square_div_64
        self.context = model.model
        self.device = torch.device(self.context.device)
        self.core = self.context.model.to(self.device).eval()
        self.precision = precision if self.device.type == "cuda" else "fp32"
        self.num_classes = len(self.context.class_names)
        self.transform = make_coco_transforms_square_div_64(
            "val", model.model_config.resolution, patch_size=model.model_config.patch_size,
            num_windows=model.model_config.num_windows,
        )

    def __call__(self, image, threshold=0.0):
        import torch
        import supervision as sv
        tensor, _ = self.transform(image.convert("RGB"), None)
        size = torch.tensor([[image.height, image.width]], device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.precision == "bf16"
        ):
            raw = self.core(tensor.unsqueeze(0).to(self.device))
            result = self.context.postprocess(raw, size)[0]
        detections = sv.Detections(
            xyxy=result["boxes"].float().cpu().numpy(),
            confidence=result["scores"].float().cpu().numpy(),
            class_id=result["labels"].cpu().numpy(),
        )
        detections = foreground_detections(detections, self.num_classes)
        return detections[detections.confidence >= threshold]
