"""Launch reproducible RF-DETR training with the official trainer."""
import argparse
import importlib.metadata
import os
import platform
import subprocess
import time
from pathlib import Path

from common import create_model, load_config, path, read_json, stamp_checkpoint_metadata, write_json

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/small_512.yaml")
    parser.add_argument("--devices", type=int, help="GPU count visible via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="Per-GPU microbatch")
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--output")
    parser.add_argument("--dataset", help="Prepared dataset directory, e.g. data/smoke for a short test")
    parser.add_argument("--resume", help="Full last.ckpt for optimizer/scheduler/EMA resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    mc, tc = config["model"], config["train"]
    for key in ("devices", "seed", "epochs", "batch_size", "grad_accum_steps"):
        value = getattr(args, key)
        if value is not None:
            tc[key] = value
    if args.output:
        tc["output_dir"] = str(path(args.output))
    if args.dataset:
        tc["dataset_dir"] = str(path(args.dataset))
    if args.resume:
        resume = path(args.resume)
        if not resume.is_file() or resume.suffix != ".ckpt":
            raise ValueError("--resume requires an existing full Lightning .ckpt")
        tc["resume"] = str(resume)
    dataset = Path(tc["dataset_dir"])
    mapping = read_json(dataset / "class_mapping.json")
    tc["class_names"] = [c["name"] for c in mapping]
    from rfdetr.config import TrainConfig, RFDETRSmallConfig, RFDETRMediumConfig
    TrainConfig(**{k: v for k, v in tc.items() if k != "resolution"})
    model_fields = dict(mc)
    variant = model_fields.pop("variant")
    model_config = {"small": RFDETRSmallConfig, "medium": RFDETRMediumConfig}[variant](**model_fields)
    resolution = tc.get("resolution", model_config.resolution)
    if resolution % (model_config.patch_size * model_config.num_windows):
        raise ValueError("Resolution must be divisible by patch_size * num_windows")
    effective = tc["batch_size"] * tc["grad_accum_steps"] * tc["devices"]
    print(f"RF-DETR-{variant}, resolution={resolution}, classes={len(mapping)}, effective batch={effective}", flush=True)
    if args.dry_run:
        print(config)
        return
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < tc["devices"]:
        raise RuntimeError("Requested CUDA devices unavailable")
    output = Path(tc["output_dir"])
    rank_zero = os.environ.get("LOCAL_RANK", "0") == "0"
    if rank_zero:
        if output.exists() and any(output.iterdir()) and not args.resume:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output = output.with_name(f"{output.name}_{timestamp}")
            tc["output_dir"] = str(output)
            config["train"]["output_dir"] = str(output)
            print(f"Output directory already exists; starting a new run in: {output}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        manifest = {
            "config": config, "effective_batch_size": effective, "python": platform.python_version(),
            "torch": torch.__version__, "rfdetr": importlib.metadata.version("rfdetr"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "dataset_audit": read_json(dataset / "audit.json"),
            "class_mapping": mapping,
        }
        name = f"resume-{int(time.time())}.json" if args.resume else "run_manifest.json"
        write_json(output / name, manifest)
        architecture = model_config.model_dump()
        architecture.update(resolution=resolution, positional_encoding_size=resolution // model_config.patch_size,
                            num_classes=len(mapping))
        write_json(output / "model_config.json", architecture)
        (output / "environment.txt").write_text(subprocess.check_output([os.sys.executable, "-m", "pip", "freeze"], text=True))
    # Use the official resolution override so pretrained positional embeddings are interpolated.
    # Seed BEFORE construction: adapting the pretrained classification head consumes RNG.
    from pytorch_lightning import seed_everything
    seed_everything(tc["seed"], workers=True)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = create_model(mc)
    model.train(**tc)
    if rank_zero:
        stamp_checkpoint_metadata(output, model.model_config.model_dump())
    # Each DDP process writes its own device peak after Lightning selects its local device.
    device = torch.cuda.current_device()
    write_json(output / f"runtime-rank{os.environ.get('LOCAL_RANK', '0')}-{int(time.time())}.json", {
        "elapsed_seconds": time.perf_counter() - started,
        "device": device, "device_name": torch.cuda.get_device_name(device),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        "note": "PyTorch allocator peak in this process; excludes other processes/driver allocations.",
    })


if __name__ == "__main__":
    main()
