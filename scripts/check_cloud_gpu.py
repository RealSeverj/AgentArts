"""Check the real configured classifier on synthetic inputs, without dataset I/O.

Examples:
  python scripts/check_cloud_gpu.py --config configs/dr_grading.json --prefetch-only
  torchrun --standalone --nproc_per_node=2 scripts/check_cloud_gpu.py \
    --config configs/dr_grading.json --output /mnt/retina/gpu_check.json

This performs actual AdamW updates in head-only and fully unfrozen stages. It
does not train on retinal images, save model weights, or measure model accuracy.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
from datetime import datetime, timezone, timedelta
import gc
import json
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F

from training.classification import build_model
from training.train import set_trainable


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def write_report(path, report):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def autocast_context(device, precision):
    if device.type == "cpu" or precision == "none":
        return nullcontext()
    return torch.autocast("cuda", dtype=torch.float16 if precision == "fp16" else torch.bfloat16)


def synthetic_loss(model, images, targets, config):
    logits = model(images)
    loss_config = config.get("loss", {})
    if config["kind"] == "multiclass":
        weights = loss_config.get("class_weights")
        weights = torch.as_tensor(weights, dtype=torch.float32, device=images.device) if weights is not None else None
        return F.cross_entropy(logits.float(), targets, weight=weights,
                               label_smoothing=float(loss_config.get("label_smoothing", 0.0)), reduction="none").mean()
    weights = loss_config.get("pos_weight")
    weights = torch.as_tensor(weights, dtype=torch.float32, device=images.device) if weights is not None else None
    return F.binary_cross_entropy_with_logits(logits.float(), targets, pos_weight=weights)


def check_stage(raw, wrapped, optimizer, scaler, config, epoch, images, targets,
                device, precision, steps, accumulation, rank, world):
    if wrapped is not None:
        # A reducer only registers the parameters trainable at construction.
        # Dispose of the old reducer before unfreezing and constructing another.
        del wrapped
        gc.collect()
    stage = set_trainable(raw, config, epoch)
    if world > 1:
        dist.barrier()
        wrapped = DDP(raw, device_ids=[device.index] if device.type == "cuda" else None,
                      broadcast_buffers=False)
    else:
        wrapped = raw
    wrapped.train()
    if stage == "head_only":
        raw.backbone.eval()
    optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    wall_times, cuda_times, losses = [], [], []
    optimizer_updates, overflow_retries = 0, 0
    while optimizer_updates < steps:
        step = optimizer_updates
        synchronize(device)
        begin = time.perf_counter()
        start_event, end_event = None, None
        if device.type == "cuda":
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
        accumulated_loss = 0.0
        for micro in range(accumulation):
            boundary = micro + 1 == accumulation
            context = wrapped.no_sync() if world > 1 and not boundary else nullcontext()
            with context:
                with autocast_context(device, precision):
                    loss = synthetic_loss(wrapped, images, targets, config)
                finite = torch.isfinite(loss).to(dtype=torch.int32)
                if world > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError(f"Nonfinite synthetic loss: rank={rank}, stage={stage}, step={step + 1}")
                scaler.scale(loss / accumulation).backward()
            accumulated_loss += float(loss.detach()) / accumulation
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), float(config["training"].get("clip_grad_norm", 1.0)))
        finite = torch.isfinite(norm).to(dtype=torch.int32)
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            if scaler.is_enabled() and overflow_retries < 8:
                # Initial loss-scale calibration is normal FP16 behavior.
                # Retry instead of counting a skipped update as a success.
                old_scale = scaler.get_scale()
                scaler.update(new_scale=old_scale / 2)
                optimizer.zero_grad(set_to_none=True)
                overflow_retries += 1
                print(json.dumps({"event": "fp16_scale_calibration_retry", "rank": rank,
                                  "stage": stage, "old_scale": old_scale,
                                  "new_scale": scaler.get_scale(), "retry": overflow_retries}), flush=True)
                continue
            raise FloatingPointError(f"Nonfinite gradients: rank={rank}, stage={stage}, step={step + 1}")
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # A reduced scaler value means the optimizer update was skipped.
        if scaler.get_scale() < old_scale:
            raise FloatingPointError(f"FP16 optimizer step skipped: rank={rank}, stage={stage}, step={step + 1}")
        optimizer_updates += 1
        optimizer.zero_grad(set_to_none=True)
        if end_event is not None:
            end_event.record()
        synchronize(device)
        wall_times.append((time.perf_counter() - begin) * 1000)
        if start_event is not None:
            cuda_times.append(float(start_event.elapsed_time(end_event)))
        losses.append(accumulated_loss)
        print(json.dumps({"event": "synthetic_optimizer_step", "rank": rank, "stage": stage,
                          "step": step + 1, "loss": accumulated_loss, "wall_ms": wall_times[-1]}, ensure_ascii=False), flush=True)
    result = {"stage": stage, "benchmark_epoch_zero_based": epoch,
              "trainable_parameters": sum(p.numel() for p in raw.parameters() if p.requires_grad),
              "total_parameters": sum(p.numel() for p in raw.parameters()),
              "optimizer_updates": optimizer_updates, "fp16_scale_calibration_retries": overflow_retries,
              "microbatches_per_update": accumulation,
              "losses": losses, "optimizer_step_wall_ms": wall_times,
              "mean_optimizer_step_wall_ms": sum(wall_times) / len(wall_times),
              "optimizer_step_cuda_elapsed_ms": cuda_times,
              "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
              "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None}
    return wrapped, result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/dr_grading.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true", help="Offline architecture check; report is marked as not checking pretrained weights")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--steps", type=int, default=2, help="Actual optimizer updates in each of the two stages")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    if config.get("kind") not in {"multiclass", "multilabel"}:
        parser.error("This checker supports configured classification models")
    for name in ("image_size", "batch_size", "grad_accum_steps", "steps"):
        if getattr(args, name) is not None and getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.prefetch_only and (args.no_pretrained or not config["model"].get("pretrained", True)):
        parser.error("--prefetch-only requires model.pretrained=true and no --no-pretrained")
    world, rank, local_rank = (int(os.environ.get(key, default)) for key, default in
                               (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if args.prefetch_only and world != 1:
        parser.error("Prefetch weights with a single process before torchrun")
    original_model = copy.deepcopy(config["model"])
    if args.image_size:
        config["model"]["img_size"] = args.image_size
    if args.no_pretrained:
        config["model"]["pretrained"] = False
        config["model"].pop("checkpoint", None)
    checkpoint = config["model"].get("checkpoint")
    if checkpoint and not Path(checkpoint).is_absolute():
        config["model"]["checkpoint"] = str((PROJECT / checkpoint).resolve())
    if args.prefetch_only:
        begin = time.perf_counter()
        model = build_model(config, pretrained=True)
        report = {"success": True, "mode": "prefetch_only", "model": config["model"],
                  "parameters": sum(p.numel() for p in model.parameters()),
                  "elapsed_seconds": time.perf_counter() - begin,
                  "scope": "Pretrained initialization succeeded; no GPU training or accuracy evaluation was performed"}
        if args.output:
            write_report(args.output, report)
        print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
        return
    use_cuda = args.device == "cuda"
    if use_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; --device cpu is only an explicit architecture/debug check")
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    else:
        torch.set_num_threads(min(4, os.cpu_count() or 1))
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=10))
    try:
        torch.manual_seed(int(config.get("seed", 20261113)) + rank)
        if use_cuda:
            torch.cuda.manual_seed_all(int(config.get("seed", 20261113)) + rank)
        precision = config["training"].get("amp", "fp16") if use_cuda else "none"
        if precision not in {"fp16", "bf16", "none"}:
            raise ValueError("Unsupported training.amp precision")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unavailable on this GPU; V100 should use FP16")
        size = config["model"].get("img_size", 392)
        height, width = (size, size) if isinstance(size, int) else size
        batch_size = args.batch_size or int(config["training"].get("batch_size", 4))
        accumulation = args.grad_accum_steps or int(config["training"].get("grad_accum_steps", 1))
        views = 2 if config["task"] == "odir_patient_multilabel" else 1
        shape = (batch_size, 3, height, width) if views == 1 else (batch_size, views, 3, height, width)
        raw = build_model(config, pretrained=not args.no_pretrained).to(device)
        images = torch.randn(shape, dtype=torch.float32, device=device)
        if config["kind"] == "multiclass":
            targets = torch.randint(int(config["num_classes"]), (batch_size,), device=device)
        else:
            targets = torch.randint(2, (batch_size, len(config["labels"])), device=device).float()
        optimizer = torch.optim.AdamW(raw.parameters(), lr=float(config["training"].get("lr", 1e-4)),
                                      weight_decay=float(config["training"].get("weight_decay", 0.05)))
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and precision == "fp16")
        benchmark_config = copy.deepcopy(config)
        benchmark_config["training"].update(freeze_backbone_epochs=1, full_finetune_epoch=1, unfreeze_last_n_blocks=0)
        stages, wrapped = [], None
        for epoch in (0, 1):
            # Remove the caller's reference too: keeping it would retain the
            # old reducer while the callee creates a new DDP wrapper.
            if wrapped is not None:
                del wrapped
                wrapped = None
                gc.collect()
            wrapped, result = check_stage(raw, wrapped, optimizer, scaler, benchmark_config, epoch,
                                          images, targets, device, precision, args.steps, accumulation, rank, world)
            stages.append(result)
        local = {"rank": rank, "local_rank": local_rank, "device": str(device),
                 "gpu_name": torch.cuda.get_device_name(device) if use_cuda else None,
                 "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3 if use_cuda else None,
                 "stages": stages}
        ranks = [local]
        if world > 1:
            ranks = [None] * world
            dist.all_gather_object(ranks, local)
        if rank == 0:
            import timm
            report = {"success": True, "utc": datetime.now(timezone.utc).isoformat(),
                      "mode": "synthetic_training_check", "accuracy_evaluated": False,
                      "scope": "Synthetic resident tensors; forward/backward/AdamW and DDP communication, no retinal data I/O. Timings include first-step warm-up and lazy optimizer-state allocation; they are not a throughput benchmark for the dataset.",
                      "torch": torch.__version__, "timm": timm.__version__, "torch_cuda": torch.version.cuda,
                      "precision": precision, "world_size": world, "batch_size_per_rank": batch_size,
                      "grad_accum_steps": accumulation, "effective_batch_size": batch_size * accumulation * world,
                      "input_shape_per_rank": list(shape), "configured_model": original_model,
                      "checked_model": config["model"], "pretrained_initialization_checked": not args.no_pretrained and bool(config["model"].get("pretrained", True)),
                      "schedule": "Explicit head_only then full peak-memory check; training config file was not modified",
                      "timing_unit": "milliseconds per optimizer update including grad_accum_steps microbatches",
                      "ranks": ranks}
            if args.output:
                write_report(args.output, report)
            print(json.dumps(report, ensure_ascii=False, allow_nan=False), flush=True)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
