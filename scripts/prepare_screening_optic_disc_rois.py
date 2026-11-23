"""Generate Model 5 RFMiD / RFMiD2 image-only optic-disc coordinates with the archived REFUGE U-Net."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.screening_evaluation import write_file, write_json
from training.structural import _tensor, build_model

LOCATOR = "bb25840047eedca5b115c3a81c9f7b8e3e73a8a0729cc6c67b9f55b1c38ef5e7"
POLICY = dict(id="disc_lcc8_square2_v1", locator=LOCATOR, size_hw=[512, 512],
              preprocessing="native RGB, PIL bilinear stretch, original structural normalization",
              precision="fp32, no TF32", labels=["optic_cup", "optic_disc"],
              disc_threshold=0.5, connectivity=8, min_component_pixels=16,
              largest_tie="topmost then leftmost first foreground pixel",
              crop="square at native disc-box center, side ceil(2*max(native box dimensions)); "
                   "clamp size to shorter native axis, translate inside image",
              fallback="whole frame on nonfinite logits, empty/tiny disc, or disc larger than feasible square",
              coordinates="integer native XYXY, right and bottom exclusive")
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def largest_component(mask):
    """Get largest 8-connected foreground bbox without a SciPy dependency."""
    height, width = mask.shape
    stride = width + 2
    ordered = np.flatnonzero(np.pad(mask, 1).ravel()).tolist()
    remaining = set(ordered)
    best = None
    best_area = 0
    offsets = (-stride-1, -stride, -stride+1, -1, 1, stride-1, stride, stride+1)
    for seed in ordered:
        if seed not in remaining:
            continue
        remaining.remove(seed)
        pending = [seed]
        left = width + 1
        top = height + 1
        right = bottom = area = 0
        while pending:
            pos = pending.pop()
            y, x = divmod(pos, stride)
            left, top, right, bottom = min(left, x), min(top, y), max(right, x), max(bottom, y)
            area += 1
            for offset in offsets:
                neighbor = pos + offset
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
        if area > best_area:
            # Undo one-pixel padding; right and bottom are exclusive.
            best_area, best = area, (left-1, top-1, right, bottom)
    return best, best_area


def disc_window(disc_mask, source_size_wh):
    box, area = largest_component(disc_mask)
    details = {"component_pixels": area, "disc_bbox_model_xyxy": box}
    if box is None or area < POLICY["min_component_pixels"]:
        return None, ("empty_disc" if box is None else "tiny_disc"), details
    width, height = source_size_wh
    mh, mw = disc_mask.shape
    x0, y0, x1, y1 = box
    x0, x1 = x0 * width / mw, x1 * width / mw
    y0, y1 = y0 * height / mh, y1 * height / mh
    diameter = max(x1-x0, y1-y0)
    side = min(math.ceil(2 * diameter), width, height)
    if math.ceil(x1) - math.floor(x0) > side or math.ceil(y1) - math.floor(y0) > side:
        return None, "disc_larger_than_feasible_square", details
    left = min(max(0, math.floor((x0 + x1 - side) / 2)), width-side)
    top = min(max(0, math.floor((y0 + y1 - side) / 2)), height-side)
    # Containment also holds for odd side lengths and off-frame centers.
    left = min(max(left, math.ceil(x1)-side), math.floor(x0))
    top = min(max(top, math.ceil(y1)-side), math.floor(y0))
    return [left, top, left+side, top+side], None, details


def load_locator(path, device):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(4*1024**2), b""):
            sha.update(block)
        if sha.hexdigest() != LOCATOR:
            raise ValueError("Archive U-Net weight hash mismatch")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if (cfg["kind"] != "segmentation" or cfg["labels"] != POLICY["labels"]
            or cfg["model"]["img_size"] != 512 or cfg["model"]["name"] != "unet"):
        raise ValueError("Archive U-Net channels or geometry mismatch")
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    gc.collect()
    return model.to(device).requires_grad_(False).eval(), cfg


def read_image(row, root, cfg):
    path = (root / row["image_path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Image path escapes root")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["image_sha256"]:
        raise ValueError(f"Image SHA mismatch: {row['sample_id']}")
    with Image.open(BytesIO(raw)) as image:
        size = image.size
        resized = image.convert("RGB").resize((512,512), Image.Resampling.BILINEAR)
        return size, _tensor(resized, cfg)


def prepare_split(args, split, model, cfg):
    key = f"grouped/{split}"
    raw = (args.release / (key + ".jsonl")).read_bytes()
    identity = json.loads((args.release / "audit.json").read_text(encoding="utf-8-sig"))["manifests"][key]
    if hashlib.sha256(raw).hexdigest() != identity["sha256"]:
        raise ValueError(f"Source manifest hash mismatch: {split}")
    # Do not pass diagnosis, annotations, outcome or mask paths into localization.
    fields = ("sample_id", "dataset", "split", "image_path", "image_sha256")
    rows = [{k: row[k] for k in fields} for row in
            (json.loads(line) for line in raw.splitlines() if line.strip())]
    expected_size = {"train": 1878, "val": 634, "test": 615, "external_test": 848}[split]
    expected_source = "RFMiD2" if split == "external_test" else "RFMiD"
    if (len(rows) != expected_size or len(rows) != identity["records"]
            or len({r["sample_id"] for r in rows}) != len(rows)
            or any(r["split"] != split or r["dataset"] != expected_source for r in rows)):
        raise ValueError(f"Invalid source cohort: {split}")
    output_rows = []
    root = args.data_root.resolve()
    with torch.inference_mode(), ThreadPoolExecutor(args.workers) as pool:
        for start in range(0,len(rows),args.batch):
            group = rows[start:start+args.batch]
            data = list(pool.map(lambda r: read_image(r,root,cfg), group))
            logits = model(torch.stack([v for _,v in data]).to(args.device))
            if logits.shape != (len(group), 2, 512, 512):
                raise ValueError("Unexpected U-Net output shape")
            finite = torch.isfinite(logits).flatten(1).all(1).cpu().tolist()
            masks = logits[:,1].sigmoid().ge(POLICY["disc_threshold"]).cpu().numpy()
            for row,(size,_),mask,valid in zip(group,data,masks,finite):
                if valid:
                    box, reason, details = disc_window(mask,size)
                else:
                    box, reason, details = None, "nonfinite_logits", {}
                output_rows.append(dict(row, source_size_wh=list(size), bbox_xyxy=box, status="ok" if box else "fallback",
                                        fallback_reason=reason, policy_id=POLICY["id"], policy_sha256=POLICY_SHA,
                                        locator_sha256=LOCATOR, **details))
            if len(output_rows) % (args.batch*32) == 0 or len(output_rows) == len(rows):
                print(f"{split} {len(output_rows)}/{len(rows)}",flush=True)
    content = "".join(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n" for row in output_rows)
    target = args.output / f"{split}.jsonl"
    write_file(target,content,lock=True)
    counts = Counter((r["dataset"],r["fallback_reason"] or "ok") for r in output_rows)
    summary = dict(split=split, samples=len(rows), source_manifest_sha256=identity["sha256"],
                   coordinate_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), locator_sha256=LOCATOR,
                   policy_id=POLICY["id"], policy_sha256=POLICY_SHA, torch_version=str(torch.__version__),
                   device=(torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else args.device),
                   by_dataset={src:{k:v for (s,k),v in counts.items() if s==src} for src,_ in sorted(counts)})
    write_json(args.output / f"{split}.meta.json", summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True,type=Path)
    parser.add_argument("--release",required=True,type=Path)
    parser.add_argument("--data-root",required=True,type=Path,help="project root containing data/")
    parser.add_argument("--output",type=Path,default=Path("data/prepared/screening_optic_disc_rois_v1/grouped"))
    parser.add_argument("--splits",nargs="+",choices=("train","val","test","external_test"),required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    model,cfg=load_locator(args.checkpoint,args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/"policy.json",dict(policy=POLICY,policy_sha256=POLICY_SHA),lock=True)
    for split in args.splits:
        prepare_split(args,split,model,cfg)


if __name__=="__main__":
    main()
