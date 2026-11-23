"""Smoke-check real image reading, pairing, unknown labels and PNG-only masks.

Only temporary fixtures are created (and automatically removed). The optional
PyTorch check runs in a subprocess so an incompatible native dependency does not
hide the PIL reader result. No packages are installed or upgraded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image

from manifest_dataset import ManifestDataset, ResizeToTensor, classification_collate, load_png_mask


def two_image_paths(manifest: Path) -> list[str]:
    found = []
    with manifest.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            paths = record.get("image_paths") or ([record["image_path"]] if record.get("image_path") else [])
            for path in paths:
                if path not in found:
                    found.append(path)
                if len(found) == 2:
                    return found
    if found:
        return found * 2
    raise ValueError("Manifest contains no image paths")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def torch_worker(manifest: Path, data_root: Path) -> dict:
    try:
        import torch
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    paths = two_image_paths(manifest)
    with tempfile.TemporaryDirectory(prefix="eye_manifest_torch_") as temporary:
        fixture = Path(temporary) / "fixture.jsonl"
        common = {"sample_id": "smoke", "dataset": "constructed_smoke_only", "task": "classification",
                  "split": "test", "patient_id": None, "labels": {"dr_grade": 2, "unknown": None},
                  "label_mask": {"dr_grade": True, "unknown": False}, "annotations": {}}
        write_jsonl(fixture, [{**common, "image_path": paths[0], "image_paths": [paths[0]]},
                             {**common, "sample_id": "smoke2", "image_path": paths[1], "image_paths": [paths[1]]}])
        dataset = ManifestDataset(fixture, data_root=data_root, image_transform=ResizeToTensor(64))
        samples = [dataset[0], dataset[1]]
        batch = classification_collate(samples, label_dtypes={"dr_grade": "long"})
        assert list(batch["images"].shape) == [2, 3, 64, 64]
        assert batch["labels"]["dr_grade"].dtype == torch.int64
        assert not batch["label_mask"]["unknown"].any().item()
        assert (batch["labels"]["unknown"] == 0).all().item()
        assert samples[0]["labels"]["unknown"] is None
        loader_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2,
                                                            collate_fn=classification_collate, num_workers=0)))
        assert list(loader_batch["images"].shape) == [2, 3, 64, 64]
        paired = {**common, "sample_id": "pair_smoke", "image_path": None, "image_paths": paths}
        write_jsonl(fixture, [paired])
        pair_dataset = ManifestDataset(fixture, data_root=data_root, image_transform=ResizeToTensor((64, 80)))
        pair_batch = classification_collate([pair_dataset[0]])
        assert list(pair_batch["images"].shape) == [1, 2, 3, 64, 80]
        return {"status": "passed", "torch_version": torch.__version__, "single_batch_shape": [2, 3, 64, 64],
                "pair_batch_shape": [1, 2, 3, 64, 80], "unknown_label_mask_preserved": True,
                "pytorch_dataloader_map_style_passed": True}


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path, default=project / "data")
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--torch-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    default_final = data_root / "prepared" / "tasks" / "dr_grading" / "train.jsonl"
    default_stage = data_root / "prepared" / "staging" / "grading_quality" / "candidates.jsonl"
    manifest = (args.manifest or (default_final if default_final.is_file() else default_stage)).resolve()
    if args.torch_worker:
        try:
            print(json.dumps(torch_worker(manifest, data_root), ensure_ascii=True))
        except Exception as exc:
            print(json.dumps({"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True))
            raise SystemExit(1)
        return
    dataset = ManifestDataset(manifest, data_root=data_root)
    assert len(dataset) > 0
    if any(parent.name.casefold() == "data" for parent in manifest.parents):
        inferred = ManifestDataset(manifest)
        assert inferred.data_root == data_root and len(inferred) == len(dataset)
    first = dataset[0]
    assert all(isinstance(image, Image.Image) and image.mode == "RGB" for image in first["images"])
    paths = two_image_paths(manifest)
    with tempfile.TemporaryDirectory(prefix="eye_manifest_pil_") as temporary:
        temporary = Path(temporary)
        fixture = temporary / "fixture.jsonl"
        pair = {"sample_id": "constructed_pair_smoke_only", "dataset": "smoke_only", "task": "pair",
                "split": "test", "image_path": None, "image_paths": paths, "patient_id": None,
                "labels": {"known": 1, "unknown": None}, "label_mask": {"known": True, "unknown": False},
                "annotations": {}, "raw_labels": {"answer_leak": 1}, "diagnostic_keywords": "not an input"}
        write_jsonl(fixture, [pair])
        sample = ManifestDataset(fixture, data_root=data_root)[0]
        assert isinstance(sample["image"], tuple) and len(sample["images"]) == 2
        assert sample["labels"]["unknown"] is None and sample["label_mask"]["unknown"] is False
        assert "raw_labels" not in sample and "diagnostic_keywords" not in sample
        assert "answer_leak" not in repr(sample)
        assert ManifestDataset(fixture, data_root=data_root)[-1]["sample_id"] == pair["sample_id"]
        pair["image_paths"] = ["../outside_data.png"]
        write_jsonl(fixture, [pair])
        try:
            ManifestDataset(fixture, data_root=data_root)[0]
        except ValueError:
            pass
        else:
            raise AssertionError("Path traversal was not rejected")
        mask = Image.new("L", (2, 2))
        mask.putdata([0, 128, 255, 0])
        png = temporary / "mask.png"
        mask.save(png)
        loaded = load_png_mask(png, size=4)
        assert loaded.size == (4, 4) and set(loaded.getdata()) == {0, 128, 255}
        tif = temporary / "unsupported.tif"
        tif.write_bytes(b"II*\x00")
        try:
            load_png_mask(tif)
        except ValueError:
            pass
        else:
            raise AssertionError("Raw TIFF mask was not rejected before Pillow")
    report = {"manifest": str(manifest), "indexed_records": len(dataset), "pil_checks_passed": True,
              "single_real_image_read": True, "constructed_pair_real_images_read": True,
              "unknown_labels_preserved": True, "answer_fields_not_returned": True,
              "path_traversal_rejected": True, "png_mask_values_preserved": True, "raw_tiff_rejected": True}
    if args.skip_torch:
        report["torch_check"] = {"status": "skipped_by_argument"}
    elif importlib.util.find_spec("torch") is None:
        report["torch_check"] = {"status": "unavailable", "reason": "PyTorch is not installed; default PIL reader remains usable"}
    else:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--torch-worker",
                                    "--manifest", str(manifest), "--data-root", str(data_root)],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
        try:
            report["torch_check"] = json.loads(output_lines[-1])
        except (IndexError, json.JSONDecodeError):
            report["torch_check"] = {"status": "failed", "exit_code": completed.returncode,
                                     "error": completed.stderr[-1500:]}
        if report["torch_check"]["status"] == "failed":
            print(json.dumps(report, ensure_ascii=True))
            raise SystemExit(1)
    report["all_required_checks_passed"] = True
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
