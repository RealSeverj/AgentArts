"""Check the pinned inference runtime without training or loading weights.

Usage: python scripts/check_inference_environment.py --device cpu|cuda|auto
An available CUDA device is only hardware detection, not a successful kernel,
model, memory-capacity or latency benchmark. The JSON states that explicitly.
"""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
import platform
import sys


PINNED_PACKAGES = (
    ("torch", "torch", "2.6.0"),
    ("torchvision", "torchvision", "0.21.0"),
    ("timm", "timm", "1.0.22"),
    ("numpy", "numpy", "1.26.4"),
    ("Pillow", "PIL.Image", "11.3.0"),
)


def check_environment(requested_device="auto"):
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Unsupported device")
    python_ok = (3, 10) <= sys.version_info[:2] < (3, 13)
    report = {
        "python": {"version": platform.python_version(), "required": ">=3.10,<3.13", "ok": python_ok},
        "packages": {},
        "device": {"requested": requested_device, "selected": None, "ok": False},
        "model_weights_loaded": False,
        "model_inference_tested": False,
        "cuda_kernel_tested": False,
        "checks_passed": False,
    }
    for distribution, module_name, expected in PINNED_PACKAGES:
        entry = {"expected": expected, "installed": None, "import_ok": False, "ok": False}
        try:
            entry["installed"] = metadata.version(distribution)
            importlib.import_module(module_name)
            entry["import_ok"] = True
            # CPU/CUDA wheel build suffixes such as +cu118 do not alter the pin.
            entry["ok"] = entry["installed"].split("+", 1)[0] == expected
        except Exception as error:
            # Avoid paths, interpreter internals and arbitrary extension error text.
            entry["error_type"] = type(error).__name__
        report["packages"][distribution] = entry
    if report["packages"]["torch"]["import_ok"]:
        import torch
        try:
            cuda_available = bool(torch.cuda.is_available())
            selected = "cuda" if requested_device == "cuda" or (requested_device == "auto" and cuda_available) else "cpu"
            device = {
                "requested": requested_device, "selected": selected,
                "ok": selected == "cpu" or cuda_available,
                "cuda_available": cuda_available,
                "torch_compiled_cuda": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "compiled_cuda_architectures": torch.cuda.get_arch_list() if cuda_available else [],
                "gpus": [],
                "compatibility_state": "hardware_detection_only_not_model_or_kernel_validation",
            }
            if cuda_available:
                for index in range(torch.cuda.device_count()):
                    properties = torch.cuda.get_device_properties(index)
                    device["gpus"].append({
                        "index": index, "name": properties.name,
                        "compute_capability": list(torch.cuda.get_device_capability(index)),
                        "total_memory_bytes": properties.total_memory,
                    })
            report["device"] = device
        except Exception as error:
            report["device"]["error_type"] = type(error).__name__
    report["checks_passed"] = bool(
        python_ok and all(entry["ok"] for entry in report["packages"].values())
        and report["device"]["ok"]
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    arguments = parser.parse_args()
    report = check_environment(arguments.device)
    print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
    return 0 if report["checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
