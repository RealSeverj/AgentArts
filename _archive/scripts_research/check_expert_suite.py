"""Audit a proposed expert suite without starting models or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from scripts.dataset_routing import validate_suite_configs, validate_training_route


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "data")
    args = parser.parse_args()
    try:
        configs = [json.loads(path.read_text(encoding="utf-8-sig")) for path in args.config]
        suite = validate_suite_configs(configs)
        routes = [{"config": str(path), "task": config["task"], **validate_training_route(config, args.data_root)}
                  for path, config in zip(args.config, configs)]
        print(json.dumps({"all_checks_passed": True, "training_started": False, "suite": suite, "routes": routes,
                          "limitations": ["Checks task/source routing and one shared split identity. No model accuracy is established.",
                                          "Patient separation is limited by the identities available in source metadata."]},
                         ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Expert suite rejected: {error}\n")


if __name__ == "__main__":
    main()
