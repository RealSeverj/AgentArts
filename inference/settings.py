"""Explicit, portable settings. Secrets are read only from the environment."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    registry: Path = ROOT / "configs/inference/models.json"
    weights_root: Path = ROOT / "weights"
    research_root: Path | None = None
    api_key: str = field(default="", repr=False)
    device: str = "auto"
    precision: str = "auto"
    cpu_threads: int = 4
    max_concurrent_requests: int = 1
    max_body_bytes: int = 16 * 1024
    docs_enabled: bool = False
    visual_origin: str | None = None
    workflow_db: Path | None = None
    routing: dict = field(default_factory=lambda: {
        "mode": "all", "policy_id": "development-all-v1", "validation_passed": False,
    })
    image_policy: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda", "cuda:0", "cuda:1"}:
            raise ValueError("Unsupported device")
        if self.precision not in {"auto", "fp32", "fp16"}:
            raise ValueError("Unsupported precision")
        for name, minimum, maximum in (("cpu_threads", 1, 64), ("max_concurrent_requests", 1, 4),
                                       ("max_body_bytes", 1024, 65536)):
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid {name}")
        if type(self.docs_enabled) is not bool or not isinstance(self.image_policy, dict):
            raise ValueError("Invalid service flags")
        if self.api_key and len(self.api_key.encode("utf-8")) < 32:
            raise ValueError("RETINA_API_KEY must contain at least 32 bytes")

    @classmethod
    def from_env(cls):
        config_path = Path(os.getenv("RETINA_SERVICE_CONFIG", ROOT / "configs/inference/service.json")).resolve()
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        allowed = {"registry", "weights_root", "research_root", "workflow_db", "device", "precision", "cpu_threads",
                   "max_concurrent_requests", "max_body_bytes", "docs_enabled", "visual_origin", "routing", "image_policy"}
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError("Unknown service configuration fields")
        for key in ("registry", "weights_root", "research_root", "workflow_db"):
            candidate = os.getenv("RETINA_" + key.upper(), config.get(key))
            if candidate is not None:
                path = Path(candidate).expanduser()
                config[key] = (path if path.is_absolute() else config_path.parent / path).resolve()
        for key in ("device", "precision"):
            config[key] = os.getenv("RETINA_" + key.upper(), config.get(key, "auto"))
        return cls(api_key=os.getenv("RETINA_API_KEY", ""), **config)
