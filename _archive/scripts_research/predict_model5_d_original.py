"""Score original fundus photographs with archived research candidate D."""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import timm
import torch
from PIL import Image
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from training import fundus_preprocessing as fundus
from training import screening_letterbox_cache as geometry
from training.classification import ClassificationTransform
from training.screening_optimization import GROUP_WIDTHS, LABELS, MODELS
from training.screening_training import ScreeningModel, sha256
from training.train import amp_context


DEFAULT_RUN = PROJECT / "02_模型训练与评估/07_模型5云端优化结果_20261122/geometry_large"
VAL_MANIFEST = PROJECT / "data/prepared/screening_v1/grouped/val.jsonl"


def read(path):
    return json.loads(path.read_bytes())


def require(ok, message):
    if not ok:
        raise ValueError(message)


class LargeWholeImageModel(ScreeningModel):
    """Identical eval forward and state keys to whole-image candidate D."""
    def __init__(self):
        nn.Module.__init__(self)
        self.backbone = timm.create_model(MODELS["vitl"][0], pretrained=False, img_size=392, num_classes=0)
        self.dropout = nn.Dropout(0.1)
        self.heads = nn.ModuleList(nn.Linear(self.backbone.num_features, width) for width in GROUP_WIDTHS)


class Model5DResearchPredictor:
    def __init__(self, run, device=None):
        self.run = Path(run)
        checkpoint = self.run / "best.pt"
        self.digest = sha256(checkpoint)
        done = read(self.run / "complete.json")
        self.lock = read(self.run / "thresholds.json")
        require(self.digest == done["checkpoint_sha256"] == self.lock["checkpoint_sha256"],
                "Final weights, completion receipt and validation thresholds must share the same fingerprint")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        identity = state["identity"]
        require(identity["candidate"] == "vitl_global_letterbox" and identity["image_size"] == 392
                and identity["backbone"] == MODELS["vitl"][0] and identity["pretrained_sha256"] == MODELS["vitl"][1]
                and identity["input_mode"] == "global_fundus_crop_letterbox_v1" and identity["locator_sha256"] is None
                and identity["geometry_recipe_sha256"] == geometry.recipe_sha(),
                "Expected the original candidate D input recipe and backbone")
        require(done["best_epoch"] == state["epoch"] and state["labels"] == self.lock["labels"] == LABELS,
                "Different best epoch or output order")
        self.config = identity["transform"]
        self.transform = ClassificationTransform(self.config, training=False, input_preprocessed=True)
        self.model = LargeWholeImageModel()
        self.model.load_state_dict(state["model"], strict=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = state["settings"]["precision"] if self.device.type == "cuda" else "none"
        self.model.to(self.device).requires_grad_(False).eval()

    @torch.inference_mode()
    def score(self, path):
        raw = Path(path).read_bytes()
        with Image.open(io.BytesIO(raw)) as image:
            original_size = list(image.size)
            view = fundus.apply_preprocessing(image, 392, self.config["preprocessing"])
        tensor = self.transform(view).unsqueeze(0).to(self.device)
        with amp_context(self.device, self.precision):
            logits = self.model(tensor).float()[0]
        require(logits.shape == (11,) and torch.isfinite(logits).all().item(), "Invalid logits")
        scores = logits.sigmoid().tolist()
        outputs = []
        for label, score in zip(LABELS, scores):
            op = self.lock["per_label"][label]
            t = op["threshold"]
            outputs.append(dict(label=label, score=score, threshold=t, flagged=None if t is None else score >= t,
                                validation_working_point_met=op["status"] == "met"))
        return dict(candidate="D", model="vitl_global_letterbox", use="offline_research",
                    checkpoint_sha256=self.digest, image_sha256=hashlib.sha256(raw).hexdigest(),
                    geometry_rgb_sha256=hashlib.sha256(view.tobytes()).hexdigest(), original_size_wh=original_size,
                    device=str(self.device), precision=self.precision, logits=logits.tolist(), outputs=outputs)

    def compare_validation(self, result, manifest):
        meta = read(self.run / "val_predictions.meta.json")
        preds = self.run / "val_predictions.jsonl"
        require(self.digest == meta["checkpoint_sha256"] and self.lock["val_predictions_sha256"] == sha256(preds)
                and self.lock["val_manifest_sha256"] == meta["manifest_sha256"] == sha256(manifest),
                "Validation data or predictions do not match the frozen thresholds")
        data = (json.loads(line) for line in Path(manifest).read_bytes().splitlines() if line.strip())
        matches = [row["sample_id"] for row in data if row["image_sha256"] == result["image_sha256"]]
        require(len(matches)==1, "Expected exactly one matching source image on the frozen validation split")
        old = next((r for r in (json.loads(s) for s in preds.read_bytes().splitlines() if s.strip())
                    if r["sample_id"] == matches[0]), None)
        require(old is not None and old["checkpoint_sha256"]==self.digest, "Matching validation prediction missing")
        delta = [abs(out["score"] - score) for out,score in zip(result["outputs"],old["probabilities"])]
        changed = [r["label"] for r,p in zip(result["outputs"],old["probabilities"])
                   if r["threshold"] is not None and (p>=r["threshold"]) != r["flagged"]]
        return dict(sample_id=matches[0], reference_precision=meta["precision"], max_abs_score_difference=max(delta),
                    max_abs_logit_difference=max(abs(a-b) for a,b in zip(result["logits"],old["logits"])),
                    changed_threshold_decisions=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--device", choices=("cpu","cuda"))
    parser.add_argument("--compare-val", action="store_true", help="Also compare original-image scores to the archived validation row")
    parser.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    args = parser.parse_args()
    model = Model5DResearchPredictor(args.run,args.device)
    for path in args.images:
        result=model.score(path)
        if args.compare_val: result["archived_validation_comparison"]=model.compare_validation(result,args.val_manifest)
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
