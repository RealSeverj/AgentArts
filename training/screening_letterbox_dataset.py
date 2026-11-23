"""Read audited deterministic geometry while retaining online stochastic training augmentation."""
import io
import json

from PIL import Image

from training import screening_letterbox_cache as cache
from training.classification import ClassificationTransform
from training.screening_training import ScreeningDataset


class CachedGeometryDataset(ScreeningDataset):
    def __init__(self,args,audit,task,split):
        super().__init__(args,audit,task,split)
        self.geometry_cache=args.letterbox_cache_dir
        self.policy=cache.recipe_sha()
        raw=(self.geometry_cache/(split+".json")).read_bytes()
        spec=json.loads(raw)
        if (spec["split"]!=split or spec["source_manifest_sha256"]!=self.manifest_sha
                or spec["recipe_sha256"]!=self.policy or spec["recipe"]!=cache.recipe()
                or len(spec["records"])!=len(self.rows)):
            raise ValueError(f"{split} cache recipe, version, source or record count differs from the run")
        self.geometry_records=spec["records"]
        for cached,source in zip(self.geometry_records,self.rows):
            if cached["sample_id"]!=source["sample_id"] or cached["image_sha256"]!=source["image_sha256"]:
                raise ValueError("Geometry cache sample order or source image fingerprint differs")
        # The original ClassificationTransform has an explicit contract for
        # already-preprocessed RGB: only the deterministic geometry is bypassed.
        from training import screening_training as baseline
        self.transform=ClassificationTransform(baseline.TRANSFORM_CONFIG,training=(split=="train"),input_preprocessed=True)

    def source_rgb(self,index):
        rec=self.geometry_records[index]
        raw=cache.image_path(self.geometry_cache,self.policy,rec["image_sha256"]).read_bytes()
        if cache.sha(raw)!=rec["png_sha256"]:
            raise ValueError(f"Corrupt geometry artifact: {rec['sample_id']}")
        with Image.open(io.BytesIO(raw)) as image:
            if image.mode!="RGB" or image.size!=(392,392):
                raise ValueError(f"Unexpected geometry cache image type or dimensions: {rec['sample_id']}")
            return image.copy()
