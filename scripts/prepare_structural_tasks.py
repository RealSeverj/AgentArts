"""Prepare image-reference manifests for structural retinal tasks.

Original datasets are read-only. Derived PNG masks preserve the original image
resolution; PAPILA's polygon masks are explicitly marked as rasterized labels.
The parent integration stage performs global duplicate/split reconciliation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
import openpyxl
import tifffile

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data"
SEED = 20261113
LESIONS = ("MA", "HE", "EX", "SE")


class Builder:
    def __init__(self, data: Path, out: Path):
        self.data = data.resolve()
        self.out = out.resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.records = defaultdict(list)
        self.quarantine = []
        self.notes = []
        self.mask_source_values = defaultdict(Counter)
        self.papila_splits = {}

    def rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.data).as_posix()

    def base(self, dataset, source_split, proposed_split, image, task, *, identity=None, patient=None, eye=None):
        with Image.open(image) as im:
            size = list(im.size)
        sample_id = identity or f"{dataset}:{source_split}:{image.stem}"
        return {"sample_id": sample_id, "record_id": f"{sample_id}:{task}", "dataset": dataset,
                "source_split": source_split, "proposed_split": proposed_split,
                "image_path": self.rel(image), "patient_id": patient, "eye": eye, "task": task,
                "labels": {}, "label_mask": {}, "annotation_path": None, "annotations": {},
                "representation": "original_fundus_image", "image_size": size}

    def add(self, record):
        self.records[record["task"]].append(record)

    def exclude(self, dataset, image, reason, **extra):
        self.quarantine.append({"dataset": dataset, "image_path": self.rel(image), "reason": reason, **extra})

    def png_spec(self, mask, target, image_size):
        with Image.open(mask) as im:
            if im.size != tuple(image_size):
                raise ValueError(f"Mask/image size mismatch: {mask}")
            mode = im.mode
            array = np.asarray(im)
        if array.ndim == 3:
            if array.shape[-1] not in (3, 4):
                raise ValueError("Unexpected PNG mask channels")
            if array.shape[-1] == 4 and not np.all(array[..., 3] == 255):
                raise ValueError("Nonopaque PNG mask alpha")
            if not np.all(array[..., :3] == array[..., :1]):
                raise ValueError("Nonidentical RGB channels in a binary vessel mask")
            array = array[..., 0]
        values = [int(x) for x in np.unique(array)]
        if not set(values) <= {0, 1, 255}:
            raise ValueError(f"Unexpected binary PNG values: {mask}: {values}")
        target.parent.mkdir(parents=True, exist_ok=True)
        binary = (array != 0).astype(np.uint8) * 255
        Image.fromarray(binary).save(target, format="PNG", compress_level=3)
        return {"path": self.rel(target), "encoding": "binary_0_255", "source_path": self.rel(mask),
                "source_mode": mode, "source_values": values,
                "transformation": "replicated_RGB_to_L_then_nonzero_to_255_no_resize"}

    def tiff_spec(self, mask, target, image_size):
        # Current Anaconda/Pillow/libtiff combination exits on PackBits loads.
        # tifffile/imagecodecs decodes these masks correctly without that DLL path.
        array = tifffile.imread(mask)
        channel_handling = "single_channel"
        if array.ndim == 3:
            if array.shape[-1] not in (3, 4):
                raise ValueError(f"Unexpected source mask channels: {mask}")
            if array.shape[-1] == 4:
                alpha = array[..., 3]
                if not np.all(alpha == np.iinfo(array.dtype).max):
                    raise ValueError(f"Nonopaque source-mask alpha: {mask}")
            rgb = array[..., :3]
            if np.all(rgb == rgb[..., :1]):
                array = rgb[..., 0]
                channel_handling = "replicated_RGB_to_single_channel_opaque_alpha_ignored"
            else:
                # IDRiD_81_EX is a valid red-on-black RGBA class mask; an
                # opaque alpha plane must never be mistaken for foreground.
                values = set(int(x) for x in np.unique(rgb))
                if not values <= {0, int(np.iinfo(rgb.dtype).max)}:
                    raise ValueError(f"Nonbinary colored source mask: {mask}")
                array = np.any(rgb != 0, axis=-1).astype(np.uint8)
                channel_handling = "binary_colored_RGB_any_nonzero_foreground_opaque_alpha_ignored"
        if array.ndim != 2 or tuple(reversed(array.shape)) != tuple(image_size):
            raise ValueError(f"Mask/image dimensions differ: {mask}")
        values = [int(x) for x in np.unique(array)]
        if len(values) > 2 or (values and min(values) < 0):
            raise ValueError(f"Unexpected nonbinary source TIFF values: {mask}: {values}")
        self.mask_source_values[mask.parts[-4] if len(mask.parts) > 4 else "TIFF"][str(values)] += 1
        result = (array != 0).astype(np.uint8) * 255
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(result).save(target, format="PNG", compress_level=3)
        return {"path": self.rel(target), "source_path": self.rel(mask), "encoding": "binary_0_255",
                "source_values": values, "transformation": "source_nonzero_to_255_no_resize",
                "source_channel_handling": channel_handling,
                "foreground_pixels": int(np.count_nonzero(result))}

    def prepare_ddr(self):
        root = self.data / "DDR/DDR-dataset"
        for source_split in ("train", "valid", "test"):
            split = "val" if source_split == "valid" else source_split
            seg_root = root / "lesion_segmentation" / source_split
            labels = seg_root / ("segmentation label" if source_split == "valid" else "label")
            images = sorted((seg_root / "image").glob("*.jpg"))
            for image in images:
                sample_id = f"DDR:lesion:{source_split}:{image.stem}"
                record = self.base("DDR", source_split, split, image, "lesion_segmentation", identity=sample_id)
                masks = {}
                for lesion in LESIONS:
                    source = labels / lesion / (image.stem + ".tif")
                    if not source.is_file():
                        raise ValueError(f"DDR source mask is missing: {source}")
                    target = self.out / "derived_masks/DDR" / source_split / lesion / (image.stem + ".png")
                    masks[lesion] = self.tiff_spec(source, target, record["image_size"])
                record["annotations"] = {"masks": masks, "mask_channels": list(LESIONS)}
                record["label_mask"] = {key: True for key in LESIONS}
                self.add(record)
                xml = root / "lesion_detection" / source_split / (image.stem + ".xml")
                tree = ET.parse(xml).getroot()
                xml_size = [int(tree.findtext("size/width")), int(tree.findtext("size/height"))]
                if xml_size != record["image_size"]:
                    raise ValueError(f"DDR XML/image dimensions differ: {xml}")
                counts = Counter()
                for obj in tree.findall("object"):
                    name = obj.findtext("name").upper()
                    if name not in LESIONS:
                        raise ValueError(f"Unknown detection class: {name}")
                    coords = [float(obj.findtext("bndbox/" + k)) for k in ("xmin", "ymin", "xmax", "ymax")]
                    x0, y0, x1, y1 = coords
                    if not (0 <= x0 < x1 <= xml_size[0] and 0 <= y0 < y1 <= xml_size[1]):
                        raise ValueError(f"Invalid source detection box {coords}: {xml}")
                    counts[name] += 1
                detection = self.base("DDR", source_split, split, image, "lesion_detection", identity=sample_id)
                detection["annotation_path"] = self.rel(xml)
                detection["annotations"] = {"xml_path": self.rel(xml), "format": "source_pascal_voc_style_xml",
                                             "class_mapping": {x.lower(): x for x in LESIONS},
                                             "bbox_origin": "source_coordinates_preserved_not_converted",
                                             "object_counts": dict(counts)}
                detection["label_mask"] = {"objects": True}
                self.add(detection)
        assert sum(r["dataset"] == "DDR" for r in self.records["lesion_segmentation"]) == 757
        assert len(self.records["lesion_detection"]) == 757
        self.notes.append("DDR lesions keep their source split proposals; global hash reconciliation must align overlap with DR grading before use.")
        print(json.dumps({"dataset": "DDR", "segmentation": 757, "detection": 757}), flush=True)

    def idrid_bad_hashes(self):
        report = json.loads((self.data / "IDRiD/grading_exact_duplicate_groups.json").read_text(encoding="utf-8"))
        bad = set()
        for group in report["groups"]:
            # Isolate every member, including consistent duplicate groups, rather
            # than choose a new gold label or silently assign a representative.
            for item in group:
                image = self.data / "IDRiD" / item["file"].replace("\\", "/")
                bad.add(hashlib.sha256(image.read_bytes()).hexdigest())
        return bad

    def prepare_idrid(self):
        root = self.data / "IDRiD/extracted"
        bad = self.idrid_bad_hashes()
        seg_root = root / "A. Segmentation"
        names = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates",
                 "SE": "4. Soft Exudates", "optic_disc": "5. Optic Disc"}
        for source_dir, source_split in (("a. Training Set", "train"), ("b. Testing Set", "test")):
            for image in sorted((seg_root / "1. Original Images" / source_dir).glob("*.jpg")):
                if hashlib.sha256(image.read_bytes()).hexdigest() in bad:
                    self.exclude("IDRiD", image, "same_bytes_as_known_ambiguous_or_duplicate_grading_group", task="lesion_segmentation")
                    continue
                record = self.base("IDRiD", source_split, "external_test", image, "lesion_segmentation",
                                   identity=f"IDRiD:segmentation:{source_split}:{image.stem}")
                masks = {}
                for name, folder in names.items():
                    suffix = "OD" if name == "optic_disc" else name
                    source = seg_root / "2. All Segmentation Groundtruths" / source_dir / folder / f"{image.stem}_{suffix}.tif"
                    if source.is_file():
                        target = self.out / "derived_masks/IDRiD" / source_split / name / (image.stem + ".png")
                        masks[name] = self.tiff_spec(source, target, record["image_size"])
                    else:
                        masks[name] = None
                record["annotations"] = {"masks": {k: masks[k] for k in LESIONS},
                                         "missing_mask_semantics": "unknown_not_confirmed_negative"}
                record["label_mask"] = {name: masks[name] is not None for name in LESIONS}
                self.add(record)
                disc = self.base("IDRiD", source_split, "external_test", image, "disc_cup_segmentation",
                                 identity=f"IDRiD:segmentation:{source_split}:{image.stem}")
                disc["annotations"] = {"experts": {"reference": {"optic_disc": masks["optic_disc"], "optic_cup": None}}}
                disc["label_mask"] = {"optic_disc": True, "optic_cup": False}
                self.add(disc)
        loc_root = root / "C. Localization"
        coord_root = loc_root / "2. Groundtruths"
        for source_dir, source_split, word in (("a. Training Set", "train", "Training"), ("b. Testing Set", "test", "Testing")):
            tables = {}
            for kind, folder in (("optic_disc_center", "1. Optic Disc Center Location"), ("fovea_center", "2. Fovea Center Location")):
                table = next((coord_root / folder).glob(f"*{word}*csv"))
                with table.open(encoding="utf-8-sig", newline="") as handle:
                    rows = csv.DictReader(handle)
                    by_id = {}
                    for row in rows:
                        ident = str(row.get("Image No") or "").strip()
                        if not ident:
                            continue
                        if ident in by_id:
                            raise ValueError("Duplicate coordinate ID")
                        by_id[ident] = [float(row["X- Coordinate"]), float(row["Y - Coordinate"])]
                tables[kind] = (table, by_id)
            images = sorted((loc_root / "1. Original Images" / source_dir).glob("*.jpg"))
            for image in images:
                if hashlib.sha256(image.read_bytes()).hexdigest() in bad:
                    self.exclude("IDRiD", image, "same_bytes_as_known_ambiguous_or_duplicate_grading_group", task="landmark_localization")
                    continue
                record = self.base("IDRiD", source_split, "external_test", image, "landmark_localization",
                                   identity=f"IDRiD:grading:{source_split}:{image.stem}")
                points = {}
                for kind, (table, lookup) in tables.items():
                    xy = lookup[image.stem]
                    w, h = record["image_size"]
                    if not (0 <= xy[0] <= w and 0 <= xy[1] <= h):
                        raise ValueError(f"Out-of-image coordinate: {image}: {xy}")
                    points[kind] = {"xy": xy, "source_path": self.rel(table),
                                    "coordinate_system": "source_pixel_xy_origin_not_declared_no_offset_applied"}
                record["annotations"] = {"points": points}
                record["label_mask"] = {kind: True for kind in points}
                self.add(record)
        self.notes.append("IDRiD is held out across all tasks. All known duplicate-group hashes are excluded; missing lesion-mask files remain unknown. Localization images duplicate grading images and are not independent samples.")
        print(json.dumps({"dataset": "IDRiD", "landmarks": len(self.records["landmark_localization"])}), flush=True)

    @staticmethod
    def polygon_mask(points, size):
        """Even-odd fill at integer pixel centers, native float x/y coordinates.

        This follows the author's zero-offset x/y usage, without importing the
        ABI-incompatible local skimage. Boundary pixels are included explicitly.
        """
        w, h = size
        output = Image.new("L", size, 0)
        draw = ImageDraw.Draw(output)
        edges = list(zip(points, points[1:] + points[:1]))
        y_start = max(0, math.ceil(min(p[1] for p in points)))
        y_end = min(h - 1, math.floor(max(p[1] for p in points)))
        for y in range(y_start, y_end + 1):
            crossings = []
            boundary = []
            for (x0, y0), (x1, y1) in edges:
                if y0 == y1:
                    if abs(y - y0) < 1e-7:
                        draw.line((max(0, math.ceil(min(x0, x1))), y, min(w - 1, math.floor(max(x0, x1))), y), fill=255)
                    continue
                if min(y0, y1) <= y <= max(y0, y1):
                    x = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                    if abs(x - round(x)) < 1e-7:
                        boundary.append(round(x))
                    if min(y0, y1) <= y < max(y0, y1):
                        crossings.append(x)
            crossings.sort()
            if len(crossings) % 2:
                raise ValueError("Unexpected polygon scanline parity")
            for left, right in zip(crossings[0::2], crossings[1::2]):
                lo, hi = max(0, math.ceil(left)), min(w - 1, math.floor(right))
                if lo <= hi:
                    draw.line((lo, y, hi, y), fill=255)
            for x in boundary:
                if 0 <= x < w:
                    draw.point((x, y), fill=255)
        if not output.getbbox():
            raise ValueError("Empty polygon rasterization")
        return output

    def prepare_papila(self):
        root = next((self.data / "PAPILA/extracted").iterdir())
        clinical = {}
        patients = defaultdict(list)
        for eye in ("OD", "OS"):
            table = root / "ClinicalData" / f"patient_data_{eye.lower()}.xlsx"
            wb = openpyxl.load_workbook(table, read_only=True, data_only=True)
            for row in wb.active.iter_rows(min_row=4, values_only=True):
                if not row[0]:
                    continue
                pid = str(row[0]).lstrip("#").zfill(3)
                diagnosis = int(row[3])
                if diagnosis not in (0, 1, 2):
                    raise ValueError("Unexpected PAPILA diagnosis")
                stem = f"RET{pid}{eye}"
                clinical[stem] = {"patient_id": f"PAPILA:RET{pid}", "eye": "right" if eye == "OD" else "left",
                                  "diagnosis": diagnosis, "table": table, "xlsx_row_id": str(row[0])}
                patients[pid].append(diagnosis)
            wb.close()
        strata = defaultdict(list)
        for pid, diagnoses in patients.items():
            assert len(diagnoses) == 2
            # Group by the full two-eye label multiset, not a single eye's label.
            strata[tuple(sorted(diagnoses))].append(pid)
        rng = random.Random(SEED)
        split_by_pid = {}
        for key, pids in sorted(strata.items()):
            pids = sorted(pids)
            rng.shuffle(pids)
            count = len(pids)
            # Approximate 70/15/15 within each patient stratum; rare strata may
            # be absent from a held-out subset, which is reported explicitly.
            train = int(count * 0.70)
            val = int(count * 0.15)
            for i, pid in enumerate(pids):
                split_by_pid[pid] = "train" if i < train else "val" if i < train + val else "test"
        self.papila_splits = split_by_pid
        contours = root / "ExpertsSegmentations/Contours"
        for image in sorted((root / "FundusImages").glob("*.jpg")):
            c = clinical[image.stem]
            pid = image.stem[3:6]
            common = dict(identity=f"PAPILA:{image.stem}", patient=c["patient_id"], eye=c["eye"])
            record = self.base("PAPILA", "all", split_by_pid[pid], image, "disc_cup_segmentation", **common)
            experts = {}
            for expert in (1, 2):
                masks = {}
                for kind, short in (("optic_disc", "disc"), ("optic_cup", "cup")):
                    source = contours / f"{image.stem}_{short}_exp{expert}.txt"
                    points = [tuple(map(float, line.split())) for line in source.read_text().splitlines() if line.strip()]
                    if len(points) < 3 or any(len(p) != 2 for p in points):
                        raise ValueError("Invalid source contour")
                    target = self.out / "derived_masks/PAPILA" / f"expert_{expert}" / f"{image.stem}_{short}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    mask = self.polygon_mask(points, tuple(record["image_size"]))
                    mask.save(target, format="PNG", compress_level=3)
                    masks[kind] = {"path": self.rel(target), "source_path": self.rel(source), "encoding": "binary_0_255",
                                   "transformation": "derived_even_odd_polygon_fill_integer_pixel_centers_no_coordinate_offset_no_resize",
                                   "foreground_pixels": mask.histogram()[255]}
                experts[f"expert_{expert}"] = masks
            record["annotations"] = {"experts": experts, "expert_labels_are_separate": True,
                                     "rasterization_reference": self.rel(root / "HelpCode/utils.py")}
            record["label_mask"] = {"optic_disc": True, "optic_cup": True}
            self.add(record)
            glaucoma = self.base("PAPILA", "all", split_by_pid[pid], image, "glaucoma_classification", **common)
            glaucoma["labels"] = {"glaucoma_status": c["diagnosis"],
                                    "glaucoma": c["diagnosis"] if c["diagnosis"] != 2 else None}
            glaucoma["label_mask"] = {"glaucoma_status": True, "glaucoma": c["diagnosis"] != 2}
            glaucoma["annotation_path"] = self.rel(c["table"])
            glaucoma["annotations"] = {"clinical_table": self.rel(c["table"]), "source_row_id": c["xlsx_row_id"],
                                        "diagnosis_encoding": {"0": "healthy", "1": "glaucoma", "2": "suspected"},
                                        "suspected_not_treated_as_healthy": True}
            self.add(glaucoma)
        assert len(clinical) == 488 and len(patients) == 244
        print(json.dumps({"dataset": "PAPILA", "patients": dict(Counter(split_by_pid.values())), "images": 488}), flush=True)

    def prepare_refuge(self):
        root = self.data / "REFUGE/extracted/REFUGE2"
        for split in ("train", "val", "test"):
            for image in sorted((root / split / "images").glob("*.jpg")):
                record = self.base("REFUGE", split, split, image, "disc_cup_segmentation")
                source = next((root / split / "mask").glob(image.stem + ".*"))
                with Image.open(source) as im:
                    if list(im.size) != record["image_size"]:
                        raise ValueError("REFUGE mask size mismatch")
                    array = np.asarray(im)
                values = set(int(x) for x in np.unique(array))
                if not values <= {0, 128, 255}:
                    raise ValueError("Unexpected REFUGE mask encoding")
                masks = {}
                for kind, selected in (("optic_disc", [0, 128]), ("optic_cup", [0])):
                    target = self.out / "derived_masks/REFUGE" / split / f"{image.stem}_{kind}.png"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    binary = np.isin(array, selected).astype(np.uint8) * 255
                    Image.fromarray(binary).save(target, format="PNG", compress_level=3)
                    masks[kind] = {"path": self.rel(target), "source_path": self.rel(source), "encoding": "binary_0_255",
                                   "source_foreground_values": selected, "transformation": "lossless_label_remap_no_resize"}
                record["annotations"] = {"experts": {"reference": masks}, "source_mask_values": {"0": "cup", "128": "disc_annulus", "255": "background"}}
                record["label_mask"] = {"optic_disc": True, "optic_cup": True}
                record["dataset_variant"] = "mixed_incomplete_REFUGE_REFUGE2_public_mirror"
                self.add(record)
                if split == "train":
                    prefix = image.name[0].lower()
                    if prefix not in ("g", "n"):
                        raise ValueError("Unexpected REFUGE train filename")
                    glaucoma = self.base("REFUGE", split, split, image, "glaucoma_classification")
                    value = int(prefix == "g")
                    glaucoma["labels"] = {"glaucoma": value, "glaucoma_status": value}
                    glaucoma["label_mask"] = {"glaucoma": True, "glaucoma_status": True}
                    glaucoma["annotations"] = {"label_source": "source_training_filename_g_n_prefix", "no_independent_label_table": True}
                    self.add(glaucoma)
        self.notes.append("REFUGE is a partial mixed mirror, not full official REFUGE2. Only its train400 have disease labels; val/test400 each contribute segmentation only.")
        print(json.dumps({"dataset": "REFUGE", "disc_cup": 1200, "glaucoma_labels": 400}), flush=True)

    def prepare_fives(self):
        root = next((self.data / "FIVES/extracted").iterdir())
        workbook = root / "Quality Assessment.xlsx"
        wb = openpyxl.load_workbook(workbook, read_only=True, data_only=True)
        for split, sheet_name in (("train", "Train"), ("test", "Test")):
            quality = {}
            for row in wb[sheet_name].iter_rows(min_row=2, values_only=True):
                if not row[0]:
                    continue
                key = f"{int(row[1])}_{str(row[0]).strip()}"
                quality[key] = {name: int(row[index]) for name, index in (("fives_IC", 2), ("fives_Blur", 3), ("fives_LC", 4))}
            images = sorted((root / split / "Original").glob("*.png"))
            for image in images:
                source_class = image.stem.split("_")[-1]
                if source_class not in "ADGN":
                    raise ValueError("Unknown FIVES source class")
                record = self.base("FIVES", split, "external_test", image, "multidisease")
                # Positive source class and normal controls are unambiguous.
                # Do not invent labels outside the dataset's assessed classes.
                labels = {"DR": None, "AMD": None, "MYOPIC_FUNDUS": None, "OPTIC_DISC_CUPPING": None,
                          "abnormality": 0 if source_class == "N" else 1}
                if source_class == "D": labels["DR"] = 1
                if source_class == "A": labels["AMD"] = 1
                if source_class == "N": labels.update(DR=0, AMD=0)
                record["labels"] = labels
                record["label_mask"] = {key: value is not None for key, value in labels.items()}
                record["raw_labels"] = {"fives_disease_class": source_class}
                record["annotations"] = {"label_source": "source_filename_class_suffix", "nonmatching_disease_classes_not_automatically_negative": True}
                self.add(record)
                glaucoma = self.base("FIVES", split, "external_test", image, "glaucoma_classification")
                value = 1 if source_class == "G" else 0 if source_class == "N" else None
                glaucoma["labels"] = {"glaucoma": value, "glaucoma_status": value}
                glaucoma["label_mask"] = {"glaucoma": value is not None, "glaucoma_status": value is not None}
                glaucoma["raw_labels"] = {"fives_disease_class": source_class}
                self.add(glaucoma)
                vessel = self.base("FIVES", split, "external_test", image, "vessel_segmentation")
                source = root / split / "Ground truth" / image.name
                target = self.out / "derived_masks/FIVES" / split / image.name
                spec = self.png_spec(source, target, vessel["image_size"])
                vessel["annotation_path"] = spec["path"]
                vessel["annotations"] = {"masks": {"vessel": spec}}
                vessel["label_mask"] = {"vessel": True}
                self.add(vessel)
                q = self.base("FIVES", split, "external_test", image, "image_quality")
                q["labels"] = quality[image.stem]
                q["label_mask"] = {key: True for key in q["labels"]}
                q["annotation_path"] = self.rel(workbook)
                q["annotations"] = {"sheet": sheet_name, "source_fields": ["IC", "Blur", "LC"],
                                      "semantic_mapping": "source_binary_fields_preserved_not_converted_to_EyeQ_levels"}
                self.add(q)
        wb.close()
        self.notes.append("FIVES entire800 are external_test for every task. Vessel and quality supervision is evaluation-only under this protocol; enable training only under a new explicit split protocol.")
        print(json.dumps({"dataset": "FIVES", "images": 800, "tasks": 4}), flush=True)

    def refresh_fives_vessels(self):
        """Normalize only FIVES masks without rebuilding the other datasets."""
        destination = self.out / "vessel_segmentation.jsonl"
        records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines() if line]
        if len(records) != 800 or any(r["dataset"] != "FIVES" for r in records):
            raise ValueError("Expected the existing 800-record FIVES vessel manifest")
        for record in records:
            previous = record["annotations"]["masks"]["vessel"]
            source = self.data / previous["source_path"]
            target = self.out / "derived_masks/FIVES" / record["source_split"] / source.name
            spec = self.png_spec(source, target, record["image_size"])
            record["annotations"]["masks"]["vessel"] = spec
            record["annotation_path"] = spec["path"]
        temporary = destination.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        temporary.replace(destination)
        report_path = self.out / "verification_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["fives_vessel_mask_normalization"] = {"records": 800, "mode": "L", "values": [0, 255],
                                                    "native_resolution": [2048, 2048], "source_RGB_preserved": True}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"FIVES_vessel_masks_normalized": 800, "manifest": str(destination)}), flush=True)

    def save(self):
        counts = {}
        total = 0
        for task, records in sorted(self.records.items()):
            ids = [r["record_id"] for r in records]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate record IDs in {task}")
            records.sort(key=lambda r: (r["dataset"], r["source_split"], r["sample_id"]))
            with (self.out / f"{task}.jsonl").open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            counts[task] = {"records": len(records), "by_dataset": dict(Counter(r["dataset"] for r in records)),
                            "by_proposed_split": dict(Counter(r["proposed_split"] for r in records))}
            total += len(records)
        with (self.out / "quarantine.jsonl").open("w", encoding="utf-8") as handle:
            for record in self.quarantine:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        report = {"candidate_generation_checks_passed": True, "not_final_until_global_split_reconciliation": True,
                  "seed": SEED, "tasks": counts, "total_task_records": total,
                  "quarantine_records": len(self.quarantine), "papila_patient_split_counts": dict(Counter(self.papila_splits.values())),
                  "papila_patient_split": self.papila_splits, "notes": self.notes,
                  "tiff_decode_backend": "tifffile + imagecodecs; original source TIFF untouched",
                  "derived_masks": "binary PNG at native image resolution, original annotation paths retained"}
        (self.out / "verification_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.out / "README.md").write_text(
            "# Structural-task candidate manifests\n\nThese are staging candidates, not the final train/test split. The central pipeline must reconcile image hashes, duplicate groups, patient identities, and all tasks before use.\n\n"
            "Image and annotation paths are relative to the project data directory. Original source images/labels were not changed or copied. Only small derived PNG masks were written here. Missing masks are null/unknown rather than negative.\n\n"
            "PAPILA uses patient-level, seed20261113, approximately70/15/15 splitting within each two-eye diagnosis stratum. Two expert annotations remain separate. Polygon PNGs are derived using an integer-pixel-center even-odd scanline rule with the original float x/y points, no coordinate offset and no resize, following the author's coordinate usage. They are not claimed to be byte-identical to skimage's rasterization.\n\n"
            "DDR and IDRiD TIFF masks were decoded with tifffile/imagecodecs and losslessly converted from source nonzero foreground to0/255 PNG. Local Pillow crashes on source PackBits TIFF decoding; do not use that path. REFUGE masks were mapped to binary disc(union of cup and annulus) and cup masks, native resolution.\n\n"
            "IDRiD is external_test across every task; known ambiguous/duplicate grading hashes are quarantined including equivalent task copies. Landmark x/y coordinates are retained without inventing an undocumented pixel-origin offset.\n\n"
            "REFUGE is an incomplete mixed mirror:1200 segmentation samples, onlytrain400 with g/n disease labels. FIVES800 stay entirely external_test, including vessel masks and quality attributes. FIVES nonmatching disease categories are not automatically treated as negatives; its quality flags are not mapped to EyeQ levels.\n\n"
            "The suspicious PAPILA category remains class2 in the three-way task and unknown in binary glaucoma classification. See JSON label_mask for valid supervision and verification_report.json for actual counts.\n",
            encoding="utf-8")
        print(json.dumps({"saved": str(self.out), "tasks": counts, "quarantine": len(self.quarantine)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fives-vessels-only", action="store_true")
    args = parser.parse_args()
    output = args.out or args.data_root / "prepared/staging/structural_tasks"
    builder = Builder(args.data_root, output)
    if args.fives_vessels_only:
        builder.refresh_fives_vessels()
        return
    builder.prepare_ddr()
    builder.prepare_idrid()
    builder.prepare_papila()
    builder.prepare_refuge()
    builder.prepare_fives()
    builder.save()


if __name__ == "__main__":
    main()
