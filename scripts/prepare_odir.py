"""Prepare patient-level ODIR pairs; never copy patient targets to each eye."""
from __future__ import annotations
import json
from pathlib import Path
import openpyxl

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"
ROOT = DATA / "ODIR-5K"
OUT = DATA / "prepared" / "staging" / "odir"
CODES = ["N", "D", "G", "C", "A", "H", "M", "O"]
BAD_FILES = {"1706_left.jpg": "author_identified_anterior_segment_photo",
             "1710_right.jpg": "author_identified_anterior_segment_photo",
             "4580_left.jpg": "author_identified_rotated_copy_of_fellow_eye"}


def rows_of(path):
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = workbook.active.iter_rows(values_only=True)
    header = list(next(rows))
    values = {}
    for row in rows:
        if row[0] is None:
            continue
        patient = str(int(row[0]))
        if patient in values:
            raise ValueError(f"Repeated patient ID in {path.name}: {patient}")
        labels = {code: int(row[header.index(code)]) for code in CODES}
        if any(value not in (0, 1) for value in labels.values()):
            raise ValueError("Nonbinary ODIR label")
        values[patient] = {"raw": row, "labels": labels}
    workbook.close()
    return values


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidates, quarantine, comparisons = [], [], []
    seen = set()
    plans = [("Training Set", "training", "train", 3500),
             ("Off-site Test Set", "off_site_test", "test", 500),
             ("On-site Test Set", "on_site_test", "test", 1000)]
    for folder, source_split, proposed, expected in plans:
        english = next((ROOT / folder / "Annotation").glob("*English*.xlsx"))
        chinese = next((ROOT / folder / "Annotation").glob("*Chinese*.xlsx"))
        rows = rows_of(english)
        alternate = rows_of(chinese)
        if len(rows) != expected or set(rows) != set(alternate):
            raise ValueError("Unexpected ODIR annotation patient coverage")
        for patient, item in rows.items():
            if patient in seen:
                raise ValueError("ODIR patient appears in multiple original splits")
            seen.add(patient)
            raw = item["raw"]
            left = ROOT / folder / "Images" / str(raw[3])
            right = ROOT / folder / "Images" / str(raw[4])
            if not left.is_file() or not right.is_file():
                raise ValueError(f"ODIR missing eye image for patient {patient}")
            paths = [left.relative_to(DATA).as_posix(), right.relative_to(DATA).as_posix()]
            record = {"sample_id": f"odir_patient_multilabel:ODIR-5K:{patient}", "dataset": "ODIR-5K",
                      "source_split": source_split, "proposed_split": proposed, "image_path": None,
                      "image_paths": paths, "patient_id": f"ODIR:{patient}", "eye": None,
                      "task": "odir_patient_multilabel", "input_unit": "patient_pair",
                      "labels": item["labels"], "label_mask": {code: True for code in CODES},
                      "annotation_path": english.relative_to(DATA).as_posix(),
                      "representation": "original_color_fundus_patient_pair",
                      "metadata": {"patient_age": raw[1], "patient_sex": raw[2], "eye_order": ["left", "right"]}}
            differences = {code: {"english": item["labels"][code], "chinese": alternate[patient]["labels"][code]}
                           for code in CODES if item["labels"][code] != alternate[patient]["labels"][code]}
            reasons = []
            if differences:
                reasons.append("chinese_english_label_disagreement")
                comparisons.append({"patient_id": f"ODIR:{patient}", "source_split": source_split, "differences": differences})
            for image in (left, right):
                if image.name in BAD_FILES:
                    reasons.append(BAD_FILES[image.name])
            if reasons:
                quarantine.append({**record, "reason": ";".join(reasons), "source_label": item["labels"], "differences": differences})
            else:
                candidates.append(record)
    for name, records in [("candidates.jsonl", candidates), ("quarantine.jsonl", quarantine)]:
        with (OUT / name).open("w", encoding="utf-8") as output:
            for row in records:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    (OUT / "label_version_comparison.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"source_patients": len(seen), "candidate_patients": len(candidates), "quarantined_patients": len(quarantine),
               "candidate_images": len(candidates) * 2, "annotation_version": "English original XLSX; disagreements quarantined",
               "labels_are_patient_level": True, "diagnostic_keywords_excluded_from_model_inputs": True}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
