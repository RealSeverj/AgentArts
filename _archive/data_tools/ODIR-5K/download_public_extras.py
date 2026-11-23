"""Fetch the public RFMiD 2.0 supplement and record BRSET access status."""
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile

import requests

DATA = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_zip_checked(archive, extracted):
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            parts = PurePosixPath(member.filename.replace("\\", "/"))
            if parts.is_absolute() or ".." in parts.parts or any(":" in p for p in parts.parts):
                raise RuntimeError("Unsafe ZIP path")
            output_path = extracted.joinpath(*parts.parts).resolve()
            if not output_path.is_relative_to(extracted.resolve()):
                raise RuntimeError("ZIP path leaves extraction directory")
            if member.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with package.open(member) as source, output_path.open("wb") as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)


def download_rfmid2():
    root = DATA / "RFMiD2"
    root.mkdir(exist_ok=True)
    response = requests.get("https://zenodo.org/api/records/7505822", timeout=45)
    response.raise_for_status()
    meta = response.json()
    write_json(root / "source_metadata.json", meta)
    item = meta["files"][0]
    target = root / item["key"]
    expected = item["checksum"].split(":", 1)[1]
    if not target.exists() or target.stat().st_size != item["size"]:
        temporary = target.with_suffix(target.suffix + ".part")
        for attempt in range(5):
            try:
                with requests.get(item["links"]["self"], stream=True, timeout=(25, 90)) as result:
                    result.raise_for_status()
                    if "html" in result.headers.get("Content-Type", ""):
                        raise RuntimeError("Expected ZIP, received HTML")
                    with temporary.open("wb") as output:
                        for block in result.iter_content(1024 * 1024):
                            if block:
                                output.write(block)
                if temporary.stat().st_size != item["size"]:
                    raise RuntimeError("Downloaded file size mismatch")
                os.replace(temporary, target)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
    md5 = hashlib.md5()
    sha = hashlib.sha256()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != expected:
        raise RuntimeError("RFMiD2 official MD5 mismatch")
    extracted = root / "extracted"
    extract_zip_checked(target, extracted)
    # The publisher's outer ZIP contains three inner ZIPs, one per split.
    nested = list(extracted.glob("*.zip"))
    for archive in nested:
        extract_zip_checked(archive, archive.with_suffix(""))
    images = [p for p in extracted.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
    image_hashes = defaultdict(list)
    for image in images:
        image_hashes[hashlib.sha256(image.read_bytes()).hexdigest()].append(str(image.relative_to(root)))
    exact_duplicate_groups = [paths for paths in image_hashes.values() if len(paths) > 1]
    tables = []
    for path in extracted.rglob("*.csv"):
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            text = raw.decode("cp1252")
            encoding = "cp1252 (fallback; original bytes preserved)"
        reader = csv.DictReader(io.StringIO(text, newline=""))
        rows = list(reader)
        split_images = [p for p in path.parent.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}]
        ids = [row["ID"].strip() for row in rows]
        id_match = len(ids) == len(set(ids)) == len(split_images) and set(ids) == {p.stem for p in split_images}
        image_ids = {p.stem for p in split_images}
        tables.append({"path": str(path.relative_to(root)), "rows": len(rows), "encoding_used": encoding,
                       "columns": reader.fieldnames, "image_count": len(split_images), "image_label_ids_match": id_match,
                       "matched_ids_count": len(set(ids) & image_ids),
                       "label_ids_without_image": sorted(set(ids) - image_ids),
                       "image_ids_without_label": sorted(image_ids - set(ids))})
    report = {"dataset": "RFMiD 2.0", "source": "https://zenodo.org/records/7505822", "license": meta["metadata"].get("license"),
              "archive_bytes": target.stat().st_size, "official_md5_verified": True, "zip_members_crc_verified": True,
              "md5": md5.hexdigest(), "sha256": sha.hexdigest(), "image_count": len(images), "csv_files": tables,
              "all_checks_passed": all(table["image_label_ids_match"] for table in tables),
              "download_complete": True,
              "unique_image_sha256_count": len(image_hashes), "exact_duplicate_image_groups": exact_duplicate_groups,
              "checked_utc": datetime.now(timezone.utc).isoformat()}
    if len(images) != 860:
        raise RuntimeError(f"Unexpected RFMiD2 image count: {len(images)}")
    write_json(root / "verification_report.json", report)
    (root / "README.md").write_text("# RFMiD 2.0\n\n原始 RFMiD 的补充数据集，独立存放，不能与 RFMiD 的 3,200 张混为同一版本。\n\n"
        "来源：https://zenodo.org/records/7505822\n\n许可：CC BY 4.0。包含 860 张图像，原始划分与标注保存在 extracted/。"
        "官方 MD5 与 ZIP 成员 CRC 均已检查，数量与标注详情见 verification_report.json。\n", encoding="utf-8")
    with (root / "README.md").open("a", encoding="utf-8") as handle:
        handle.write("\n实读原包划分：Training 509、Validation 177、Test 174 张。原 CSV 含非 UTF-8 字符，读取时使用 CP1252 回退；下载文件的原始字节保持不变。\n")
        handle.write("\n## 原始标注与文件名存在不一致\n\n官方 MD5 通过，但原包有 6 张图像的文件名 ID 无对应标注，同时有 6 行标注 ID 无同名图像；仅 854 张可直接按 ID 配对。这是原包内容检查发现的问题，没有自动猜测或修改对应关系。训练前应排除这些未匹配项或向作者核实。\n\n")
        for table in tables:
            if not table["image_label_ids_match"]:
                handle.write("- " + table["path"] + "：缺图标注 ID=" + repr(table["label_ids_without_image"]) + "; 缺标注图像 ID=" + repr(table["image_ids_without_label"]) + "。\n")
        if exact_duplicate_groups:
            handle.write("\n另检出 3 组跨集合逐字节重复图像（SHA-256 相同），860 张文件实际有 857 个不同图像内容。用作训练/验证前应去重并避免跨集合泄漏，原数据未删除或重命名。\n\n")
            for group in exact_duplicate_groups:
                handle.write("- " + " / ".join(group) + "\n")
    print(json.dumps({"dataset": "RFMiD2", "images": len(images), "bytes": target.stat().st_size,
                      "archive_md5_verified": True, "matched_image_label_ids": sum(table["matched_ids_count"] for table in tables),
                      "annotation_issues": not report["all_checks_passed"]}), flush=True)


def record_brset():
    root = DATA / "BRSET"
    root.mkdir(exist_ok=True)
    page = "https://physionet.org/content/brazilian-ophthalmological/1.0.2/"
    url = "https://physionet.org/files/brazilian-ophthalmological/1.0.2/labels.csv"
    with requests.get(url, stream=True, timeout=(20, 40)) as response:
        status = response.status_code
    write_json(root / "access_status.json", {"dataset": "BRSET", "version": "1.0.2", "official_page": page,
               "checked_file_url": url, "anonymous_status_code": status, "images_downloaded": 0,
               "status": "credentialed_access_required", "checked_utc": datetime.now(timezone.utc).isoformat()})
    (root / "README.md").write_text("# BRSET — 尚未取得影像\n\n官方当前版本：https://physionet.org/content/brazilian-ophthalmological/1.0.2/\n\n"
        "公开页面标明 Credentialed Access；匿名读取原始标注返回 HTTP " + str(status) + "。需要完成 PhysioNet 身份认证、所需培训并签署数据使用协议后访问。"
        "本目录仅记录获取状态，不包含训练影像；没有代用户注册、签署或提交申请。\n", encoding="utf-8")
    print(json.dumps({"dataset": "BRSET", "anonymous_http_status": status, "images_downloaded": 0}), flush=True)


if __name__ == "__main__":
    record_brset()
    download_rfmid2()
