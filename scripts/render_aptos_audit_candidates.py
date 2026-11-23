"""Render factual audit contact sheets; never edit source images or splits."""
from pathlib import Path
import json
from PIL import Image, ImageOps, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'cloud_upload/aptos_vitl_20261113/leakage_audit'
screen = json.loads((OUT / 'cached_pair_screen.json').read_text(encoding='utf-8'))
candidates = [x for x in screen['broad_candidates'] if x['phash_distance'] <= 2 and x['dhash_distance'] <= 4]
for i, item in enumerate(candidates, 1):
    item['audit_pair_id'] = i
    item['test_absolute_path'] = str(ROOT / 'data' / item['test_image'])
    item['other_absolute_path'] = str(ROOT / 'data' / item['other_image'])
    page = Image.new('RGB', (1640, 920), 'white')
    draw = ImageDraw.Draw(page)
    draw.text((16, 8), f"Pair {i:02d} | pHash={item['phash_distance']} dHash={item['dhash_distance']} RMSE={item['rgb32_rmse']:.6f} corr={item['gray32_correlation']:.6f}", fill='black')
    for j, key in enumerate(['test_image', 'other_image']):
        path = ROOT / 'data' / item[key]
        with Image.open(path) as source:
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            shown = ImageOps.contain(rgb, (800, 800), Image.Resampling.LANCZOS)
        x = 16 + j * 820
        page.paste(shown, (x + (800-shown.width)//2, 64 + (800-shown.height)//2))
        draw.text((x, 36), ('TEST ' if j == 0 else item['other_split'].upper()+' ') + path.stem, fill='black')
    sheet = OUT / f'pair_{i:02d}.jpg'
    page.save(sheet, quality=94, subsampling=0)
    item['contact_sheet'] = str(sheet)
(OUT / 'candidate_12.json').write_text(json.dumps({'selection':'pHash <= 2 AND dHash <= 4; TEST vs train/val', 'count':len(candidates), 'pairs':candidates}, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({'count':len(candidates), 'candidate_json':str(OUT/'candidate_12.json'), 'sheet_glob':str(OUT/'pair_*.jpg')}, ensure_ascii=False))
