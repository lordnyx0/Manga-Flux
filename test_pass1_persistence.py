from pathlib import Path
import json, sys
sys.path.append(r'C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux')
from core.analysis.pass1_pipeline import run_pass1

try:
    img_path = r'C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\Doukutsu_Ou_kara_Hajimeru_Rakuen_Life\chapters\26_2\inputs\page_001.jpg'
    out_mask = r'C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\test_yolo_meta\mask.png'
    out_meta = r'C:\Users\Nyx\Desktop\MANGACOLOR\Manga-Flux\outputs\test_yolo_meta'

    meta_file = run_pass1(
        page_image=img_path,
        style_reference='dummy',
        output_mask=out_mask,
        output_metadata_dir=out_meta,
        page_num=1,
        page_prompt='dummy'
    )

    if meta_file:
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        dets = meta.get('detections', [])
        print(f'Metadata successfully generated!')
        print(f'Total YOLO Detections Persisted: {len(dets)}')
        for d in dets:
            print(f"- Class: {d.get('class_name', 'unknown')}, BBox: {d.get('bbox')}")
    else:
        print('Failed to generate metadata (return None)')
except Exception as e:
    import traceback
    traceback.print_exc()
