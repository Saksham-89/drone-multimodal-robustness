"""Prepare matched image + DOTA TXT annotation directories for C2Former.

C2Former's DroneVehicleDataset (mmrotate/datasets/dronevehicle.py) expects:

  img_prefix/
    {id}.jpg          — RGB image
    {id}_tir.jpg      — TIR image

  ann_folder/
    {id}_tir.txt      — DOTA format: x1 y1 x2 y2 x3 y3 x4 y4 classname difficult

Images are symlinked (not copied) to avoid duplicating 14 GB.

Usage:
    python scripts/prepare_c2former_data.py --split train --data-root data/DroneVehicle
    python scripts/prepare_c2former_data.py --split val   --data-root data/DroneVehicle
    python scripts/prepare_c2former_data.py --split test  --data-root data/DroneVehicle

Output directories created inside each split folder:
    {split}MatchedImg/    — paired images
    {split}MatchedLabel/  — DOTA TXT annotations
"""
import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

XML_CLASS_MAP = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'van': 'van',
    'freight car': 'freight_car',
    'freight_car': 'freight_car',
    'feright car': 'freight_car',   # typo present in the HuggingFace dataset
}

SPLIT_DIRS = {
    'train': ('trainimg', 'trainimgr', 'trainlabel'),
    'val':   ('valimg',   'valimgr',   'vallabel'),
    'test':  ('testimg',  'testimgr',  'testlabel'),
}


def xml_to_dota_lines(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall('object'):
        cls_raw = obj.find('name').text.strip()
        cls = XML_CLASS_MAP.get(cls_raw)
        if cls is None:
            print(f'  WARNING: unknown class "{cls_raw}" in {xml_path}, skipping')
            continue
        diff_el = obj.find('difficult')
        difficult = diff_el.text.strip() if diff_el is not None else '0'
        p = obj.find('polygon')
        if p is None:
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            x1 = bndbox.find('xmin').text.strip()
            y1 = bndbox.find('ymin').text.strip()
            x2 = bndbox.find('xmax').text.strip()
            y2 = bndbox.find('ymax').text.strip()
            coords = f'{x1} {y1} {x2} {y1} {x2} {y2} {x1} {y2}'
        else:
            coords = ' '.join([
                p.find('x1').text.strip(), p.find('y1').text.strip(),
                p.find('x2').text.strip(), p.find('y2').text.strip(),
                p.find('x3').text.strip(), p.find('y3').text.strip(),
                p.find('x4').text.strip(), p.find('y4').text.strip(),
            ])
        lines.append(f'{coords} {cls} {difficult}')
    return lines


def prepare(data_root, split):
    data_root = Path(data_root).resolve()
    rgb_dir_name, tir_dir_name, label_dir_name = SPLIT_DIRS[split]

    split_dir = data_root / split
    rgb_src = split_dir / rgb_dir_name
    tir_src = split_dir / tir_dir_name
    label_src = split_dir / label_dir_name

    matched_img = split_dir / f'{split}MatchedImg'
    matched_label = split_dir / f'{split}MatchedLabel'
    matched_img.mkdir(exist_ok=True)
    matched_label.mkdir(exist_ok=True)

    xml_files = sorted(label_src.glob('*.xml'))
    created, skipped = 0, 0

    for xml_file in xml_files:
        stem = xml_file.stem  # e.g. '00001'

        rgb_file = rgb_src / f'{stem}.jpg'
        tir_file = tir_src / f'{stem}.jpg'
        if not rgb_file.exists() or not tir_file.exists():
            skipped += 1
            continue

        # Symlink RGB → {id}.jpg
        rgb_dst = matched_img / f'{stem}.jpg'
        if not rgb_dst.exists():
            os.symlink(rgb_file, rgb_dst)

        # Symlink TIR → {id}_tir.jpg
        tir_dst = matched_img / f'{stem}_tir.jpg'
        if not tir_dst.exists():
            os.symlink(tir_file, tir_dst)

        # Write DOTA TXT → {id}_tir.txt
        txt_path = matched_label / f'{stem}_tir.txt'
        lines = xml_to_dota_lines(xml_file)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))

        created += 1

    print(f'{split}: {created} pairs created → {matched_img.name}/ + {matched_label.name}/'
          + (f' ({skipped} skipped — missing image)' if skipped else ''))


def main():
    p = argparse.ArgumentParser(description='Prepare C2Former matched data directories')
    p.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    p.add_argument('--data-root', default='data/DroneVehicle')
    args = p.parse_args()
    prepare(args.data_root, args.split)


if __name__ == '__main__':
    main()
