"""Convert DroneVehicle XML annotations to COCO JSON format for UA-CMDet.

TSDroneVehicleDataset extends TSCocoDataset, so it expects standard COCO JSON
with polygon segmentations carrying the OBB coordinates.

Usage (run once per split/modality):
    python scripts/convert_xml_to_coco.py \
        --xml-dir data/DroneVehicle/train/trainlabel \
        --img-dir data/DroneVehicle/train/trainimg \
        --output data/DV_train.json

Run for all six combinations:
    train RGB  : trainlabel  + trainimg  → data/DV_train.json
    train TIR  : trainlabelr + trainimgr → data/DV_train_r.json
    val   RGB  : vallabel    + valimg    → data/DV_val.json
    test  RGB  : testlabel   + testimg   → data/DV_test.json
    test  TIR  : testlabelr  + testimgr  → data/DV_test_r.json
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

# UA-CMDet class order — category IDs are 1-based
CLASSES = ('car', 'freight_car', 'truck', 'bus', 'van')
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(CLASSES)}

# XML may use 'freight car' (space); models use 'freight_car' (underscore)
XML_CLASS_MAP = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'van': 'van',
    'freight car': 'freight_car',
    'freight_car': 'freight_car',
    'feright car': 'freight_car',   # typo present in the HuggingFace dataset
    'feright_car': 'freight_car',   # typo with underscore variant
}


def _polygon_area(coords):
    """Shoelace formula: coords is flat [x1,y1,x2,y2,x3,y3,x4,y4]."""
    xs, ys = coords[0::2], coords[1::2]
    n = len(xs)
    area = sum(xs[i] * ys[(i + 1) % n] - xs[(i + 1) % n] * ys[i] for i in range(n))
    return abs(area) / 2.0


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        cls_raw = obj.find('name').text.strip()
        cls = XML_CLASS_MAP.get(cls_raw)
        if cls is None:
            print(f'  WARNING: unknown class "{cls_raw}" in {xml_path}, skipping')
            continue
        diff_el = obj.find('difficult')
        difficult = int(diff_el.text) if diff_el is not None else 0
        poly = obj.find('polygon')
        if poly is None:
            # Some annotations use <bndbox> instead of <polygon> — convert to
            # an axis-aligned degenerate polygon (4 corners of the bndbox).
            bndbox = obj.find('bndbox')
            if bndbox is None:
                continue
            x1 = float(bndbox.find('xmin').text)
            y1 = float(bndbox.find('ymin').text)
            x2 = float(bndbox.find('xmax').text)
            y2 = float(bndbox.find('ymax').text)
            coords = [x1, y1, x2, y1, x2, y2, x1, y2]
        else:
            coords = [
                float(poly.find('x1').text), float(poly.find('y1').text),
                float(poly.find('x2').text), float(poly.find('y2').text),
                float(poly.find('x3').text), float(poly.find('y3').text),
                float(poly.find('x4').text), float(poly.find('y4').text),
            ]
        xs, ys = coords[0::2], coords[1::2]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        objects.append({
            'class': cls,
            'coords': coords,
            'bbox': [xmin, ymin, xmax - xmin, ymax - ymin],
            'area': _polygon_area(coords),
            'difficult': difficult,
        })
    return w, h, objects


def convert(xml_dir, img_dir, output_path):
    xml_dir = Path(xml_dir)
    img_dir = Path(img_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = [{'id': i + 1, 'name': name} for i, name in enumerate(CLASSES)]
    images, annotations = [], []
    ann_id = 1

    xml_files = sorted(xml_dir.glob('*.xml'))
    skipped = 0
    for img_id, xml_file in enumerate(xml_files, start=1):
        stem = xml_file.stem
        img_name = stem + '.jpg'
        if not (img_dir / img_name).exists():
            skipped += 1
            continue
        w, h, objects = parse_xml(xml_file)
        images.append({'id': img_id, 'file_name': img_name, 'width': w, 'height': h})
        for obj in objects:
            annotations.append({
                'id': ann_id,
                'image_id': img_id,
                'category_id': CLASS_TO_ID[obj['class']],
                'bbox': obj['bbox'],
                'segmentation': [obj['coords']],
                'area': obj['area'],
                'iscrowd': 0,
                'difficult': obj['difficult'],
            })
            ann_id += 1

    coco = {'images': images, 'annotations': annotations, 'categories': categories}
    with open(output_path, 'w') as f:
        json.dump(coco, f)
    print(f'Done: {len(images)} images, {len(annotations)} annotations → {output_path}'
          + (f' ({skipped} images skipped — no matching jpg)' if skipped else ''))


def main():
    p = argparse.ArgumentParser(description='DroneVehicle XML → COCO JSON for UA-CMDet')
    p.add_argument('--xml-dir', required=True, help='directory of .xml annotation files')
    p.add_argument('--img-dir', required=True, help='directory of .jpg image files')
    p.add_argument('--output', required=True, help='output .json path')
    args = p.parse_args()
    convert(args.xml_dir, args.img_dir, args.output)


if __name__ == '__main__':
    main()
