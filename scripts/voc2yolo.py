"""Convert PASCAL VOC XML annotations to YOLOv8 TXT format with stratified train/val/test split."""

import os
import glob
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


CLASS_MAP = {"with_mask": 0, "without_mask": 1, "mask_weared_incorrect": 2}
RANDOM_SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train / val / test


def parse_xml(xml_path):
    """Parse a PASCAL VOC XML file, returning (filename, width, height, boxes).

    boxes is a list of (class_id, x_center, y_center, box_width, box_height).
    Objects marked <difficult>1</difficult> are skipped.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.find("filename").text
    size = root.find("size")
    width = float(size.find("width").text)
    height = float(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        difficult = obj.find("difficult")
        if difficult is not None and int(difficult.text) == 1:
            continue

        name = obj.find("name").text
        cls_id = CLASS_MAP[name]

        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)

        x_center = (xmin + xmax) / 2.0 / width
        y_center = (ymin + ymax) / 2.0 / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height

        boxes.append((cls_id, x_center, y_center, box_width, box_height))

    return filename, boxes


def stratified_split(records, ratios, seed):
    """Split records into subsets with stratified class distribution.

    Each record is (filename, boxes). Images are grouped by their primary class
    (most frequent class), then each group is split according to ratios.
    """
    random.seed(seed)

    # Group images by primary class
    groups = defaultdict(list)
    for rec in records:
        boxes = rec[1]
        if not boxes:
            continue
        class_counts = Counter(b[0] for b in boxes)
        primary_class = class_counts.most_common(1)[0][0]
        groups[primary_class].append(rec)

    splits = [[], [], []]
    for cls_id, group in groups.items():
        random.shuffle(group)
        n = len(group)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        # Assign deterministically
        splits[0].extend(group[:n_train])
        splits[1].extend(group[n_train:n_train + n_val])
        splits[2].extend(group[n_train + n_val:])

    return splits


def write_yolo_label(boxes, txt_path):
    """Write YOLO TXT label file: cls_id x_center y_center w h (6 decimal places)."""
    with open(txt_path, "w", encoding="utf-8") as f:
        for cls_id, xc, yc, bw, bh in boxes:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_images = project_root / "data" / "raw" / "images"
    src_annotations = project_root / "data" / "raw" / "annotations"
    dst_images = project_root / "data" / "processed" / "images"
    dst_labels = project_root / "data" / "processed" / "labels"

    # Parse all XMLs
    xml_paths = sorted(glob.glob(str(src_annotations / "*.xml")))
    print(f"Found {len(xml_paths)} annotation files.")

    records = []
    parse_errors = 0
    for xml_path in xml_paths:
        try:
            records.append(parse_xml(xml_path))
        except Exception as e:
            print(f"  Skipping {Path(xml_path).name}: {e}")
            parse_errors += 1

    print(f"Parsed {len(records)} annotations successfully ({parse_errors} errors).")

    # Stratified split
    splits = stratified_split(records, SPLIT_RATIOS, RANDOM_SEED)
    split_names = ["train", "val", "test"]

    # Create output directories
    for name in split_names:
        (dst_images / name).mkdir(parents=True, exist_ok=True)
        (dst_labels / name).mkdir(parents=True, exist_ok=True)

    # Write labels and copy images
    split_stats = {}
    all_class_counts = Counter()
    src_images_map = {p.stem: p for p in Path(src_images).iterdir() if p.is_file()}

    for name, split_records in zip(split_names, splits):
        class_counts = Counter()
        for filename, boxes in split_records:
            # Write TXT label
            stem = Path(filename).stem
            txt_path = dst_labels / name / f"{stem}.txt"
            write_yolo_label(boxes, txt_path)

            # Copy image (find matching image by stem, preserving original extension)
            ext = None
            for candidate in src_images_map.values():
                if candidate.stem == stem:
                    ext = candidate.suffix
                    break
            if ext is None:
                print(f"  Warning: no image found for {filename}")
                continue
            img_src = src_images / f"{stem}{ext}"
            img_dst = dst_images / name / f"{stem}{ext}"
            shutil.copy2(img_src, img_dst)

            class_counts.update(b[0] for b in boxes)

        split_stats[name] = {"images": len(split_records), "instances": class_counts}
        all_class_counts.update(class_counts)   # ✅ 修复点：直接传 Counter，不用 .values()

    # Print statistics
    cls_names = {v: k for k, v in CLASS_MAP.items()}
    print("\n" + "=" * 50)
    print("Dataset Conversion Summary")
    print("=" * 50)
    total_images = sum(s["images"] for s in split_stats.values())
    print(f"Total images:      {total_images}")
    for name in split_names:
        s = split_stats[name]
        print(f"  {name:6s}:        {s['images']:4d} images  "
              f"{sum(s['instances'].values()):4d} instances")
    print(f"\nTotal instances:   {sum(all_class_counts.values())}")
    print("Per class:")
    for cls_id, count in sorted(all_class_counts.items()):
        print(f"  {cls_names[cls_id]:30s} (id={cls_id}): {count:5d}")
    print("=" * 50)
    print(f"\nOutput structure:")
    for name in split_names:
        print(f"  data/processed/images/{name}/")
        print(f"  data/processed/labels/{name}/")


if __name__ == "__main__":
    main()