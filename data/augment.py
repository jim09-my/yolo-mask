"""
Oversampling and affine augmentation for the minority class (mask_weared_incorrect).
Generates new training images with transformed bounding boxes.
"""

import random
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import Counter


CLASS_MINORITY = 2   # mask_weared_incorrect
TARGET_COUNT = 1000  # desired number of minority instances after augmentation
ROTATION_RANGE = (-15, 15)      # degrees
SCALE_RANGE = (0.8, 1.2)        # scale factor
TRANSLATION_RANGE = (-0.05, 0.05)  # relative to image dimensions
FLIP_PROB = 0.5


def augment_image_and_boxes(image, boxes, rot_deg, scale, tx, ty, flip):
    """
    Apply affine transformation to image and corresponding bounding boxes.
    boxes: list of (class_id, x_center, y_center, width, height) normalized.
    Returns transformed image and transformed boxes (normalized).
    """
    h, w = image.shape[:2]
    # Build transformation matrix
    center = (w / 2, h / 2)
    rot_mat = cv2.getRotationMatrix2D(center, rot_deg, scale)
    # Add translation
    rot_mat[0, 2] += tx * w
    rot_mat[1, 2] += ty * h
    # Apply to image
    img_aug = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    
    # Transform boxes: convert to absolute pixel coordinates
    transformed_boxes = []
    for cls_id, xc, yc, bw, bh in boxes:
        # Absolute coordinates of top-left and bottom-right
        x1 = (xc - bw/2) * w
        y1 = (yc - bh/2) * h
        x2 = (xc + bw/2) * w
        y2 = (yc + bh/2) * h
        # Apply affine to the four corners
        corners = np.array([[x1, y1, 1], [x1, y2, 1],
                            [x2, y1, 1], [x2, y2, 1]], dtype=np.float32)
        transformed = rot_mat @ corners.T
        tx_corners = transformed[0, :]
        ty_corners = transformed[1, :]
        new_x1 = np.min(tx_corners)
        new_x2 = np.max(tx_corners)
        new_y1 = np.min(ty_corners)
        new_y2 = np.max(ty_corners)
        # Clip to image boundaries
        new_x1 = np.clip(new_x1, 0, w-1)
        new_y1 = np.clip(new_y1, 0, h-1)
        new_x2 = np.clip(new_x2, 0, w-1)
        new_y2 = np.clip(new_y2, 0, h-1)
        # Skip degenerate boxes
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            continue
        # Convert back to normalized YOLO format
        new_xc = (new_x1 + new_x2) / 2 / w
        new_yc = (new_y1 + new_y2) / 2 / h
        new_bw = (new_x2 - new_x1) / w
        new_bh = (new_y2 - new_y1) / h
        transformed_boxes.append((cls_id, new_xc, new_yc, new_bw, new_bh))
    
    # Optionally flip horizontally
    if flip:
        img_aug = cv2.flip(img_aug, 1)
        new_boxes = []
        for cls_id, xc, yc, bw, bh in transformed_boxes:
            new_xc = 1 - xc
            new_boxes.append((cls_id, new_xc, yc, bw, bh))
        transformed_boxes = new_boxes
    
    return img_aug, transformed_boxes


def oversample_minority(train_img_dir, train_label_dir, output_img_dir, output_label_dir):
    """
    Generate augmented copies of images containing minority class instances.
    """
    train_img_dir = Path(train_img_dir)
    train_label_dir = Path(train_label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    # First, copy all original files to output (we will add augmented ones later)
    for src_img in train_img_dir.glob("*.*"):
        shutil.copy2(src_img, output_img_dir / src_img.name)
    for src_txt in train_label_dir.glob("*.txt"):
        shutil.copy2(src_txt, output_label_dir / src_txt.name)

    # Gather all training images that contain minority class
    candidate_files = []
    for txt_path in train_label_dir.glob("*.txt"):
        with open(txt_path) as f:
            lines = f.readlines()
        classes = [int(line.split()[0]) for line in lines if line.strip()]
        if CLASS_MINORITY in classes:
            # Also need the corresponding image
            img_stem = txt_path.stem
            img_ext = None
            for ext in ['.png', '.jpg', '.jpeg']:
                if (train_img_dir / f"{img_stem}{ext}").exists():
                    img_ext = ext
                    break
            if img_ext:
                candidate_files.append((img_stem, img_ext, lines))
    
    if not candidate_files:
        print("No minority class found in training set. Skipping augmentation.")
        return

    # Count current minority instances
    current_minority_count = sum(1 for _, _, lines in candidate_files for line in lines if int(line.split()[0]) == CLASS_MINORITY)
    needed = max(0, TARGET_COUNT - current_minority_count)
    if needed <= 0:
        print(f"Already have {current_minority_count} minority instances, target {TARGET_COUNT}. No augmentation needed.")
        return

    print(f"Current minority instances: {current_minority_count}. Need {needed} more.")
    generated = 0
    attempts = 0
    max_attempts = needed * 3

    while generated < needed and attempts < max_attempts:
        stem, ext, lines = random.choice(candidate_files)
        img_path = train_img_dir / f"{stem}{ext}"
        img = cv2.imread(str(img_path))
        if img is None:
            attempts += 1
            continue
        # Parse boxes from lines
        boxes = []
        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            boxes.append((cls_id, xc, yc, bw, bh))
        
        # Ensure at least one minority box exists (by construction it does)
        # Random affine params
        rot = random.uniform(*ROTATION_RANGE)
        scale = random.uniform(*SCALE_RANGE)
        tx = random.uniform(*TRANSLATION_RANGE)
        ty = random.uniform(*TRANSLATION_RANGE)
        flip = random.random() < FLIP_PROB
        
        img_aug, boxes_aug = augment_image_and_boxes(img, boxes, rot, scale, tx, ty, flip)
        if not boxes_aug:
            attempts += 1
            continue
        
        # Save augmented image and label
        new_stem = f"{stem}_aug_{generated}"
        out_img = output_img_dir / f"{new_stem}{ext}"
        out_txt = output_label_dir / f"{new_stem}.txt"
        cv2.imwrite(str(out_img), img_aug)
        with open(out_txt, 'w') as f:
            for box in boxes_aug:
                f.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
        generated += 1
        attempts += 1

    print(f"Generated {generated} augmented images. Total minority instances now: {current_minority_count + generated}")


if __name__ == "__main__":
    # Example usage: run after initial dataset conversion
    project_root = Path(__file__).resolve().parent.parent
    train_img_orig = project_root / "data" / "processed" / "images" / "train"
    train_label_orig = project_root / "data" / "processed" / "labels" / "train"
    output_img = project_root / "data" / "augmented" / "images"
    output_label = project_root / "data" / "augmented" / "labels"
    oversample_minority(train_img_orig, train_label_orig, output_img, output_label)