import os
import shutil
import random
from pathlib import Path
from paths import ROOT, DATA_YAML, DIRS, CLASS_NAMES

IMG_SRC = DIRS["traffic_images"]
LABEL_SRC = ROOT / "dataset" / "labels"  # nếu đã convert
TRAIN_IMG_DIR = DIRS["dataset_images_train"]
VAL_IMG_DIR = DIRS["dataset_images_val"]
TRAIN_LABEL_DIR = DIRS["dataset_labels_train"]
VAL_LABEL_DIR = DIRS["dataset_labels_val"]

for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LABEL_DIR, VAL_LABEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

images = [f for f in os.listdir(IMG_SRC) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def copy_files(img_list, img_dst, label_dst):
    for img in img_list:
        shutil.copy(IMG_SRC / img, img_dst / img)
        stem = Path(img).stem
        src_label = LABEL_SRC / f"{stem}.txt"
        if src_label.exists():
            shutil.copy(src_label, label_dst / src_label.name)

copy_files(train_imgs, TRAIN_IMG_DIR, TRAIN_LABEL_DIR)
copy_files(val_imgs, VAL_IMG_DIR, VAL_LABEL_DIR)
print(f"OK: {len(train_imgs)} train, {len(val_imgs)} val.")