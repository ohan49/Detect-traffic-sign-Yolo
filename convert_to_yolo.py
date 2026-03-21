import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

IMG_DIR = "traffic_train/images"
CSV_FILE = "traffic_train/annotation.csv"
LABEL_DIR = "dataset/labels"
IMG_OUT_DIR = "dataset/images"

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(IMG_OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_FILE)

# Dùng file_name để ghép ảnh
images = df["file_name"].unique()
train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

def convert(split_imgs, split_name):
    out_label_dir = os.path.join(LABEL_DIR, split_name)
    out_img_dir = os.path.join(IMG_OUT_DIR, split_name)
    os.makedirs(out_label_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    for img_name in split_imgs:
        rows = df[df["file_name"] == img_name]
        label_file = os.path.join(out_label_dir, os.path.splitext(img_name)[0] + ".txt")
        with open(label_file, "w", encoding="utf-8") as f:
            for _, row in rows.iterrows():
                xmin, ymin, w_box, h_box = eval(row["bbox"])
                img_w, img_h = row["width"], row["height"]

                x_center = (xmin + w_box / 2) / img_w
                y_center = (ymin + h_box / 2) / img_h
                width = w_box / img_w
                height = h_box / img_h

                class_id = int(row["category_id"]) - 1
                if class_id < 0:
                    continue
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        src_img = os.path.join(IMG_DIR, img_name)
        dst_img = os.path.join(out_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

convert(train_imgs, "train")
convert(val_imgs, "val")

print("Done. Labels -> dataset/labels/{train,val}, images -> dataset/images/{train,val}")