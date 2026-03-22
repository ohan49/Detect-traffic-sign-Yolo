import os
import glob
import shutil
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(r"d:\Khoa Học Liên Ngành\Basic Machine Learning\Project\Traffic-sign")
AUTO_LABEL_SRC = PROJECT_DIR/"auto_label"/"images"
RUNS_DETECT = PROJECT_DIR/"runs"/"detect"
TRAIN_IMG_DIR = PROJECT_DIR/"dataset"/"images"/"train"
TRAIN_LABEL_DIR = PROJECT_DIR/"dataset"/"labels"/"train"
CHECKPOINT_FILE = PROJECT_DIR/"checkpoint.txt"

def get_best_model():
    candidates = sorted(
        glob.glob(str(PROJECT_DIR / "runs" / "detect" / "*" / "weights" / "best.pt")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None

def mkdirs():
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)

def load_checkpoint():
    processed = set()
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            processed = set(line.strip() for line in f)
    return processed

def save_checkpoint(stem):
    with open(CHECKPOINT_FILE, "a") as f:
        f.write(stem + "\n")

def auto_label_and_copy(best_model_path, conf=0.25):
    print(f"[1] Auto-label using model: {best_model_path}")
    model = YOLO(str(best_model_path))

    res = model.predict(
        source=str(AUTO_LABEL_SRC),
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(RUNS_DETECT),
        name="auto_label",
        workers=0,
        conf=conf,
        batch=1,     # giảm tải bộ nhớ
        device=0     # chạy trên GPU
    )

    label_dir = RUNS_DETECT / "auto_label" / "labels"
    if not label_dir.exists():
        raise FileNotFoundError("Không tìm thấy label auto sau predict.")

    img_map = {p.stem: p for p in AUTO_LABEL_SRC.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}}

    processed = load_checkpoint()
    copied = 0
    for txt in label_dir.glob("*.txt"):
        stem = txt.stem
        if stem in processed:
            continue

        src_img = img_map.get(stem)
        if not src_img:
            continue

        dst_img = TRAIN_IMG_DIR / src_img.name
        dst_label = TRAIN_LABEL_DIR / f"{stem}.txt"

        if dst_label.exists() and dst_img.exists():
            save_checkpoint(stem)
            continue

        shutil.copy2(src_img, dst_img)
        shutil.copy2(txt, dst_label)
        save_checkpoint(stem)
        copied += 1

    print(f"[2] Copy xong {copied} ảnh + label từ auto_label vào train set")
    return copied

def main():
    mkdirs()
    best = get_best_model()
    if not best:
        print("Không tìm thấy best.pt. Vui lòng train YOLO trước để có best model.")
        return

    print("Sử dụng best model:", best)
    if AUTO_LABEL_SRC.exists() and any(AUTO_LABEL_SRC.iterdir()):
        copied = auto_label_and_copy(best, conf=0.25)
        if copied == 0:
            print("Không có ảnh nào được auto-label (hoặc không tìm file).")
    else:
        print("Không tìm thư mục auto_label/images hoặc rỗng.")

if __name__ == "__main__":
    main()