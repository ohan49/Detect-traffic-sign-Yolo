import os
import glob
import shutil
import subprocess
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(r"d:\Khoa Học Liên Ngành\Basic Machine Learning\Project\Traffic-sign")
DATA_YAML = PROJECT_DIR/"dataset"/"data.yaml"
AUTO_LABEL_SRC = PROJECT_DIR/"auto_label"/"images"
TRAIN_IMG_DIR = PROJECT_DIR/"dataset"/"images"/"train"
TRAIN_LABEL_DIR = PROJECT_DIR/"dataset"/"labels"/"train"
RUNS_DETECT = PROJECT_DIR/"runs"/"detect"

def get_best_model():
    candidates = sorted(
        glob.glob(str(PROJECT_DIR / "runs" / "detect" / "*" / "weights" / "best.pt")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None

def mkdirs():
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LABEL_DIR.mkdir(parents=True, exist_ok=True)

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
    )
    label_dir = RUNS_DETECT / "auto_label" / "labels"
    if not label_dir.exists():
        raise FileNotFoundError("Không tìm thấy label auto sau predict.")

    # Build map file stem -> original image path
    img_map = {}
    for p in AUTO_LABEL_SRC.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            img_map[p.stem] = p

    copied = 0
    for txt in label_dir.glob("*.txt"):
        stem = txt.stem
        if stem not in img_map:
            continue
        # copy image + label vào train
        src_img = img_map[stem]
        dst_img = TRAIN_IMG_DIR / src_img.name
        dst_label = TRAIN_LABEL_DIR / f"{stem}.txt"
        shutil.copy2(src_img, dst_img)
        shutil.copy2(txt, dst_label)
        copied += 1
    print(f"[2] Copy xong {copied} ảnh + label từ auto_label vào train set")
    return copied

def train_with_yolo(model_path, resume=False, epochs=50):
    model_arg = f'model="{model_path}"'
    resume_arg = "resume=True" if resume else ""
    cmd = (
        f'yolo task=detect mode=train {model_arg} data="{DATA_YAML}" '
        f'imgsz=640 epochs={epochs} batch=8 device=0 workers=0 {resume_arg}'
    )
    print("[3] Chạy train:", cmd)
    subprocess.run(cmd, shell=True, check=True)

def main():
    mkdirs()
    best = get_best_model()
    if not best:
        print("Không tìm best.pt. Dùng yolov8n.pt train lần đầu.")
        best = "yolov8n.pt"
        resume = False
    else:
        print("Sử dụng best model:", best)
        resume = True

    # Auto label thêm dữ liệu từ auto_label/images
    if AUTO_LABEL_SRC.exists() and any(AUTO_LABEL_SRC.iterdir()):
        copied = auto_label_and_copy(best, conf=0.25)
        if copied == 0:
            print("Không có ảnh nào được auto-label (hoặc không tìm file).")
    else:
        print("Không tìm thư mục auto_label/images hoặc rỗng. Bỏ qua bước auto-label.")
    # Train tiếp
    train_with_yolo(best, resume=resume, epochs=50)

if __name__ == "__main__":
    main()