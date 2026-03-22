import glob
import os
from pathlib import Path
from ultralytics import YOLO

PROJECT_DIR = Path(r"d:\Khoa Học Liên Ngành\Basic Machine Learning\Project\Traffic-sign")
AUTO_LABEL_SRC = PROJECT_DIR/"auto_label"/"images"
RUNS_DETECT = PROJECT_DIR/"runs"/"detect"

BATCH_SIZE = 1000

def get_best_model():
    candidates = sorted(
        glob.glob(str(PROJECT_DIR / "runs" / "detect" / "*" / "weights" / "best.pt")),
        key=os.path.getmtime,
    )
    return candidates[-1] if candidates else None

def main():
    best = get_best_model()
    if not best:
        print("Không tìm thấy best.pt. Vui lòng train YOLO trước để có best model.")
        return

    print("Sử dụng best model:", best)
    model = YOLO(str(best))

    
    images = [p for p in AUTO_LABEL_SRC.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"}]
    total_batches = (len(images) + BATCH_SIZE - 1) // BATCH_SIZE

   
    for i in range(0, len(images), BATCH_SIZE):
        batch_imgs = images[i:i+BATCH_SIZE]
        print(f"===> Đang xử lý batch {i//BATCH_SIZE+1}/{total_batches} ({len(batch_imgs)} ảnh)")

        model.predict(
            source=[str(p) for p in batch_imgs],
            save=True,
            save_txt=True,
            save_conf=True,
            project=str(RUNS_DETECT),
            name="auto_label",  
            workers=0,
            conf=0.35,
            batch=8,             
            device=0,
            imgsz=640,
            exist_ok=True
    )

if __name__ == "__main__":
    main()