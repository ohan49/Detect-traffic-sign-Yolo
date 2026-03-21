from pathlib import Path

ROOT = Path(r"d:\Khoa Học Liên Ngành\Basic Machine Learning\Project\Traffic-sign")
DATA_YAML = ROOT / "dataset" / "data.yaml"

DIRS = {
    "auto_label_images": ROOT / "auto_label" / "images",
    "train_images": ROOT / "dataset" / "images" / "train",
    "val_images": ROOT / "dataset" / "images" / "val",
    "train_labels": ROOT / "dataset" / "labels" / "train",
    "val_labels": ROOT / "dataset" / "labels" / "val",
    "runs_detect": ROOT / "runs" / "detect",
    "traffic_images": ROOT / "traffic_train" / "images",
    "annotation_csv": ROOT / "traffic_train" / "annotation.csv",
}

CLASS_NAMES = [
    "Cấm ngược chiều",
    "Cấm dừng và đỗ",
    "Cấm rẽ",
    "Giới hạn tốc độ",
    "Cấm còn lại",
    "Nguy hiểm",
    "Hiệu lệnh",
]