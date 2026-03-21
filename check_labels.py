import os

# Thư mục chứa nhãn
TRAIN_LABEL_DIR = "dataset/labels/train"
VAL_LABEL_DIR = "dataset/labels/val"

def check_labels(label_dir):
    print(f"📂 Kiểm tra {label_dir}...")
    for f in os.listdir(label_dir):
        if not f.endswith(".txt"):
            continue
        path = os.path.join(label_dir, f)
        with open(path, "r") as file:
            lines = file.readlines()
            for i, line in enumerate(lines, start=1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"⚠️ {f} dòng {i}: không đúng 5 giá trị → {line.strip()}")
                    continue
                try:
                    class_id = int(parts[0])
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    print(f"⚠️ {f} dòng {i}: dữ liệu không phải số → {line.strip()}")
                    continue

                # Kiểm tra class_id
                if not (0 <= class_id <= 6):
                    print(f"⚠️ {f} dòng {i}: class_id {class_id} ngoài phạm vi [0–6]")

                # Kiểm tra tọa độ
                for j, c in enumerate(coords):
                    if not (0 <= c <= 1):
                        print(f"⚠️ {f} dòng {i}: tọa độ {c} ngoài phạm vi [0–1]")

    print("✅ Kiểm tra xong.\n")

# Kiểm tra train và val
check_labels(TRAIN_LABEL_DIR)
check_labels(VAL_LABEL_DIR)