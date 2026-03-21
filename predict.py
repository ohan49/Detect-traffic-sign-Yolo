import glob
import os
from ultralytics import YOLO

best_models = glob.glob("runs/detect/*/weights/best.pt")
if not best_models:
    raise FileNotFoundError("Không tìm thấy best.pt trong runs/detect/*/weights/")
# lấy file best.pt mới nhất theo thời gian sửa
best_model = max(best_models, key=os.path.getmtime)
print("Sử dụng model:", best_model)

model = YOLO(best_model)
results = model.predict(source="traffic_public_test\images", save=True, save_txt=True, workers=0)
print("✅ Predict hoàn thành. Kiểm tra runs/detect/predict/")