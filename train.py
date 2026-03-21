import subprocess
import os
import glob

project_dir = r"d:\Khoa Học Liên Ngành\Basic Machine Learning\Project\Traffic-sign"
venv_path = os.path.join(project_dir, ".venv", "Scripts", "activate.bat")

candidates = glob.glob(os.path.join(project_dir, "runs", "detect", "*", "weights", "last.pt"))
if candidates:
    checkpoint = max(candidates, key=os.path.getmtime)
    print("Resume từ", checkpoint)
    model_arg = f'model="{checkpoint}"'
    resume_arg = "resume=True"
else:
    print("Không tìm checkpoint, train từ yolov8n.pt")
    model_arg = "model=yolov8n.pt"
    resume_arg = ""

command = (
    f'yolo task=detect mode=train {model_arg} data=dataset/data.yaml '
    f'imgsz=640 epochs=100 batch=8 device=0 workers=1 {resume_arg}'
)
print("Chạy:", command)

os.chdir(project_dir)
subprocess.run(f'call "{venv_path}" && {command}', shell=True, check=True)

