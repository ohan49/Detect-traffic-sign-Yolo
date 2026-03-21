from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/*/weights/best.pt")

cap = cv2.VideoCapture("traffic_public_test/video.mp4")  # hoặc 0
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("detect_out.mp4", fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, imgsz=640, conf=0.25, device="0")  # GPU 0
    boxes = results[0].boxes.cpu().numpy() if results[0].boxes is not None else []
    for box in boxes:
        x1,y1,x2,y2,conf,cls = box[:6]
        cls = int(cls)
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame,label,(int(x1),int(y1)-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    out.write(frame)
    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release(); out.release(); cv2.destroyAllWindows()