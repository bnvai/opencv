from ultralytics import YOLO
import cv2

# Load model pretrained
model = YOLO("yolov8n.pt")

# Load video mp4
cap = cv2.VideoCapture("data\\traffic.mp4")

# COCO vehicle class IDs
vehicle_classes = [2, 3, 5, 7]  # car=2, motorcycle=3, bus=5, truck=7

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if cls in vehicle_classes and conf > 0.4:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0), 2)

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()