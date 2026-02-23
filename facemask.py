import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.abspath("face_landmarker.task")  # dùng file bạn tải 'face_landmarker.task'

def draw_points(frame_bgr, face_landmarks, radius=1):
    h, w = frame_bgr.shape[:2]
    for lm in face_landmarks:
        x = int(lm.x * w)
        y = int(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame_bgr, (x, y), radius, (0, 255, 0), -1)
    return frame_bgr

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Không mở được camera. Thử đổi index 0/1/2 hoặc đóng app đang dùng webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = detector.detect(mp_image)

            if result.face_landmarks:
                frame = draw_points(frame, result.face_landmarks[0], radius=1)
                cv2.putText(frame, "Face: YES", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face: NO", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("FaceLandmarker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        # QUAN TRỌNG: đóng detector để tránh lỗi __del__ lúc shutdown
        detector.close()

if __name__ == "__main__":
    main()