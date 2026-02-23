import os
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def get_endpoint_volume():
    dev = AudioUtilities.GetSpeakers()
    if hasattr(dev, "EndpointVolume"):
        return dev.EndpointVolume.QueryInterface(IAudioEndpointVolume)
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    interface = dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return cast(interface, POINTER(IAudioEndpointVolume))

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def main():
    # model path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "hand_landmarker.task")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")

    # mediapipe tasks
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # volume controller
    vol = get_endpoint_volume()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # calibration
    dist_min, dist_max = 10, 50
    pinch_threshold = 60   # chỉ khi chụm tay mới update
    alpha = 0.15
    smooth = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(mp_image)

        status = "No hand"
        vol_percent = None

        if res.hand_landmarks:
            hand = res.hand_landmarks[0]

            def to_px(lm):
                return int(lm.x * w), int(lm.y * h)

            x1, y1 = to_px(hand[4])  # thumb tip
            x2, y2 = to_px(hand[8])  # index tip

            cv2.circle(frame, (x1, y1), 8, (255,255,255), -1)
            cv2.circle(frame, (x2, y2), 8, (255,255,255), -1)
            cv2.line(frame, (x1, y1), (x2, y2), (255,255,255), 2)

            dist = float(np.hypot(x2-x1, y2-y1))
            dist_clip = float(np.clip(dist, dist_min, dist_max))

            target = float(np.interp(dist_clip, [dist_min, dist_max], [0.0, 1.0]))
            target = clamp01(target)

            # gate: chỉ update khi pinch
            if dist < pinch_threshold:
                status = "Pinch: updating volume"
                if smooth is None:
                    smooth = target
                else:
                    smooth = (1-alpha)*smooth + alpha*target
                try:
                    vol.SetMasterVolumeLevelScalar(float(smooth), None)
                except Exception as e:
                    status = f"Set volume failed: {e}"
                vol_percent = int(round(100*(smooth if smooth is not None else target)))
            else:
                status = "Open hand: pinch to control"
                vol_percent = int(round(100*target))

        cv2.putText(frame, status, (50, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        if vol_percent is not None:
            cv2.putText(frame, f"Volume: {vol_percent}%", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow("Hand Volume Control", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()