
from pathlib import Path
import cv2
from ultralytics import YOLO

VIDEO_PATH = Path("videos/3.mp4")
OUTPUT_DIR = Path("outputs/traffic_light_video3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X1, Y1, X2, Y2 = 1000, 0, 1590, 550
UPSCALE = 2


def main():
    if not VIDEO_PATH.exists():
        print("Khong tim thay video:", VIDEO_PATH.resolve())
        return

    print("Dang tai YOLO...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Khong mo duoc video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Tong so frame:", total_frames)

    # Kiem tra dinh ky moi 30 frame.
    for frame_no in range(0, total_frames, 30):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()

        if not ok:
            continue

        height, width = frame.shape[:2]

        # Gioi han toa do de khong cat vuot anh.
        x1 = max(0, min(X1, width - 1))
        y1 = max(0, min(Y1, height - 1))
        x2 = max(x1 + 1, min(X2, width))
        y2 = max(y1 + 1, min(Y2, height))

        roi = frame[y1:y2, x1:x2].copy()
        if roi.size == 0:
            continue

        # Phong to vung den truoc khi dua vao YOLO.
        enlarged = cv2.resize(
            roi,
            None,
            fx=UPSCALE,
            fy=UPSCALE,
            interpolation=cv2.INTER_CUBIC
        )

        results = model.predict(
            source=enlarged,
            classes=[9],       # COCO: traffic light
            conf=0.05,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        result = results[0]
        count = 0 if result.boxes is None else len(result.boxes)
        print(f"Frame {frame_no}: phat hien {count} vung den")

        # Luu ca vung cat va anh co khung YOLO.
        cv2.imwrite(
            str(OUTPUT_DIR / f"crop_{frame_no:04d}.jpg"),
            roi
        )
        cv2.imwrite(
            str(OUTPUT_DIR / f"detected_{frame_no:04d}.jpg"),
            result.plot()
        )

        # Ve hinh chu nhat ROI tren frame goc de kiem tra vi tri cat.
        preview = frame.copy()
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imwrite(
            str(OUTPUT_DIR / f"roi_position_{frame_no:04d}.jpg"),
            preview
        )

    cap.release()
    print("Da xong. Mo thu muc:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
