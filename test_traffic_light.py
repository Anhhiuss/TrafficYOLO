
from pathlib import Path
import cv2
from ultralytics import YOLO

# Duong dan video va thu muc luu ket qua
VIDEO_PATH = Path("videos/3.mp4")
OUTPUT_DIR = Path("outputs/traffic_light_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not VIDEO_PATH.exists():
        print("Khong tim thay video:", VIDEO_PATH.resolve())
        return

    print("Dang tai mo hinh YOLO...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Khong mo duoc video.")
        return

    # Kiem tra 7 thoi diem trong video
    frame_indices = [0, 60, 120, 180, 240, 300, 360]

    for frame_no in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()

        if not ok:
            continue

        print(f"Dang kiem tra frame {frame_no}...")

        results = model.predict(
            source=frame,
            classes=[9],       # COCO class 9: traffic light
            conf=0.10,
            imgsz=640,
            device="cpu",
            verbose=False
        )

        result = results[0]
        count = 0 if result.boxes is None else len(result.boxes)
        print(f"So vung den phat hien: {count}")

        # Ve khung phat hien len anh
        annotated = result.plot()
        output_path = OUTPUT_DIR / f"frame_{frame_no:04d}.jpg"
        cv2.imwrite(str(output_path), annotated)

    cap.release()
    print("Hoan tat. Anh ket qua:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
