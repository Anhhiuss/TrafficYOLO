
from pathlib import Path
from ultralytics import YOLO

# Kết qu
VIDEO_PATH = Path("videos/3.mp4")

# Thư mục lưu kết quả
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not VIDEO_PATH.exists():
        print(f"Khong tim thay video: {VIDEO_PATH.resolve()}")
        return

    print("Dang tai mo hinh YOLO...")
    model = YOLO("yolov8n.pt")

    print("Bat dau tracking bang ByteTrack...")

    # stream = True để video đọc được từng frame, tránh bị tích lũy kết quả
    results = model.track(
        source=str(VIDEO_PATH.resolve()),
        tracker="bytetrack.yaml",
        conf=0.35,
        imgsz=640,
        device="cpu",
        persist=True,
        stream=True,
        save=True,
        project=str(OUTPUT_DIR.resolve()),
        name="tracking",
        exist_ok=True,
        verbose=False
    )

    frame_count = 0

    for result in results:
        frame_count += 1

        # Lấy ID của các đối tượng trong video
        if result.boxes is not None and result.boxes.id is not None:
            ids = result.boxes.id.int().cpu().tolist()
            unique_ids = sorted(set(ids))

            if frame_count % 30 == 0:
                print(
                    f"Frame {frame_count}: "
                    f"{len(ids)} detection, "
                    f"ID dang thay: {unique_ids}"
                )

        elif frame_count % 30 == 0:
            print(f"Frame {frame_count}: chua co ID tracking")

    print("\nTracking hoan tat!")
    print(f"So frame da xu ly: {frame_count}")
    print(f"Thu muc ket qua: {(OUTPUT_DIR / 'tracking').resolve()}")


if __name__ == "__main__":
    main()
