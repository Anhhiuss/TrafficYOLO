
from pathlib import Path
from ultralytics import YOLO

# Video đầu vào
VIDEO_PATH = Path("videos/3.mp4")

# Thư mục lưu kết quả
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Check video xem có tồn tại không
    if not VIDEO_PATH.exists():
        print(f"Khong tim thay video: {VIDEO_PATH}")
        return

    # Mô hình YOLO
    print("Dang tai mo hinh YOLO...")
    model = YOLO("yolov8n.pt")

    # Nhận diện đối tượng trong video
    print("Dang nhan dien phuong tien...")
    model.predict(
        source=str(VIDEO_PATH),
        conf=0.35,
        imgsz=640,
        save=True,
        project=str(OUTPUT_DIR),
        name="detection",
        exist_ok=True,
        device="cpu",
        verbose=False
    )

    print("Nhan dien hoan tat!")
    print("Video ket qua:", OUTPUT_DIR / "detection")


if __name__ == "__main__":
    main()
