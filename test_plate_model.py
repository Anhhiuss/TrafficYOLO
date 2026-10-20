from pathlib import Path
from ultralytics import YOLO

model_path = Path("models/license_plate_detector.pt")
image_path = Path("outputs/frame_for_counting.jpg")

if not model_path.exists():
    print("Khong tim thay file mo hinh:", model_path)
elif not image_path.exists():
    print("Khong tim thay anh kiem tra:", image_path)
else:
    print("Dang tai mo hinh bien so...")

    try:
        model = YOLO(str(model_path))
        results = model.predict(
            source=str(image_path),
            conf=0.25,
            imgsz=640,
            device="cpu",
            channels_last=False,
            save=True,
            verbose=False
        )

        count = 0
        for result in results:
            if result.boxes is not None:
                count += len(result.boxes)

        print("So vung bien so du doan:", count)
        print("Kiem tra anh ket qua trong thu muc runs/detect/predict.")
    except Exception as e:
        print("Mo hinh chua chay duoc:", e)
