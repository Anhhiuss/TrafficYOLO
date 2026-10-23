
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

# Video dau vao
VIDEO_PATH = Path("videos/3.mp4")

# Thu muc luu ket qua
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cau hinh
CONFIDENCE = 0.35
IMAGE_SIZE = 640
TRACKER = "bytetrack.yaml"

# Cac class phuong tien trong mo hinh COCO
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def main():
    if not VIDEO_PATH.exists():
        print(f"Khong tim thay video: {VIDEO_PATH.resolve()}")
        return

    model = YOLO("yolov8n.pt")

    # So ID duy nhat theo class
    seen_ids = defaultdict(set)

    # Luu toa do tam theo tung ID
    trajectories = defaultdict(list)

    # Luu so detection theo class o moi frame
    frame_counts = []

    print("Bat dau tracking, dem xe va luu quy dao...")

    results = model.track(
        source=str(VIDEO_PATH.resolve()),
        tracker=TRACKER,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        stream=True,
        persist=True,
        verbose=False
    )

    frame_number = 0

    for result in results:
        frame_number += 1

        current_counts = Counter()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            xyxy = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()

            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [-1] * len(class_ids)

            for box, class_id, track_id in zip(
                xyxy, class_ids, track_ids
            ):
                # Bo qua cac class khong phai phuong tien
                if class_id not in VEHICLE_CLASSES:
                    continue

                vehicle_type = VEHICLE_CLASSES[class_id]
                current_counts[vehicle_type] += 1

                if track_id < 0:
                    continue

                seen_ids[vehicle_type].add(track_id)

                # Tinh tam bounding box
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                trajectories[track_id].append(
                    (
                        frame_number,
                        vehicle_type,
                        round(center_x, 2),
                        round(center_y, 2),
                    )
                )

        frame_counts.append(
            {
                "frame": frame_number,
                "car": current_counts["car"],
                "motorcycle": current_counts["motorcycle"],
                "bus": current_counts["bus"],
                "truck": current_counts["truck"],
            }
        )

        if frame_number % 60 == 0:
            print(f"Da xu ly {frame_number} frame")

    # Xuat quỹ dao ra CSV
    trajectory_path = OUTPUT_DIR / "trajectory.csv"

    with trajectory_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame",
                "track_id",
                "vehicle_type",
                "center_x",
                "center_y",
            ]
        )

        for track_id, points in sorted(trajectories.items()):
            for frame, vehicle_type, x, y in points:
                writer.writerow(
                    [frame, track_id, vehicle_type, x, y]
                )

    # Xuat thong ke theo frame
    counts_path = OUTPUT_DIR / "vehicle_counts.csv"

    with counts_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame",
                "car",
                "motorcycle",
                "bus",
                "truck",
            ],
        )
        writer.writeheader()
        writer.writerows(frame_counts)

    # Bao cao so ID khac nhau da xuat hien
    print("\n===== KET QUA =====")
    for vehicle_type in VEHICLE_CLASSES.values():
        print(
            f"{vehicle_type}: "
            f"{len(seen_ids[vehicle_type])} track ID"
        )

    print(f"Tong frame: {frame_number}")
    print(f"Quy dao: {trajectory_path.resolve()}")
    print(f"Thong ke: {counts_path.resolve()}")


if __name__ == "__main__":
    main()
