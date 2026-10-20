
import csv
from collections import defaultdict
from pathlib import Path

import cv2

VIDEO_PATH = Path("videos/16h15.25.9.22.mp4")
TRAJECTORY_PATH = Path("outputs/trajectory.csv")
OUTPUT_PATH = Path("outputs/trajectory_video.mp4")

# So diem lich su toi da cho moi ID
MAX_TRAIL_LENGTH = 30


def main():
    if not VIDEO_PATH.exists():
        print(f"Khong tim thay video: {VIDEO_PATH.resolve()}")
        return

    if not TRAJECTORY_PATH.exists():
        print(f"Khong tim thay CSV: {TRAJECTORY_PATH.resolve()}")
        return

    # Doc quy dao tu CSV va gom theo frame
    trajectories = defaultdict(list)

    with TRAJECTORY_PATH.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            x = int(float(row["center_x"]))
            y = int(float(row["center_y"]))

            trajectories[frame].append((track_id, x, y))

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Khong mo duoc video dau vao.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("Khong tao duoc video dau ra.")
        cap.release()
        return

    # Lich su vi tri cua tung ID
    trails = defaultdict(list)
    frame_number = 0

    print("Dang ve quy dao len video...")

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        for track_id, x, y in trajectories.get(frame_number, []):
            trails[track_id].append((x, y))
            trails[track_id] = trails[track_id][-MAX_TRAIL_LENGTH:]

            cv2.putText(
                frame,
                f"ID {track_id}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        # Ve duong di cua cac ID da duoc ghi nhan
        for track_id, points in trails.items():
            for i in range(1, len(points)):
                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (0, 255, 0),
                    2
                )

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        writer.write(frame)

        if frame_number % 60 == 0:
            print(f"Da xu ly {frame_number} frame")

    cap.release()
    writer.release()

    print("\nVe quy dao hoan tat!")
    print(f"Video ket qua: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
