
import csv
from pathlib import Path

import cv2

VIDEO_PATH = Path("videos/3.mp4")
CSV_PATH = Path("outputs/counted_vehicles.csv")
OUTPUT_DIR = Path("outputs/count_event_frames")
LINE_START = (0, 925)
LINE_END = (1400, 750)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Kiem tra 5 su kien da duoc dem
with CSV_PATH.open("r", newline="", encoding="utf-8-sig") as f:
    events = list(csv.DictReader(f))

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Khong mo duoc video.")
    raise SystemExit(1)

for event in events:
    frame_number = int(event["frame"])
    track_id = event["track_id"]

    # Lay frame truoc va sau su kien
    for offset in (-3, -1, 0, 1, 3):
        target_frame = frame_number + offset

        if target_frame < 1:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
        ok, frame = cap.read()

        if not ok:
            continue

        cv2.line(
            frame, LINE_START, LINE_END, (0, 0, 255), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Red-light event ID {track_id}, frame {target_frame}, "
            f"signal {event.get('signal_state', 'UNKNOWN')}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        output_file = OUTPUT_DIR / f"event_{frame_number}_offset_{offset}.jpg"
        cv2.imwrite(str(output_file), frame)

    print(f"Da tao anh kiem tra cho ID {track_id}, frame {frame_number}")

cap.release()

print("Thu muc anh:", OUTPUT_DIR.resolve())
