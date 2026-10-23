
from pathlib import Path
import csv

import cv2
import numpy as np

VIDEO_PATH = Path("videos/3.mp4")
OUTPUT_DIR = Path("outputs/traffic_light_color_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Toa do cac bong den tren frame goc 1600 x 1200.

SIGNAL_GROUPS = {
    "RIGHT": {
        "RED_LAMP": (1385, 289, 1417, 323),
        "YELLOW_LAMP": (1385, 327, 1417, 361),
        "GREEN_ARROW": (1378, 366, 1426, 405),
        "COUNTDOWN": (1384, 216, 1420, 250),
    },
    "TOP_MIDDLE": {
        "SIGNAL": (545, 16, 570, 80),
        "TIMER_LEFT": (577, 20, 605, 63),
        "ARROW": (628, 15, 656, 67),
        "TIMER_RIGHT": (672, 10, 700, 52),
    },
}

TOP_MIDDLE_LABELS = {
    "SIGNAL": (25, 30),
    "TIMER_LEFT": (25, 55),
    "ARROW": (25, 80),
    "TIMER_RIGHT": (25, 105),
}


def classify_lamp(roi, min_pixels=8):
    if roi.size == 0:
        return "UNKNOWN", 0

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Chi tinh pixel co mau ro va du sang.
    bright = (s >= 100) & (v >= 150)

    red_mask = bright & ((h <= 10) | (h >= 170))
    yellow_mask = bright & (h >= 15) & (h <= 38)
    green_mask = bright & (h >= 35) & (h <= 95)

    counts = {
        "RED": int(np.count_nonzero(red_mask)),
        "YELLOW": int(np.count_nonzero(yellow_mask)),
        "GREEN": int(np.count_nonzero(green_mask)),
    }

    # Nguong ban dau; se hieu chinh sau khi xem anh ket qua.
    color, pixels = max(counts.items(), key=lambda item: item[1])

    if pixels < min_pixels:
        return "OFF_OR_UNKNOWN", pixels

    return color, pixels


def main():
    if not VIDEO_PATH.exists():
        print("Khong tim thay video:", VIDEO_PATH.resolve())
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Khong mo duoc video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {width}x{height}")

    # Gioi han toa do de tranh vuot kich thuoc frame.
    signal_groups = {}
    for group_name, group_lamps in SIGNAL_GROUPS.items():
        signal_groups[group_name] = {}
        for lamp_name, (x1, y1, x2, y2) in group_lamps.items():
            signal_groups[group_name][lamp_name] = (
                max(0, min(x1, width - 1)),
                max(0, min(y1, height - 1)),
                max(1, min(x2, width)),
                max(1, min(y2, height)),
            )

    csv_path = OUTPUT_DIR / "lamp_colors.csv"
    frame_no = 0
    csv_columns = [
        f"{group_name.lower()}_{lamp_name.lower()}"
        for group_name, group_lamps in SIGNAL_GROUPS.items()
        for lamp_name in group_lamps
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", *csv_columns])

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = {}
            preview = frame.copy()

            for group_name, group_lamps in signal_groups.items():
                results[group_name] = {}
                for lamp_name, (x1, y1, x2, y2) in group_lamps.items():
                    roi = frame[y1:y2, x1:x2]
                    min_pixels = 50 if lamp_name == "GREEN_ARROW" or (
                        group_name == "TOP_MIDDLE" and lamp_name == "ARROW"
                    ) else 8
                    color, pixels = classify_lamp(roi, min_pixels)
                    results[group_name][lamp_name] = color

                    cv2.rectangle(
                        preview, (x1, y1), (x2, y2), (0, 255, 255), 2
                    )
                    label_position = (
                        TOP_MIDDLE_LABELS[lamp_name]
                        if group_name == "TOP_MIDDLE"
                        else (max(10, x1 - 330), max(25, y1))
                    )
                    cv2.putText(
                        preview,
                        f"{group_name} {lamp_name}: {color}",
                        label_position,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2
                    )

            writer.writerow([
                frame_no,
                *[
                    results[group_name][lamp_name]
                    for group_name, group_lamps in SIGNAL_GROUPS.items()
                    for lamp_name in group_lamps
                ],
            ])

            if frame_no % 15 == 0:
                cv2.imwrite(
                    str(OUTPUT_DIR / f"frame_{frame_no:04d}.jpg"),
                    preview,
                )
                group_summary = " | ".join(
                    f"{group_name}: "
                    + ", ".join(
                        f"{lamp_name}={color}"
                        for lamp_name, color in group_results.items()
                    )
                    for group_name, group_results in results.items()
                )
                print(f"Frame {frame_no}: {group_summary}")

            frame_no += 1

    cap.release()
    print("Da xong:", OUTPUT_DIR.resolve())
    print("CSV:", csv_path.resolve())


if __name__ == "__main__":
    main()
