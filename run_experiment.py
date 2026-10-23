import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# 1. THIET LAP THUC NGHIEM
# =========================
VIDEO_PATH = Path("videos/3.mp4")
OUTPUT_DIR = Path("outputs/experiment")
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "experiment_video.mp4"
OUTPUT_CSV = OUTPUT_DIR / "violations.csv"
MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = Path("models/license_plate_detector.pt")
TRACKER_CONFIG = Path("vehicle_bytetrack.yaml")

CONFIDENCE = 0.30
RECOVERY_CONFIDENCE = 0.25
IMAGE_SIZE = 1280

# Duong thang can theo hai dau vung danh dau tren video 1600 x 1200.
LINE_START = (0, 925)
LINE_END = (1400, 750)
LINE_POINTS = (LINE_START, LINE_END)
LINE_X_COORDS = np.array([point[0] for point in LINE_POINTS])
LINE_Y_COORDS = np.array([point[1] for point in LINE_POINTS])
LINE_SIDE_MARGIN = 5
MAX_CROSSING_FRAME_GAP = 5
MAX_MISSED_FRAMES = 15
MAX_VIOLATION_MISSED_FRAMES = 15
MAX_TRACK_SPEED = 40
# Vung den dieu khien huong xe dang theo doi.
APPROACH_SIGNAL_ROI = (545, 16, 570, 80)

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
GREEN = (50, 220, 50)  # BGR
RED = (0, 0, 255)      # BGR
SHORT_VEHICLE_TYPES = {
    "motorcycle": "MOTO",
    "car": "CAR",
    "bus": "BUS",
    "truck": "TRUCK",
}


# =========================
# 3. LOGIC PHAT HIEN VA PHONG TO BIEN SO
# =========================
def save_plate_crop(vehicle_crop, plate_model, event_dir):
    """Detect and return an enlarged plate image without saving it separately."""
    if plate_model is None or vehicle_crop.size == 0:
        return None

    results = plate_model.predict(
        source=vehicle_crop,
        conf=0.25,
        imgsz=640,
        device="cpu",
        channels_last=False,
        verbose=False
    )
    if not results or results[0].boxes is None:
        return None

    boxes = results[0].boxes
    if len(boxes) == 0:
        return None

    confidences = boxes.conf.cpu().tolist()
    best_index = max(
        range(len(confidences)),
        key=lambda i: confidences[i]
    )
    x1, y1, x2, y2 = boxes.xyxy[best_index].cpu().tolist()
    h, w = vehicle_crop.shape[:2]

    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    plate_crop = vehicle_crop[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None

    return cv2.resize(
        plate_crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


# =========================
# 4. LOGIC NHAN DIEN MAU DEN VA HUONG CAT VACH
# =========================
def classify_approach_signal(frame):
    x1, y1, x2, y2 = APPROACH_SIGNAL_ROI
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    bright = (saturation >= 100) & (value >= 150)
    color_masks = {
        "RED": bright & ((hue <= 10) | (hue >= 170)),
        "YELLOW": bright & (hue >= 15) & (hue <= 38),
        "GREEN": bright & (hue >= 35) & (hue <= 95),
    }
    color, pixel_count = max(
        ((name, int(np.count_nonzero(mask))) for name, mask in color_masks.items()),
        key=lambda item: item[1],
    )
    return color if pixel_count >= 8 else "UNKNOWN"


def get_line_side(center_x, center_y):
    if not LINE_POINTS[0][0] <= center_x <= LINE_POINTS[-1][0]:
        return None

    line_y = np.interp(center_x, LINE_X_COORDS, LINE_Y_COORDS)
    if center_y >= line_y + LINE_SIDE_MARGIN:
        return "below"
    if center_y <= line_y - LINE_SIDE_MARGIN:
        return "above"
    return None


def is_red_light_crossing(previous_side, current_side, signal_state):
    return (
        previous_side == "below"
        and current_side == "above"
        and signal_state == "RED"
    )


def is_track_continuous(previous_track, current_box, vehicle_type, frame_number):
    if previous_track["vehicle_type"] != vehicle_type:
        return False

    elapsed = max(1, frame_number - previous_track["last_seen"])
    old_box = previous_track["box"]
    predicted_x = (
        (old_box[0] + old_box[2]) / 2
        + previous_track["velocity_x"] * elapsed
    )
    predicted_y = (
        (old_box[1] + old_box[3]) / 2
        + previous_track["velocity_y"] * elapsed
    )
    current_x = (current_box[0] + current_box[2]) / 2
    current_y = (current_box[1] + current_box[3]) / 2
    old_diagonal = np.hypot(
        old_box[2] - old_box[0], old_box[3] - old_box[1]
    )
    center_distance = np.hypot(current_x - predicted_x, current_y - predicted_y)
    max_shift = max(25, old_diagonal * 0.25) + 8 * elapsed
    intersection_x1 = max(old_box[0], current_box[0])
    intersection_y1 = max(old_box[1], current_box[1])
    intersection_x2 = min(old_box[2], current_box[2])
    intersection_y2 = min(old_box[3], current_box[3])
    intersection = max(0, intersection_x2 - intersection_x1) * max(
        0, intersection_y2 - intersection_y1
    )
    old_area = (old_box[2] - old_box[0]) * (old_box[3] - old_box[1])
    current_area = (current_box[2] - current_box[0]) * (
        current_box[3] - current_box[1]
    )
    union = old_area + current_area - intersection
    iou = intersection / union if union > 0 else 0
    return center_distance <= max_shift and iou >= 0.05


def is_vehicle_box_plausible(box, vehicle_type, frame_width, frame_height):
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    area_fraction = box_width * box_height / (frame_width * frame_height)

    limits = {
        "motorcycle": (0.25, 0.55, 0.08),
        "car": (0.45, 0.70, 0.16),
        "bus": (0.70, 0.85, 0.25),
        "truck": (0.65, 0.85, 0.25),
    }
    max_width, max_height, max_area = limits[vehicle_type]
    return (
        box_width <= frame_width * max_width
        and box_height <= frame_height * max_height
        and area_fraction <= max_area
    )


def recover_oversized_vehicle_boxes(model, frame, boxes, class_ids, width, height):
    if not any(
        class_id in VEHICLE_CLASSES
        and not is_vehicle_box_plausible(
            clamp_box(box, width, height),
            VEHICLE_CLASSES[class_id],
            width,
            height,
        )
        for box, class_id in zip(boxes, class_ids)
    ):
        return boxes

    result = model.predict(
        frame,
        conf=RECOVERY_CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        classes=list(VEHICLE_CLASSES),
        verbose=False,
    )[0]
    if result.boxes is None:
        return boxes

    raw_boxes = result.boxes.xyxy.cpu().tolist()
    raw_classes = result.boxes.cls.int().cpu().tolist()
    original_boxes = [list(box) for box in boxes]
    used_raw_indices = set()

    for index, (tracked_box, class_id) in enumerate(
        zip(original_boxes, class_ids)
    ):
        if class_id not in VEHICLE_CLASSES:
            continue
        tracked_box = clamp_box(tracked_box, width, height)
        if is_vehicle_box_plausible(
            tracked_box, VEHICLE_CLASSES[class_id], width, height
        ):
            continue

        tx = (tracked_box[0] + tracked_box[2]) / 2
        ty = (tracked_box[1] + tracked_box[3]) / 2
        candidates = []
        for raw_index, (raw_box, raw_class) in enumerate(
            zip(raw_boxes, raw_classes)
        ):
            if raw_class != class_id or raw_index in used_raw_indices:
                continue
            candidate = clamp_box(raw_box, width, height)
            if not is_vehicle_box_plausible(
                candidate, VEHICLE_CLASSES[raw_class], width, height
            ):
                continue

            cx = (candidate[0] + candidate[2]) / 2
            cy = (candidate[1] + candidate[3]) / 2
            if not (
                tracked_box[0] <= cx <= tracked_box[2]
                and tracked_box[1] <= cy <= tracked_box[3]
            ):
                continue

            overlaps_existing = False
            for other_index, (other_box, other_class) in enumerate(
                zip(original_boxes, class_ids)
            ):
                if other_index == index or other_class not in VEHICLE_CLASSES:
                    continue
                other_box = clamp_box(other_box, width, height)
                if not is_vehicle_box_plausible(
                    other_box, VEHICLE_CLASSES[other_class], width, height
                ):
                    continue
                ix1 = max(candidate[0], other_box[0])
                iy1 = max(candidate[1], other_box[1])
                ix2 = min(candidate[2], other_box[2])
                iy2 = min(candidate[3], other_box[3])
                intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                candidate_area = (
                    (candidate[2] - candidate[0])
                    * (candidate[3] - candidate[1])
                )
                other_area = (
                    (other_box[2] - other_box[0])
                    * (other_box[3] - other_box[1])
                )
                union = candidate_area + other_area - intersection
                if union > 0 and intersection / union > 0.6:
                    overlaps_existing = True
                    break
            if overlaps_existing:
                continue

            candidates.append((np.hypot(cx - tx, cy - ty), raw_index, raw_box))

        if candidates:
            _, raw_index, replacement_box = min(candidates)
            boxes[index] = replacement_box
            used_raw_indices.add(raw_index)

    return boxes


# =========================
# 5. CHAY THUC NGHIEM, TRACKING VA GHI VIDEO
# =========================
def main():
    if not VIDEO_PATH.exists():
        print("Khong tim thay video:", VIDEO_PATH.resolve())
        return

    print("Dang tai mo hinh YOLO...")
    model = YOLO(MODEL_PATH)
    recovery_model = None
    plate_model = None
    if PLATE_MODEL_PATH.exists():
        print("Dang tai mo hinh bien so...")
        try:
            plate_model = YOLO(str(PLATE_MODEL_PATH))
        except ModuleNotFoundError as error:
            if error.name != "yolov5":
                raise
            print(
                "CHU Y: Mo hinh bien so can yolov5, khong tai duoc; "
                "van tiep tuc luu bang chung xe vi pham."
            )
    else:
        print("CHU Y: Chua co mo hinh bien so tai:", PLATE_MODEL_PATH)
        print("Video van chay, nhung chua the cat bien so tu dong.")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Khong mo duoc video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print("Khong tao duoc video dau ra.")
        return

    previous_side = {}
    tracked_vehicles = {}
    counted_ids = set()
    counts = Counter()
    events = []
    frame_number = 0

    print("Bat dau phan tich giao thong...")
    results = model.track(
        source=str(VIDEO_PATH.resolve()), tracker=str(TRACKER_CONFIG.resolve()),
        conf=CONFIDENCE, imgsz=IMAGE_SIZE, device="cpu",
        persist=True, stream=True, verbose=False
    )

    for result in results:
        frame_number += 1
        seen_track_ids = set()
        raw_frame = result.orig_img.copy()
        frame = raw_frame.copy()
        signal_state = classify_approach_signal(raw_frame)
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(class_ids)
            if any(
                class_id in VEHICLE_CLASSES
                and not is_vehicle_box_plausible(
                    clamp_box(box, width, height),
                    VEHICLE_CLASSES[class_id],
                    width,
                    height,
                )
                for box, class_id in zip(xyxy, class_ids)
            ):
                if recovery_model is None:
                    recovery_model = YOLO(MODEL_PATH)
                xyxy = recover_oversized_vehicle_boxes(
                    recovery_model, raw_frame, xyxy, class_ids, width, height
                )

            for box, class_id, track_id in zip(xyxy, class_ids, track_ids):
                if class_id not in VEHICLE_CLASSES:
                    continue

                vehicle_type = VEHICLE_CLASSES[class_id]
                vehicle_box = clamp_box(box, width, height)
                x1, y1, x2, y2 = vehicle_box
                if x2 <= x1 or y2 <= y1:
                    continue
                if not is_vehicle_box_plausible(
                    vehicle_box, vehicle_type, width, height
                ):
                    continue

                previous_track = (
                    tracked_vehicles.get(track_id) if track_id >= 0 else None
                )
                if (
                    previous_track is not None
                    and not is_track_continuous(
                        previous_track, vehicle_box, vehicle_type, frame_number
                    )
                ):
                    tracked_vehicles.pop(track_id, None)
                    previous_side.pop(track_id, None)
                    if previous_track["violation"]:
                        counted_ids.discard(track_id)
                    previous_track = None

                # Tinh vach bang tam cua bbox phuong tien, khong dung bbox ghep.
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_side = (
                    get_line_side(center_x, center_y)
                    if track_id >= 0 else None
                )
                previous_observation = (
                    previous_side.get(track_id) if track_id >= 0 else None
                )
                previous_is_recent = (
                    previous_observation is not None
                    and frame_number - previous_observation[1]
                    <= MAX_CROSSING_FRAME_GAP
                )
                old_side = (
                    previous_observation[0] if previous_is_recent else None
                )

                if current_side is not None:
                    previous_side[track_id] = (current_side, frame_number)
                elif (
                    LINE_POINTS[0][0] <= center_x <= LINE_POINTS[-1][0]
                    and previous_is_recent
                ):
                    previous_side[track_id] = (
                        previous_observation[0],
                        frame_number,
                    )
                elif not (
                    LINE_POINTS[0][0] <= center_x <= LINE_POINTS[-1][0]
                ):
                    previous_side.pop(track_id, None)

                is_red_light_violation = track_id >= 0 and (
                    is_red_light_crossing(
                        old_side, current_side, signal_state
                    )
                    and track_id not in counted_ids
                )

                display_box = vehicle_box
                if track_id >= 0:
                    velocity_x = 0.0
                    velocity_y = 0.0
                    smoothed_box = vehicle_box
                    if previous_track is not None:
                        elapsed = max(
                            1, frame_number - previous_track["last_seen"]
                        )
                        old_box = previous_track["box"]
                        smoothed_box = tuple(
                            round(0.65 * old_value + 0.35 * new_value)
                            for old_value, new_value in zip(
                                old_box, vehicle_box
                            )
                        )
                        old_center_x = (old_box[0] + old_box[2]) / 2
                        old_center_y = (old_box[1] + old_box[3]) / 2
                        velocity_x = float(np.clip(
                            (center_x - old_center_x) / elapsed,
                            -MAX_TRACK_SPEED,
                            MAX_TRACK_SPEED,
                        ))
                        velocity_y = float(np.clip(
                            (center_y - old_center_y) / elapsed,
                            -MAX_TRACK_SPEED,
                            MAX_TRACK_SPEED,
                        ))
                    tracked_vehicles[track_id] = {
                        "box": smoothed_box,
                        "vehicle_type": vehicle_type,
                        "last_seen": frame_number,
                        "velocity_x": velocity_x,
                        "velocity_y": velocity_y,
                        "violation": (
                            track_id in counted_ids
                            or is_red_light_violation
                        ),
                    }
                    seen_track_ids.add(track_id)
                    display_box = smoothed_box
                dx1, dy1, dx2, dy2 = clamp_box(display_box, width, height)

                # Giu khung do tren cac xe da ghi nhan vi pham.
                box_color = (
                    RED
                    if track_id in counted_ids or is_red_light_violation
                    else GREEN
                )

                # Ve vien mong 1px nhu anh mau.
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), box_color, 2, cv2.LINE_AA)

                # Chi hien ID va loai phuong tien, tuyet doi khong hien ID nguoi.
                short_type = {
                    "motorcycle": "MOTO",
                    "car": "CAR",
                    "bus": "BUS",
                    "truck": "TRUCK"
                }

                vehicle_label = SHORT_VEHICLE_TYPES[vehicle_type]
                label = (
                    f"ID: {track_id} | {vehicle_label}"
                    if track_id >= 0 else f"{vehicle_label} | ID: --"
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.43
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
               # Uu tien dat nhan tren khung, neu sat mep tren thi dat ben trong
                if dy1 >= text_h + baseline + 8:
                    label_top = dy1 - text_h - baseline - 5
                else:
                    label_top = min(height - text_h - baseline - 5, dy1 + 3)

                label_bottom = label_top + text_h + baseline + 5
                label_right = min(width - 1, dx1 + text_w + 8)
                cv2.rectangle(frame, (dx1, label_top), (label_right, label_bottom), box_color, -1)
                cv2.putText(frame, label, (dx1 + 4, label_bottom - baseline - 2), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

                if is_red_light_violation:
                    counted_ids.add(track_id)
                    counts[vehicle_type] += 1
                    event_number = len(events) + 1
                    event_dir = EVIDENCE_DIR / f"event_{event_number:04d}"
                    event_dir.mkdir(parents=True, exist_ok=True)

                    # Luu anh xe va anh toan canh truoc khi thu nhan dien bien so.
                    evidence_crop = raw_frame[dy1:dy2, dx1:dx2].copy()
                    vehicle_crop_path = event_dir / "vehicle_crop.jpg"
                    if not cv2.imwrite(str(vehicle_crop_path), evidence_crop):
                        raise OSError(
                            f"Khong luu duoc anh xe: {vehicle_crop_path.resolve()}"
                        )

                    evidence_frame = frame.copy()
                    cv2.putText(
                        evidence_frame,
                        f"RED LIGHT VIOLATION | SIGNAL: {signal_state} | "
                        f"FRAME: {frame_number}",
                        (10, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        RED,
                        1,
                        cv2.LINE_AA,
                    )
                    evidence_path = event_dir / "evidence.jpg"
                    if not cv2.imwrite(str(evidence_path), evidence_frame):
                        raise OSError(
                            f"Khong luu duoc anh bang chung: "
                            f"{evidence_path.resolve()}"
                        )

                    plate_zoom = save_plate_crop(
                        evidence_crop, plate_model, event_dir
                    )

                    if plate_zoom is not None:
                        panel_width = max(1, min(480, int(width * 0.4)))
                        panel_height = height
                        panel = np.full(
                            (panel_height, panel_width, 3),
                            (30, 30, 30),
                            dtype=np.uint8
                        )
                        cv2.putText(
                            panel,
                            "DETECTED LICENSE PLATE",
                            (15, min(30, panel_height - 1)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        max_w = max(1, panel_width - 30)
                        max_h = max(1, panel_height - 60)
                        scale = min(
                            max_w / plate_zoom.shape[1],
                            max_h / plate_zoom.shape[0],
                            1.0
                        )
                        plate_display = cv2.resize(
                            plate_zoom,
                            (
                                max(1, int(plate_zoom.shape[1] * scale)),
                                max(1, int(plate_zoom.shape[0] * scale))
                            ),
                            interpolation=(
                                cv2.INTER_AREA if scale < 1
                                else cv2.INTER_CUBIC
                            )
                        )
                        px = (panel_width - plate_display.shape[1]) // 2
                        py = 45 + (
                            max_h - plate_display.shape[0]
                        ) // 2
                        panel[
                            py:py + plate_display.shape[0],
                            px:px + plate_display.shape[1]
                        ] = plate_display

                        target_h = max(
                            evidence_frame.shape[0], panel.shape[0]
                        )

                        def fit_height(image, target_height):
                            new_w = max(
                                1,
                                int(
                                    image.shape[1]
                                    * target_height
                                    / image.shape[0]
                                )
                            )
                            return cv2.resize(
                                image,
                                (new_w, target_height),
                                interpolation=cv2.INTER_AREA
                            )

                        left = fit_height(evidence_frame, target_h)
                        right = fit_height(panel, target_h)
                        combined = np.hstack((left, right))
                    else:
                        combined = evidence_frame

                    if plate_zoom is not None and not cv2.imwrite(
                        str(evidence_path), combined
                    ):
                        raise OSError(
                            f"Khong cap nhat duoc anh bang chung: "
                            f"{evidence_path.resolve()}"
                        )

                    events.append({
                        "event_id": event_number,
                        "frame": frame_number,
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "violation_type": "red_light_line_crossing",
                        "signal_state": signal_state,
                        "center_x": center_x,
                        "center_y": center_y,
                        "direction": "below_to_above",
                        "evidence_image": str(evidence_path),
                        "vehicle_crop": str(vehicle_crop_path),
                        "plate_crop": (
                            "embedded_in_evidence"
                            if plate_zoom is not None else ""
                        ),
                    })
                    print(
                        f"Vi pham #{event_number}: Vehicle ID={track_id}, "
                        f"loai={vehicle_type}, frame={frame_number}, den={signal_state}"
                    )

        for track_id, state in list(tracked_vehicles.items()):
            if track_id in seen_track_ids:
                continue
            missed_frames = frame_number - state["last_seen"]
            max_missed = (
                MAX_VIOLATION_MISSED_FRAMES
                if state["violation"] else MAX_MISSED_FRAMES
            )
            if missed_frames > max_missed:
                tracked_vehicles.pop(track_id)
                previous_side.pop(track_id, None)
                if state["violation"]:
                    counted_ids.discard(track_id)
                continue

        # Hien thi thong ke va trang thai den tren tung frame dau ra.
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (15, 20, 25), -1)
        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
        cv2.putText(frame, "AI TRAFFIC MONITOR", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Car: {counts['car']}  Moto: {counts['motorcycle']}", (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Bus: {counts['bus']}  Truck: {counts['truck']}", (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Red-light violations: {len(events)}", (18, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Approach signal: {signal_state}", (18, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.4, RED if signal_state == "RED" else GREEN, 1, cv2.LINE_AA)
        writer.write(frame)

        if frame_number % 60 == 0:
            print(f"Da xu ly {frame_number} frame")

    writer.release()
    fieldnames = ["event_id", "frame", "track_id", "vehicle_type", "violation_type", "signal_state", "center_x", "center_y", "direction", "evidence_image", "vehicle_crop", "plate_crop"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(events)

    print("\n===== KET QUA PHAN TICH =====")
    for vehicle_type in VEHICLE_CLASSES.values():
        print(f"{vehicle_type}: {counts[vehicle_type]}")
    print("Tong vi pham vuot vach khi den do:", len(events))
    print("Video:", OUTPUT_VIDEO.resolve())
    print("CSV:", OUTPUT_CSV.resolve())
    print("Thu muc bang chung:", EVIDENCE_DIR.resolve())


if __name__ == "__main__":
    main()
