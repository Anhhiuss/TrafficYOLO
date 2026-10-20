import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# 1. CAU HINH DU AN
# =========================
VIDEO_PATH = Path("videos/16h15.25.9.22.mp4")
OUTPUT_DIR = Path("outputs")
EVIDENCE_DIR = OUTPUT_DIR / "evidence"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "counted_video.mp4"
OUTPUT_CSV = OUTPUT_DIR / "counted_vehicles.csv"
MODEL_PATH = "yolov8n.pt"
PLATE_MODEL_PATH = Path("models/license_plate_detector.pt")

CONFIDENCE = 0.35
IMAGE_SIZE = 640

# Vach chi dung cho logic dem, khong ve len video.
LINE_START = (270, 310)
LINE_END = (540, 310)

BELOW_THRESHOLD = 310
ABOVE_THRESHOLD = 305

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
PERSON_CLASS = 0  # COCO class: person; khong hien ID nguoi
GREEN = (50, 220, 50)  # BGR
RED = (0, 0, 255)      # BGR


def associate_rider_box(vehicle_box, person_boxes):
    """Ghep nguoi gan xe may de ve mot khung bao ca nguoi va xe.

    Chi dung toa do nguoi, khong dung/hien ID theo doi cua nguoi.
    Day la ghep gan dung theo vi tri, khong phai nhan dien quan he tuyet doi.
    """
    vx1, vy1, vx2, vy2 = vehicle_box
    vw = max(1, vx2 - vx1)
    vehicle_cx = (vx1 + vx2) / 2
    best_person = None
    best_score = float("inf")

    for person in person_boxes:
        px1, py1, px2, py2 = person
        person_cx = (px1 + px2) / 2
        person_bottom = py2

        # Nguoi can nam gan theo chieu ngang va o phia tren/gan xe.
        horizontal_limit = max(35, vw * 0.45)
        vertical_ok = (vy1 - 150) <= person_bottom <= (vy2 + 35)
        horizontal_ok = abs(person_cx - vehicle_cx) <= horizontal_limit
        if not (vertical_ok and horizontal_ok):
            continue

        # Uu tien nguoi gan tam xe va co chan gan phan tren cua xe.
        score = abs(person_cx - vehicle_cx) + abs(person_bottom - vy1) * 0.25
        if score < best_score:
            best_score = score
            best_person = person

    if best_person is None:
        return vehicle_box

    px1, py1, px2, py2 = best_person
    return (
        min(vx1, px1), min(vy1, py1),
        max(vx2, px2), max(vy2, py2),
    )


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


def main():
    if not VIDEO_PATH.exists():
        print("Khong tim thay video:", VIDEO_PATH.resolve())
        return

    print("Dang tai mo hinh YOLO...")
    model = YOLO(MODEL_PATH)
    plate_model = None
    if PLATE_MODEL_PATH.exists():
        print("Dang tai mo hinh bien so...")
        plate_model = YOLO(str(PLATE_MODEL_PATH))
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
    counted_ids = set()
    counts = Counter()
    events = []
    frame_number = 0

    # Luu lich su vi tri cua tung xe de ve quy dao
    trajectories = {}
    MAX_TRAIL_LENGTH = 5

    print("Bat dau phan tich giao thong...")
    results = model.track(
        source=str(VIDEO_PATH.resolve()), tracker="bytetrack.yaml",
        conf=CONFIDENCE, imgsz=IMAGE_SIZE, device="cpu",
        persist=True, stream=True, channels_last=False, verbose=False
    )

    for result in results:
        frame_number += 1
        raw_frame = result.orig_img.copy()
        frame = raw_frame.copy()
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [-1] * len(class_ids)

            # Lay bbox nguoi de ghep voi xe may. Khong lay/hien ID nguoi.
            person_boxes = [
                clamp_box(box, width, height)
                for box, cls_id in zip(xyxy, class_ids)
                if cls_id == PERSON_CLASS
            ]

            for box, class_id, track_id in zip(xyxy, class_ids, track_ids):
                if class_id not in VEHICLE_CLASSES or track_id < 0:
                    continue

                vehicle_type = VEHICLE_CLASSES[class_id]
                vehicle_box = clamp_box(box, width, height)
                x1, y1, x2, y2 = vehicle_box
                if x2 <= x1 or y2 <= y1:
                    continue

                # Tinh vach bang tam cua bbox phuong tien, khong dung bbox ghep.
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                # Cap nhat quy dao theo ID xe
                if track_id not in trajectories:
                    trajectories[track_id] = []

                trail = trajectories[track_id]
                trail.append((center_x, center_y))

                # Chi giu 20 vi tri gan nhat
                if len(trail) > MAX_TRAIL_LENGTH:
                    del trail[:-MAX_TRAIL_LENGTH]

                # Ve quy dao bang cac doan mong
                for i in range(1, len(trail)):
                    cv2.line(
                        frame,
                        trail[i - 1],
                        trail[i],
                        (255, 180, 0),
                        1,
                        cv2.LINE_AA
                    )
                if center_y >= BELOW_THRESHOLD:
                    current_side = "below"
                elif center_y <= ABOVE_THRESHOLD:
                    current_side = "above"
                else:
                    current_side = None

                old_side = previous_side.get(track_id)
                if current_side is not None:
                    previous_side[track_id] = current_side
                crossed_line = old_side == "below" and current_side == "above"
                inside_line_range = min(LINE_START[0], LINE_END[0]) <= center_x <= max(LINE_START[0], LINE_END[0])
                is_new_event = crossed_line and inside_line_range and track_id not in counted_ids

                # Ghep bbox nguoi + xe may thanh mot khung chung.
                display_box = associate_rider_box(vehicle_box, person_boxes) if class_id == 3 else vehicle_box
                dx1, dy1, dx2, dy2 = clamp_box(display_box, width, height)
                # Chi danh dau do trong khung hinh xe vua vuot vach
                box_color = RED if is_new_event else GREEN

                # Ve vien mong 1px nhu anh mau.
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), box_color, 1, cv2.LINE_AA)

                # Chi hien ID va loai phuong tien, tuyet doi khong hien ID nguoi.
                short_type = {
                    "motorcycle": "MOTO",
                    "car": "CAR",
                    "bus": "BUS",
                    "truck": "TRUCK"
                }

                label = f"ID: {track_id} | {short_type[vehicle_type]}"
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

                if is_new_event:
                    counted_ids.add(track_id)
                    counts[vehicle_type] += 1
                    event_number = len(events) + 1
                    event_dir = EVIDENCE_DIR / f"event_{event_number:04d}"
                    event_dir.mkdir(parents=True, exist_ok=True)

                    # Luu crop gom ca nguoi + xe neu da ghep duoc.
                    evidence_crop = raw_frame[dy1:dy2, dx1:dx2].copy()
                    vehicle_crop_path = event_dir / "vehicle_crop.jpg"
                    cv2.imwrite(str(vehicle_crop_path), evidence_crop)
                    plate_zoom = save_plate_crop(
                        evidence_crop, plate_model, event_dir
                    )

                    evidence_frame = frame.copy()
                    cv2.putText(evidence_frame, "LINE CROSSING EVENT", (10, height - 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1, cv2.LINE_AA)

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

                    evidence_path = event_dir / "evidence.jpg"
                    cv2.imwrite(str(evidence_path), combined)

                    events.append({
                        "event_id": event_number,
                        "frame": frame_number,
                        "track_id": track_id,
                        "vehicle_type": vehicle_type,
                        "violation_type": "virtual_line_crossing",
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
                    print(f"Su kien #{event_number}: Vehicle ID={track_id}, loai={vehicle_type}, frame={frame_number}")

        # Bang thong ke nho, khong ve vach ao len video.
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (275, 101), (15, 20, 25), -1)
        frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)
        cv2.putText(frame, "AI TRAFFIC MONITOR", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Car: {counts['car']}  Moto: {counts['motorcycle']}", (18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Bus: {counts['bus']}  Truck: {counts['truck']}", (18, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GREEN, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Line events: {len(counted_ids)}", (18, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)

        if frame_number % 60 == 0:
            print(f"Da xu ly {frame_number} frame")

    writer.release()
    fieldnames = ["event_id", "frame", "track_id", "vehicle_type", "violation_type", "center_x", "center_y", "direction", "evidence_image", "vehicle_crop", "plate_crop"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as file:
        csv_writer = csv.DictWriter(file, fieldnames=fieldnames)
        csv_writer.writeheader()
        csv_writer.writerows(events)

    print("\n===== KET QUA PHAN TICH =====")
    for vehicle_type in VEHICLE_CLASSES.values():
        print(f"{vehicle_type}: {counts[vehicle_type]}")
    print("Tong su kien vuot vach:", len(counted_ids))
    print("Video:", OUTPUT_VIDEO.resolve())
    print("CSV:", OUTPUT_CSV.resolve())
    print("Thu muc bang chung:", EVIDENCE_DIR.resolve())


if __name__ == "__main__":
    main()
