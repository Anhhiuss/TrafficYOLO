from pathlib import Path
import cv2

VIDEO_PATH = Path("videos/16h15.25.9.22.mp4")
OUTPUT_PATH = Path("outputs/frame_for_counting.jpg")

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    print("Khong mo duoc video:", VIDEO_PATH.resolve())
    raise SystemExit(1)

# Lay frame thu 180, gan giua video 384 frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
success, frame = cap.read()
cap.release()

if not success:
    print("Khong doc duoc frame.")
    raise SystemExit(1)

height, width = frame.shape[:2]

# Ve toa do de xac dinh vi tri vach dem
for x in range(0, width, 100):
    cv2.line(frame, (x, 0), (x, height - 1), (255, 180, 0), 1)
    cv2.putText(
        frame, str(x), (x + 3, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1
    )

for y in range(0, height, 100):
    cv2.line(frame, (0, y), (width - 1, y), (255, 180, 0), 1)
    cv2.putText(
        frame, str(y), (5, y + 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1
    )

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), frame)

print("Kich thuoc frame:", width, "x", height)
print("Da luu anh:", OUTPUT_PATH.resolve())
