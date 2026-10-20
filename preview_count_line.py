
from pathlib import Path
import cv2

VIDEO_PATH = Path("videos/16h15.25.9.22.mp4")
OUTPUT_PATH = Path("outputs/count_line_preview.jpg")

# Vach dem thu nghiem
LINE_START = (270, 315)
LINE_END = (540, 315)

cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    print("Khong mo duoc video.")
    raise SystemExit(1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 180)
ok, frame = cap.read()
cap.release()

if not ok:
    print("Khong doc duoc frame.")
    raise SystemExit(1)

cv2.line(frame, LINE_START, LINE_END, (0, 0, 255), 3)

cv2.circle(frame, LINE_START, 6, (0, 255, 255), -1)
cv2.circle(frame, LINE_END, 6, (0, 255, 255), -1)

cv2.putText(
    frame,
    "COUNTING LINE - TEST",
    (LINE_START[0], LINE_START[1] - 10),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.55,
    (0, 0, 255),
    2
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), frame)

print("Da luu anh:", OUTPUT_PATH.resolve())
print("Diem dau:", LINE_START)
print("Diem cuoi:", LINE_END)
