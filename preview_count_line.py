
from pathlib import Path
import cv2

VIDEO_PATH = Path("videos/3.mp4")
OUTPUT_PATH = Path("outputs/count_line_preview.jpg")

# Same straight line coordinates as the counting scripts (1600 x 1200).
LINE_START = (0, 925)
LINE_END = (1400, 750)

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

line_preview = frame.copy()
cv2.line(
    line_preview, LINE_START, LINE_END, (0, 0, 255), 5, cv2.LINE_AA
)
frame = cv2.addWeighted(line_preview, 0.22, frame, 0.78, 0)

cv2.putText(
    frame,
    "VIRTUAL COUNTING LINE",
    (20, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 0, 255),
    1,
    cv2.LINE_AA
)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT_PATH), frame)

print("Da luu anh:", OUTPUT_PATH.resolve())
print("Diem dau:", LINE_START)
print("Diem cuoi:", LINE_END)
