from pathlib import Path
import cv2
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "videos" / "transfagarasan.mp4"
OUTPUT_DIR = PROJECT_ROOT / "collected_crops" / "unlabeled"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF = 0.40
ALLOWED_CLASSES = [2]   # car
FRAME_STEP = 8
MIN_BOX_W = 120
MIN_BOX_H = 120
MAX_CROPS = 3000


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    model = YOLO(YOLO_MODEL_NAME)
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("Could not open video.")
        return

    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % FRAME_STEP != 0:
            continue

        results = model.predict(
            source=frame,
            conf=YOLO_CONF,
            classes=ALLOWED_CLASSES,
            verbose=False
        )

        result = results[0]
        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.tolist()

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            w = x2 - x1
            h = y2 - y1

            if w < MIN_BOX_W or h < MIN_BOX_H:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            filename = OUTPUT_DIR / f"frame_{frame_idx:06d}_car_{i:02d}.jpg"
            cv2.imwrite(str(filename), crop)
            saved_count += 1

            if saved_count >= MAX_CROPS:
                cap.release()
                print(f"Stopped after saving {saved_count} crops.")
                return

    cap.release()
    print(f"Finished. Saved {saved_count} crops to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()