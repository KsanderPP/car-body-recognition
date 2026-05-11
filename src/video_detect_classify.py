from pathlib import Path
from collections import defaultdict

import cv2
import torch
import timm
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_PATH = PROJECT_ROOT / "videos" / "transfagarasan.mp4"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_stage2.pth"
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "outputs" / "videos" / "transfagarasan_tracked.mp4"

OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_MODEL_NAME = "yolov8n.pt"
YOLO_CONF = 0.35
ALLOWED_CLASSES = [2]   # tylko car
MIN_BOX_W = 90
MIN_BOX_H = 90

CLASSIFY_EVERY_N_FRAMES = 8
MIN_CLASSIFY_CONF = 0.60
TRACK_FORGET_AFTER = 40
IMAGE_SIZE = 224


def load_classifier(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, class_names


def classify_crop(frame_bgr, box, model, class_names):
    x1, y1, x2, y2 = box

    crop_bgr = frame_bgr[y1:y2, x1:x2]
    if crop_bgr.size == 0:
        return None, None

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        pred_conf = probs[0, pred_idx].item()

    return pred_class, pred_conf


def main():
    if not VIDEO_PATH.exists():
        print(f"Video not found: {VIDEO_PATH}")
        return

    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return

    print(f"Using device: {device}")
    print(f"Input video: {VIDEO_PATH}")
    print(f"Output video: {OUTPUT_VIDEO_PATH}")

    yolo_model = YOLO(YOLO_MODEL_NAME)
    classifier_model, class_names = load_classifier(CHECKPOINT_PATH)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    delay_ms = max(1, int(1000 / fps))

    track_memory = defaultdict(lambda: {
        "label": None,
        "conf": 0.0,
        "last_classified_frame": -999,
        "last_seen_frame": -999,
        "votes": defaultdict(float),
    })

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        results = yolo_model.track(
            frame,
            persist=True,
            conf=YOLO_CONF,
            classes=ALLOWED_CLASSES,
            verbose=False
        )

        result = results[0]

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = result.boxes.id.int().cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()

            for box, track_id, det_conf in zip(boxes, track_ids, confs):
                x1, y1, x2, y2 = box.tolist()

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(width, x2)
                y2 = min(height, y2)

                w = x2 - x1
                h = y2 - y1

                if w < MIN_BOX_W or h < MIN_BOX_H:
                    continue

                mem = track_memory[track_id]
                mem["last_seen_frame"] = frame_idx

                should_classify = (
                    frame_idx - mem["last_classified_frame"] >= CLASSIFY_EVERY_N_FRAMES
                )

                if should_classify:
                    pred_class, pred_conf = classify_crop(
                        frame,
                        (x1, y1, x2, y2),
                        classifier_model,
                        class_names
                    )

                    mem["last_classified_frame"] = frame_idx

                    if pred_class is not None and pred_conf is not None and pred_conf >= MIN_CLASSIFY_CONF:
                        mem["votes"][pred_class] += pred_conf

                        best_label = max(mem["votes"], key=mem["votes"].get)
                        total_votes = sum(mem["votes"].values())
                        best_score = mem["votes"][best_label] / total_votes if total_votes > 0 else 0.0

                        mem["label"] = best_label
                        mem["conf"] = best_score

                label = mem["label"]
                label_conf = mem["conf"]

                color = (0, 255, 0)
                text = f"ID {track_id}"

                if label is not None:
                    text += f" | {label} {label_conf:.2f}"
                else:
                    text += " | car"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    text,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    color,
                    2,
                    cv2.LINE_AA
                )

        stale_ids = [
            tid for tid, mem in track_memory.items()
            if frame_idx - mem["last_seen_frame"] > TRACK_FORGET_AFTER
        ]
        for tid in stale_ids:
            del track_memory[tid]

        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Vehicle detection + body type tracking", frame)
        writer.write(frame)

        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord("q"):
            print("Stopped by user.")
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print("Processing finished.")
    print(f"Saved result to: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()