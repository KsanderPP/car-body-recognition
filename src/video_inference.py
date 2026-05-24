from pathlib import Path
import cv2
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import timm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_stage2_adapted.pth"

YOLO_MODEL_NAME = "yolov8n.pt"
IMAGE_SIZE = 224
YOLO_CONF = 0.40
CAR_CLASS_ID = 2

BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 180, 0)

MIN_BOX_W = 80
MIN_BOX_H = 80

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def classify_crop(crop_bgr, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(crop_rgb)
    x = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        pred_conf = probs[0, pred_idx].item()

    return pred_class, pred_conf


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

        label_y1 = max(0, y1 - 30)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x2, label_y2), TEXT_BG_COLOR, -1)

        cv2.putText(
            frame,
            label,
            (x1 + 5, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            TEXT_COLOR,
            2,
            cv2.LINE_AA
        )


def process_video(video_path, stop_flag=None, frame_step=4):
    if not video_path.exists():
        print(f"Video not found: {video_path}")
        return

    classifier, class_names = load_classifier(CHECKPOINT_PATH)
    detector = YOLO(YOLO_MODEL_NAME)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Could not open video.")
        return

    frame_idx = 0
    window_name = "Vehicle Detection and Body Type Classification"
    last_detections = []

    while True:
        if stop_flag is not None and stop_flag():
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        display_frame = frame.copy()

        if frame_idx % frame_step == 0:
            current_detections = []

            results = detector.predict(
                source=frame,
                conf=YOLO_CONF,
                classes=[CAR_CLASS_ID],
                verbose=False
            )

            result = results[0]

            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)

                for box in boxes:
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

                    pred_class, pred_conf = classify_crop(crop, classifier, class_names)
                    label = f"{pred_class} {pred_conf:.2f}"

                    current_detections.append({
                        "box": (x1, y1, x2, y2),
                        "label": label
                    })

            last_detections = current_detections

        draw_detections(display_frame, last_detections)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()