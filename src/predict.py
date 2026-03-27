from pathlib import Path
import sys

import torch
from PIL import Image
from torchvision import transforms
import timm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best_stage2.pth"

IMAGE_SIZE = 224
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path):
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


def predict_image(image_path, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_class = class_names[pred_idx]
        pred_conf = probs[0, pred_idx].item()

    return pred_class, pred_conf, probs[0].cpu().numpy()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <path_to_image>")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    model, class_names = load_model(CHECKPOINT_PATH)
    pred_class, pred_conf, probs = predict_image(image_path, model, class_names)

    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {pred_conf:.4f}")
    print("\nAll class probabilities:")

    for cls, prob in zip(class_names, probs):
        print(f"{cls}: {prob:.4f}")