from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODEL_PATHS = [Path("final_deepfake_model.pth"), Path("src") / "forensic_model.pth"]
CLASS_NAMES = ["manipulated", "real"]


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_model(device: torch.device, weights_path: Path | None = None) -> nn.Module:
    if weights_path is None:
        for candidate in MODEL_PATHS:
            if candidate.exists():
                weights_path = candidate
                break

    if weights_path is None or not weights_path.exists():
        raise FileNotFoundError(
            "Could not find a checkpoint. Expected final_deepfake_model.pth or src/forensic_model.pth."
        )

    model = build_model()
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {weights_path}")
    return model


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def predict_pil_image(
    model: nn.Module,
    image: Image.Image,
    device: torch.device,
    transform: transforms.Compose,
) -> tuple[str, float]:
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        pred_index = int(torch.argmax(probabilities).item())

    label = CLASS_NAMES[pred_index]
    confidence = float(probabilities[pred_index].item())
    return label, confidence


def predict_image(model: nn.Module, image_path: Path, device: torch.device, transform: transforms.Compose) -> tuple[str, float]:
    image = Image.open(image_path).convert("RGB")
    return predict_pil_image(model, image, device, transform)


def iter_images(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deepfake inference on images in a folder.")
    parser.add_argument("--input-dir", default="test-run", help="Folder containing images to classify.")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional checkpoint path. Defaults to final_deepfake_model.pth or src/forensic_model.pth.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    weights_path = Path(args.weights) if args.weights else None
    model = load_model(device, weights_path)
    transform = build_transform()

    image_paths = list(iter_images(input_dir))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return 1

    print(f"Scanning {len(image_paths)} image(s) in {input_dir}")
    print("-" * 72)

    for image_path in image_paths:
        label, confidence = predict_image(model, image_path, device, transform)
        verdict = "FAKE" if label == "manipulated" else "REAL"
        print(f"{image_path.name:<30} {verdict:<5}  ({label}, {confidence * 100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())