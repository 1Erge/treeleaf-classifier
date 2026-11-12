
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT / "training" / "models" / "best_model.pth"
)


class LeafPredictor:
    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        if checkpoint["architecture"] != "resnet50":
            raise ValueError(
                f"Unsupported architecture: {checkpoint['architecture']}"
            )

        self.image_size = checkpoint["image_size"]
        self.class_to_idx = checkpoint["class_to_idx"]

        self.idx_to_class = {
            index: class_name
            for class_name, index in self.class_to_idx.items()
        }

        self.num_classes = len(self.class_to_idx)

        if set(self.idx_to_class.keys()) != set(range(self.num_classes)):
            raise ValueError("Invalid class mapping in checkpoint")

        self.model = models.resnet50(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            self.num_classes,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"Model loaded successfully: {checkpoint_path}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.num_classes}")
        print(f"Image size: {self.image_size}")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")

        image = ImageOps.pad(
            image,
            (self.image_size, self.image_size),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
        )

        tensor = self.transform(image)

        return tensor.unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        input_tensor = self.preprocess(image)

        logits = self.model(input_tensor)

        probabilities = torch.softmax(logits, dim=1)[0]

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(3, self.num_classes),
        )

        top_predictions = []

        for probability, index in zip(
            top_probabilities.tolist(),
            top_indices.tolist(),
        ):
            top_predictions.append({
                "class_name": self.idx_to_class[index],
                "probability": float(probability),
            })

        predicted_class = top_predictions[0]

        return {
            "predicted_class": predicted_class["class_name"],
            "confidence": predicted_class["probability"],
            "top_predictions": top_predictions,
        }
