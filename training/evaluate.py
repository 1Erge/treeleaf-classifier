
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "training" / "models" / "best_model.pth"

OUTPUT_DIR = PROJECT_ROOT / "training" / "evaluation"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

DEVICE = torch.device("cpu")


def create_directories():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    class_to_idx = checkpoint["class_to_idx"]

    model = resnet50(weights=None)
    model.fc = nn.Linear(
        model.fc.in_features,
        len(class_to_idx),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(
        f"Best validation accuracy: "
        f"{checkpoint['best_val_accuracy']:.4f}"
    )

    return model, class_to_idx, checkpoint


def create_test_loader(expected_class_to_idx):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=transform,
    )

    if dataset.class_to_idx != expected_class_to_idx:
        raise ValueError("Test class mapping differs from checkpoint.")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return dataset, loader


def evaluate(model, loader, dataset, class_to_idx):
    all_predictions = []
    all_labels = []
    all_probabilities = []

    misclassified = []

    idx_to_class = {
        index: name
        for name, index in class_to_idx.items()
    }

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)

            probabilities = torch.softmax(outputs, dim=1)

            confidences, predictions = probabilities.max(dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probabilities.extend(
                probabilities.cpu().numpy().tolist()
            )

            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predictions[i].item()

                if true_label != predicted_label:
                    image_index = (
                        len(all_labels) - len(labels) + i
                    )

                    image_path, _ = dataset.samples[image_index]

                    misclassified.append({
                        "image": str(
                            Path(image_path).relative_to(PROJECT_ROOT)
                        ),
                        "true_class": idx_to_class[true_label],
                        "predicted_class": idx_to_class[predicted_label],
                        "confidence": float(
                            confidences[i].item()
                        ),
                    })

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
        misclassified,
    )


def save_results(
    labels,
    predictions,
    probabilities,
    misclassified,
    class_to_idx,
    checkpoint,
):
    class_names = [
        name
        for name, index in sorted(
            class_to_idx.items(),
            key=lambda item: item[1],
        )
    ]

    accuracy = accuracy_score(labels, predictions)

    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(class_names))),
    )

    # Classification report.
    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        OUTPUT_DIR / "classification_report.csv",
        encoding="utf-8-sig",
    )

    with open(
        OUTPUT_DIR / "classification_report.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # Confusion matrix.
    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )

    matrix_df.to_csv(
        OUTPUT_DIR / "confusion_matrix.csv",
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(20, 18))

    plt.imshow(matrix, interpolation="nearest")
    plt.title("ResNet50 - Confusion Matrix")
    plt.colorbar()

    positions = np.arange(len(class_names))

    plt.xticks(
        positions,
        class_names,
        rotation=90,
        fontsize=8,
    )

    plt.yticks(
        positions,
        class_names,
        fontsize=8,
    )

    plt.xlabel("Predicted class")
    plt.ylabel("True class")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "confusion_matrix.png",
        dpi=200,
    )

    plt.close()

    # Misclassified images.
    misclassified_df = pd.DataFrame(misclassified)

    misclassified_df.to_csv(
        OUTPUT_DIR / "misclassified_images.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Evaluation summary.
    summary = {
        "architecture": "ResNet50",
        "checkpoint_epoch": checkpoint["epoch"],
        "best_validation_accuracy": checkpoint[
            "best_val_accuracy"
        ],
        "test_images": int(len(labels)),
        "correct_predictions": int(
            (labels == predictions).sum()
        ),
        "incorrect_predictions": int(
            (labels != predictions).sum()
        ),
        "test_accuracy": float(accuracy),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    with open(
        OUTPUT_DIR / "evaluation_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nCLASSIFICATION REPORT")
    print(report_df.round(4).to_string())

    print("\nMISCLASSIFIED IMAGES")

    if misclassified:
        for item in misclassified:
            print(
                f"{item['image']}: "
                f"{item['true_class']} -> "
                f"{item['predicted_class']} "
                f"(confidence: {item['confidence']:.4f})"
            )
    else:
        print("No misclassified images.")

    print(f"\nResults saved to: {OUTPUT_DIR}")


def main():
    create_directories()

    print("Loading best model...")

    model, class_to_idx, checkpoint = load_model()

    print("\nLoading test dataset...")

    dataset, loader = create_test_loader(class_to_idx)

    print(f"Test images: {len(dataset)}")
    print(f"Classes: {len(class_to_idx)}")

    print("\nEvaluating model...")

    labels, predictions, probabilities, misclassified = evaluate(
        model,
        loader,
        dataset,
        class_to_idx,
    )

    save_results(
        labels,
        predictions,
        probabilities,
        misclassified,
        class_to_idx,
        checkpoint,
    )


if __name__ == "__main__":
    main()
