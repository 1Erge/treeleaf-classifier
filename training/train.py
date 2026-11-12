
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

MODEL_DIR = PROJECT_ROOT / "training" / "models"
LOG_DIR = PROJECT_ROOT / "training" / "logs"
PLOT_DIR = PROJECT_ROOT / "training" / "plots"

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_WORKERS = 2

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 5

HEAD_LR = 0.001
FINE_TUNE_LR = 0.0001

SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DIRECTORIES
# ============================================================

def create_directories():
    for directory in [MODEL_DIR, LOG_DIR, PLOT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

def create_datasets():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std,
        ),
    ])

    train_dataset = datasets.ImageFolder(
        DATA_DIR / "train",
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        DATA_DIR / "validation",
        transform=eval_transform,
    )

    test_dataset = datasets.ImageFolder(
        DATA_DIR / "test",
        transform=eval_transform,
    )

    # All splits must use exactly the same class mapping.
    class_to_idx = train_dataset.class_to_idx

    if val_dataset.class_to_idx != class_to_idx:
        raise ValueError("Validation class mapping differs from train.")

    if test_dataset.class_to_idx != class_to_idx:
        raise ValueError("Test class mapping differs from train.")

    if len(class_to_idx) != 32:
        raise ValueError(
            f"Expected 32 classes, found {len(class_to_idx)}"
        )

    return train_dataset, val_dataset, test_dataset, class_to_idx


def create_dataloaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# MODEL
# ============================================================

def create_model(num_classes: int):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)

    # Freeze backbone.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace classifier.
    in_features = model.fc.in_features

    model.fc = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)


def unfreeze_last_layer(model):
    # Unfreeze layer4 and classifier.
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    for parameter in model.fc.parameters():
        parameter.requires_grad = True


# ============================================================
# TRAINING / VALIDATION
# ============================================================

def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None

    if training:
        model.train()

        # Keep frozen BatchNorm layers in evaluation mode.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if not any(
                    parameter.requires_grad
                    for parameter in module.parameters()
                ):
                    module.eval()

    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(loader, leave=False)

    for images, labels in progress:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }


# ============================================================
# CHECKPOINTS
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_accuracy,
    class_to_idx,
    filename,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_accuracy,
        "class_to_idx": class_to_idx,
        "image_size": IMAGE_SIZE,
        "architecture": "resnet50",
    }

    torch.save(checkpoint, MODEL_DIR / filename)


# ============================================================
# TRAINING PHASE
# ============================================================

def train_phase(
    model,
    train_loader,
    val_loader,
    class_to_idx,
    epochs,
    learning_rate,
    phase_name,
    history,
    start_epoch=1,
):
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )

    best_accuracy = 0.0

    print(f"\n{'=' * 60}")
    print(f"PHASE: {phase_name}")
    print(f"{'=' * 60}")

    for epoch in range(epochs):
        epoch_number = start_epoch + epoch

        print(
            f"\nEpoch {epoch_number} | {phase_name}"
        )

        start_time = time.time()

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
        )

        elapsed = time.time() - start_time

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Accuracy: {train_metrics['accuracy']:.4f}"
        )

        print(
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f}"
        )

        print(f"Time: {elapsed:.1f} sec")

        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = val_metrics["accuracy"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch_number,
                best_accuracy=best_accuracy,
                class_to_idx=class_to_idx,
                filename="best_model.pth",
            )

            print("Best model saved!")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_number,
            best_accuracy=best_accuracy,
            class_to_idx=class_to_idx,
            filename="last_model.pth",
        )

    return best_accuracy


# ============================================================
# PLOTS AND LOGS
# ============================================================

def save_history(history):
    history_path = LOG_DIR / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(
            history,
            file,
            indent=4,
            ensure_ascii=False,
        )

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train")
    plt.plot(epochs, history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / "training_metrics.png",
        dpi=150,
    )

    plt.close()


# ============================================================
# TEST EVALUATION
# ============================================================

def evaluate_test(model, test_loader):
    checkpoint = torch.load(
        MODEL_DIR / "best_model.pth",
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    metrics = run_epoch(
        model,
        test_loader,
        criterion,
    )

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    print(f"Test Loss: {metrics['loss']:.4f}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()
    create_directories()

    print("=" * 60)
    print("TREELEAF CLASSIFIER - RESNET50 TRAINING")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")

    train_dataset, val_dataset, test_dataset, class_to_idx = (
        create_datasets()
    )

    print("\nDataset information:")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    print(f"Classes: {len(class_to_idx)}")

    with open(
        LOG_DIR / "class_to_idx.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_to_idx,
            file,
            indent=4,
            ensure_ascii=False,
        )

    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    model = create_model(len(class_to_idx))

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Phase 1: Train classifier head.
    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        epochs=HEAD_EPOCHS,
        learning_rate=HEAD_LR,
        phase_name="CLASSIFIER HEAD",
        history=history,
    )

    # Phase 2: Fine-tune layer4 and classifier.
    unfreeze_last_layer(model)

    train_phase(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_to_idx=class_to_idx,
        epochs=FINE_TUNE_EPOCHS,
        learning_rate=FINE_TUNE_LR,
        phase_name="FINE-TUNING",
        history=history,
        start_epoch=HEAD_EPOCHS + 1,
    )

    save_history(history)

    test_metrics = evaluate_test(
        model,
        test_loader,
    )

    with open(
        LOG_DIR / "test_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            test_metrics,
            file,
            indent=4,
        )

    print("\nTraining completed!")
    print(f"Models: {MODEL_DIR}")
    print(f"Logs: {LOG_DIR}")
    print(f"Plots: {PLOT_DIR}")


if __name__ == "__main__":
    main()
