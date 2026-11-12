
import csv
import json
import sys
from pathlib import Path

import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLITS_PATH = PROJECT_ROOT / "data" / "processed" / "splits.csv"
OUTPUT_DIR = PROJECT_ROOT / "training" / "evaluation"

API_URL = "http://127.0.0.1:8000/api/v1/predict"

BATCH_SIZE = 20
REQUEST_TIMEOUT = 300


def load_test_images():
    with open(SPLITS_PATH, encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    return [
        row for row in rows
        if row["split"] == "test"
    ]


def send_batch(batch):
    files = []

    try:
        with requests.Session() as session:
            opened_files = []

            try:
                for row in batch:
                    image_path = (
                        PROJECT_ROOT
                        / "data"
                        / "processed"
                        / row["processed_path"]
                    )

                    if not image_path.exists():
                        raise FileNotFoundError(
                            f"Image not found: {image_path}"
                        )

                    file_handle = open(image_path, "rb")
                    opened_files.append(file_handle)

                    files.append((
                        "files",
                        (
                            image_path.name,
                            file_handle,
                            "image/jpeg",
                        ),
                    ))

                response = session.post(
                    API_URL,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )

                response.raise_for_status()

                data = response.json()

                predictions = data["predictions"]

                if len(predictions) != len(batch):
                    raise ValueError(
                        f"Expected {len(batch)} predictions, "
                        f"received {len(predictions)}"
                    )

                return predictions

            finally:
                for file_handle in opened_files:
                    file_handle.close()

    except requests.exceptions.RequestException as error:
        raise RuntimeError(f"API request failed: {error}") from error


def save_csv(results, path):
    fieldnames = [
        "filename",
        "true_class",
        "predicted_class",
        "confidence",
        "correct",
    ]

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 65)
    print("TREELEAF CLASSIFIER — FULL API TEST")
    print("=" * 65)

    test_images = load_test_images()

    if not test_images:
        raise ValueError("Test dataset is empty")

    total = len(test_images)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nTest images: {total}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {total_batches}")
    print(f"API URL: {API_URL}")

    results = []

    for batch_number, start in enumerate(
        range(0, total, BATCH_SIZE),
        start=1,
    ):
        batch = test_images[start:start + BATCH_SIZE]

        print(
            f"\nBatch {batch_number}/{total_batches}: "
            f"{len(batch)} images"
        )

        predictions = send_batch(batch)

        for row, prediction in zip(batch, predictions):
            true_class = row["class_slug"]
            predicted_class = prediction["predicted_class"]
            confidence = float(prediction["confidence"])

            correct = true_class == predicted_class

            results.append({
                "filename": prediction["filename"],
                "true_class": true_class,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": correct,
            })

        batch_correct = sum(
            item["correct"] for item in results[-len(batch):]
        )

        print(
            f"  Correct: {batch_correct}/{len(batch)} "
            f"({batch_correct / len(batch):.2%})"
        )

    if len(results) != total:
        raise RuntimeError(
            f"Expected {total} results, got {len(results)}"
        )

    y_true = [item["true_class"] for item in results]
    y_pred = [item["predicted_class"] for item in results]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    average_confidence = sum(
        item["confidence"] for item in results
    ) / total

    correct_results = [
        item for item in results if item["correct"]
    ]

    incorrect_results = [
        item for item in results if not item["correct"]
    ]

    average_correct_confidence = (
        sum(item["confidence"] for item in correct_results)
        / len(correct_results)
        if correct_results else 0
    )

    average_incorrect_confidence = (
        sum(item["confidence"] for item in incorrect_results)
        / len(incorrect_results)
        if incorrect_results else 0
    )

    labels = sorted(set(y_true) | set(y_pred))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "api_test_results.csv"
    json_path = OUTPUT_DIR / "api_test_summary.json"
    report_path = OUTPUT_DIR / "api_classification_report.json"
    matrix_path = OUTPUT_DIR / "api_confusion_matrix.csv"

    save_csv(results, csv_path)

    summary = {
        "total_images": total,
        "correct": len(correct_results),
        "incorrect": len(incorrect_results),
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "average_confidence": average_confidence,
        "average_correct_confidence": average_correct_confidence,
        "average_incorrect_confidence": average_incorrect_confidence,
        "batch_size": BATCH_SIZE,
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4, ensure_ascii=False)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    with open(
        matrix_path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true_class/predicted_class", *labels])

        for label, row in zip(labels, matrix):
            writer.writerow([label, *row.tolist()])

    print("\n" + "=" * 65)
    print("FINAL API EVALUATION")
    print("=" * 65)

    print(f"Total images:                  {total}")
    print(f"Correct:                       {len(correct_results)}")
    print(f"Incorrect:                     {len(incorrect_results)}")
    print(f"Accuracy:                      {accuracy:.4%}")
    print(f"Macro precision:               {precision:.4%}")
    print(f"Macro recall:                  {recall:.4%}")
    print(f"Macro F1:                      {f1:.4%}")
    print(f"Weighted F1:                   {weighted_f1:.4%}")
    print(f"Average confidence:            {average_confidence:.4%}")
    print(f"Average correct confidence:    {average_correct_confidence:.4%}")
    print(f"Average incorrect confidence:  {average_incorrect_confidence:.4%}")

    if incorrect_results:
        print("\nMISCLASSIFIED IMAGES")
        print("-" * 65)

        for item in incorrect_results:
            print(
                f"{item['filename']}: "
                f"{item['true_class']} -> "
                f"{item['predicted_class']} "
                f"({item['confidence']:.2%})"
            )

    print("\nREPORTS SAVED:")
    print(csv_path)
    print(json_path)
    print(report_path)
    print(matrix_path)


if __name__ == "__main__":
    main()
