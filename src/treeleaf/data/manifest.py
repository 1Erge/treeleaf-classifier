"""Построение CSV-манифеста и JSON-отчёта по Flavia."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from .classes import FLAVIA_CLASSES
from .loader import FlaviaLoader


CSV_FIELDS = [
    "relative_path",
    "filename",
    "image_id",
    "source_label",
    "class_index",
    "class_slug",
    "common_name",
    "scientific_name",
    "width",
    "height",
    "image_format",
    "size_bytes",
]


def build_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    summary_path: str | Path,
) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    manifest_file = Path(manifest_path)
    summary_file = Path(summary_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    loader = FlaviaLoader(root)
    class_counts: Counter[str] = Counter()
    invalid_files: list[dict] = []
    total_files = 0
    valid_files = 0

    # Сканирование и проверка изображений. Для CSV используется временный файл,
    # чтобы не держать весь датасет в оперативной памяти.
    with manifest_file.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                continue

            total_files += 1
            image_id = loader._parse_image_id(path)
            if image_id is None:
                invalid_files.append({
                    "file": path.relative_to(root).as_posix(),
                    "reason": "Имя файла не начинается с четырёхзначного ID",
                })
                continue

            from .classes import get_class_by_image_id
            class_info = get_class_by_image_id(image_id)
            if class_info is None:
                invalid_files.append({
                    "file": path.relative_to(root).as_posix(),
                    "reason": f"Неизвестный ID изображения: {image_id}",
                })
                continue

            try:
                from PIL import Image, UnidentifiedImageError
                with Image.open(path) as image:
                    width, height = image.size
                    image_format = image.format or path.suffix.lstrip(".").upper()
                    image.verify()
                if width <= 0 or height <= 0:
                    raise ValueError("Некорректные размеры изображения")
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                invalid_files.append({
                    "file": path.relative_to(root).as_posix(),
                    "reason": str(exc),
                })
                continue

            writer.writerow({
                "relative_path": path.relative_to(root).as_posix(),
                "filename": path.name,
                "image_id": image_id,
                "source_label": class_info.source_label,
                "class_index": class_info.class_index,
                "class_slug": class_info.slug,
                "common_name": class_info.common_name,
                "scientific_name": class_info.scientific_name,
                "width": width,
                "height": height,
                "image_format": image_format,
                "size_bytes": path.stat().st_size,
            })
            valid_files += 1
            class_counts[class_info.slug] += 1

    classes_report = []
    for class_info in FLAVIA_CLASSES:
        classes_report.append({
            "class_index": class_info.class_index,
            "source_label": class_info.source_label,
            "slug": class_info.slug,
            "common_name": class_info.common_name,
            "scientific_name": class_info.scientific_name,
            "image_id_range": [
                class_info.image_id_start,
                class_info.image_id_end,
            ],
            "images": class_counts.get(class_info.slug, 0),
        })

    summary = {
        "dataset": "Flavia",
        "dataset_root": str(root),
        "expected_classes": 32,
        "found_classes_with_images": sum(
            1 for count in class_counts.values() if count > 0
        ),
        "total_image_files": total_files,
        "valid_labeled_images": valid_files,
        "invalid_or_unrecognized_files": len(invalid_files),
        "expected_approximate_images": 1907,
        "class_distribution": classes_report,
        "invalid_files": invalid_files,
        "manifest": str(manifest_file),
    }

    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
