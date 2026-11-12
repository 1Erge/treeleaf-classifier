from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Позволяет запускать скрипт из корня репозитория без pip install -e .
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from treeleaf.data.manifest import build_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Проверка Flavia и создание CSV/JSON манифестов."
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/Leaves"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/flavia_manifest.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("data/manifests/flavia_summary.json"),
    )
    args = parser.parse_args()

    summary = build_manifest(
        dataset_root=args.root,
        manifest_path=args.manifest,
        summary_path=args.summary,
    )

    print("\n=== Flavia dataset report ===")
    print(f"Всего файлов изображений: {summary['total_image_files']}")
    print(f"Корректных размеченных изображений: {summary['valid_labeled_images']}")
    print(f"Ошибочных/неопознанных файлов: {summary['invalid_or_unrecognized_files']}")
    print(f"Классов с изображениями: {summary['found_classes_with_images']}/32")
    print("\nРаспределение по классам:")
    for item in summary["class_distribution"]:
        print(
            f"{item['class_index']:02d} | "
            f"{item['slug']:<30} | {item['images']:>3}"
        )

    print(f"\nCSV: {args.manifest}")
    print(f"JSON: {args.summary}")

    if summary["valid_labeled_images"] == 0:
        print(
            "\nНе найдено ни одного корректного изображения. "
            "Проверьте путь и структуру распакованного архива.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
