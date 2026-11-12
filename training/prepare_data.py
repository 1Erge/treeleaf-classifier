import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parent.parent

MANIFEST_PATH = PROJECT_ROOT / "data/manifests/flavia_manifest.csv"
DATA_ROOT = PROJECT_ROOT / "data/raw/Leaves"
OUTPUT_ROOT = PROJECT_ROOT / "data/processed"

IMAGE_SIZE = 224
SEED = 42

RATIOS = {
    "train": 0.70,
    "validation": 0.15,
    "test": 0.15,
}

REQUIRED_COLUMNS = [
    "relative_path",
    "image_id",
    "source_label",
    "class_index",
    "class_slug",
    "common_name",
    "scientific_name",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_image_path(relative_path: str) -> Path:
    """
    Поддерживает оба варианта путей:
    Leaves/1001.jpg
    1001.jpg
    """

    relative = Path(relative_path)

    candidates = [
        DATA_ROOT / relative,
        DATA_ROOT.parent / relative,
    ]

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved.is_file():
            if (
                resolved != DATA_ROOT.resolve()
                and DATA_ROOT.resolve() not in resolved.parents
            ):
                # Разрешаем путь через DATA_ROOT.parent,
                # но только внутри data/raw.
                raw_root = DATA_ROOT.parent.resolve()

                if raw_root not in resolved.parents:
                    continue

            return resolved

    raise FileNotFoundError(
        f"Изображение не найдено: {relative_path}"
    )


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Манифест не найден: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"В манифесте отсутствуют колонки: {sorted(missing)}"
        )

    if df.empty:
        raise ValueError("Манифест пустой")

    df["class_index"] = df["class_index"].astype(int)
    df["image_id"] = df["image_id"].astype(str)

    if df["relative_path"].duplicated().any():
        raise ValueError(
            "В манифесте обнаружены повторяющиеся relative_path"
        )

    if not df["class_index"].between(0, 31).all():
        raise ValueError("class_index должен находиться в диапазоне 0–31")

    return df


def calculate_hashes(df: pd.DataFrame) -> pd.DataFrame:
    hashes = []
    paths = []

    print("\nПроверка исходных изображений и SHA-256...")

    for index, row in df.iterrows():
        path = resolve_image_path(row["relative_path"])

        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(
                f"Повреждённое изображение: {path}"
            ) from error

        hashes.append(sha256_file(path))
        paths.append(str(path))

        if (index + 1) % 200 == 0:
            print(f"Проверено: {index + 1}/{len(df)}")

    df["sha256"] = hashes
    df["source_path"] = paths

    # Одинаковое содержимое не может иметь разные классы.
    conflicts = df.groupby("sha256")["class_index"].nunique()

    if (conflicts > 1).any():
        raise ValueError(
            "Обнаружены одинаковые изображения с разными метками классов"
        )

    return df


def split_class_groups(
    class_df: pd.DataFrame,
    rng: random.Random,
) -> dict:
    """
    Распределяет группы одинаковых изображений.
    Все файлы с одинаковым SHA-256 остаются в одном split.
    """

    groups = defaultdict(list)

    for _, row in class_df.iterrows():
        groups[row["sha256"]].append(row)

    group_list = list(groups.values())

    if len(group_list) < 3:
        raise ValueError(
            f"Для класса {class_df.iloc[0]['class_slug']} "
            f"недостаточно уникальных изображений для трёх выборок"
        )

    rng.shuffle(group_list)

    # Сначала распределяем крупные группы.
    group_list.sort(key=len, reverse=True)

    total = len(class_df)

    targets = {
        name: total * ratio
        for name, ratio in RATIOS.items()
    }

    counts = {
        name: 0
        for name in RATIOS
    }

    assignments = {}

    split_names = list(RATIOS.keys())

    # Гарантируем наличие изображений во всех трёх выборках.
    for index, group in enumerate(group_list[:3]):
        split = split_names[index]
        assignments[id(group)] = split
        counts[split] += len(group)

    # Остальные группы распределяем с учётом целевых размеров.
    for group in group_list[3:]:
        group_size = len(group)

        def score(split):
            candidate = dict(counts)
            candidate[split] += group_size

            return sum(
                (candidate[name] - targets[name]) ** 2
                for name in RATIOS
            )

        selected = min(split_names, key=score)

        assignments[id(group)] = selected
        counts[selected] += group_size

    result = {
        name: []
        for name in RATIOS
    }

    for group in group_list:
        split = assignments[id(group)]
        result[split].extend(group)

    return result


def prepare_image(
    source: Path,
    destination: Path,
    image_size: int,
):
    with Image.open(source) as image:
        original_width, original_height = image.size

        image = image.convert("RGB")

        image = ImageOps.pad(
            image,
            (image_size, image_size),
            method=Image.Resampling.LANCZOS,
            color=(255, 255, 255),
            centering=(0.5, 0.5),
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        image.save(
            destination,
            format="JPEG",
            quality=95,
            optimize=True,
        )

    return original_width, original_height


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if args.image_size <= 0:
        raise ValueError("Размер изображения должен быть положительным")

    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{OUTPUT_ROOT} уже существует и не пуста. "
                "Используй --overwrite для повторной подготовки."
            )

        print(f"Удаление старых данных: {OUTPUT_ROOT}")
        shutil.rmtree(OUTPUT_ROOT)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    df = load_manifest()

    print(f"Исходных изображений: {len(df)}")
    print(f"Классов: {df['class_index'].nunique()}")

    df = calculate_hashes(df)

    all_assignments = []

    for class_index, class_df in df.groupby("class_index"):
        split_groups = split_class_groups(
            class_df,
            rng,
        )

        for split_name, rows in split_groups.items():
            for row in rows:
                all_assignments.append({
                    "row": row,
                    "split": split_name,
                })

    # Дополнительная проверка отсутствия утечки между выборками.
    hash_splits = defaultdict(set)

    for item in all_assignments:
        row = item["row"]
        hash_splits[row["sha256"]].add(item["split"])

    leakage = {
        digest: splits
        for digest, splits in hash_splits.items()
        if len(splits) > 1
    }

    if leakage:
        raise RuntimeError(
            "Обнаружена утечка дубликатов между выборками"
        )

    csv_rows = []
    split_counts = {
        name: 0
        for name in RATIOS
    }

    class_distribution = defaultdict(
        lambda: {
            name: 0
            for name in RATIOS
        }
    )

    print("\nПреобразование изображений...")

    for index, item in enumerate(all_assignments, start=1):
        row = item["row"]
        split = item["split"]

        class_slug = row["class_slug"]

        filename = (
            f"{row['image_id']}_{row['sha256'][:8]}.jpg"
        )

        relative_output = (
            Path(split) / class_slug / filename
        )

        destination = OUTPUT_ROOT / relative_output

        original_width, original_height = prepare_image(
            Path(row["source_path"]),
            destination,
            args.image_size,
        )

        split_counts[split] += 1
        class_distribution[class_slug][split] += 1

        csv_rows.append({
            "split": split,
            "relative_path": row["relative_path"],
            "processed_path": str(relative_output),
            "filename": filename,
            "image_id": row["image_id"],
            "source_label": row["source_label"],
            "class_index": row["class_index"],
            "class_slug": class_slug,
            "common_name": row["common_name"],
            "scientific_name": row["scientific_name"],
            "sha256": row["sha256"],
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": args.image_size,
            "processed_height": args.image_size,
        })

        if index % 200 == 0:
            print(f"Обработано: {index}/{len(all_assignments)}")

    splits_df = pd.DataFrame(csv_rows)

    splits_df.to_csv(
        OUTPUT_ROOT / "splits.csv",
        index=False,
    )

    duplicate_groups = (
        df.groupby(["class_index", "sha256"])
        .size()
    )

    summary = {
        "dataset": "Flavia",
        "source_root": str(DATA_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "seed": args.seed,
        "image_size": args.image_size,
        "ratios": RATIOS,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "applied_during": "training_and_inference",
        },
        "source_images": len(df),
        "processed_images": len(splits_df),
        "classes": int(df["class_index"].nunique()),
        "unique_sha256_images": int(df["sha256"].nunique()),
        "duplicate_groups": int((duplicate_groups > 1).sum()),
        "duplicate_extra_files": int(
            (duplicate_groups - 1).clip(lower=0).sum()
        ),
        "split_counts": split_counts,
        "class_distribution": dict(class_distribution),
        "manifest": str(MANIFEST_PATH),
    }

    with (
        OUTPUT_ROOT / "preprocessing_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=4,
        )

    print("\n" + "=" * 50)
    print("ПОДГОТОВКА ДАТАСЕТА ЗАВЕРШЕНА")
    print("=" * 50)

    print(f"Всего изображений: {len(splits_df)}")
    print(f"Уникальных изображений: {summary['unique_sha256_images']}")
    print(f"Классов: {summary['classes']}")

    for name, count in split_counts.items():
        percentage = count / len(splits_df) * 100
        print(f"{name}: {count} ({percentage:.2f}%)")

    print(f"\nРезультат: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()