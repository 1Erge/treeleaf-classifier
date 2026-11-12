from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

from tqdm import tqdm


URL = (
    "https://sourceforge.net/projects/flavia/files/"
    "Leaf%20Image%20Dataset/1.0/Leaves.tar.bz2/download"
)
EXPECTED_SHA256 = "0a35d5bad8342406655af5550008f750b596652e687316344e2cdeb8fd65d895"
DEFAULT_OUTPUT = Path("data/downloads/Leaves.tar.bz2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "TreeLeafClassifier/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        total = int(response.headers.get("Content-Length", 0))
        temp_path = destination.with_suffix(destination.suffix + ".part")

        try:
            with temp_path.open("wb") as output, tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                desc="Downloading Flavia",
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    progress.update(len(chunk))
            temp_path.replace(destination)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Загрузка официального Flavia Leaf Image Dataset."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--url", default=URL)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Не скачивать, а проверить уже существующий архив.",
    )
    args = parser.parse_args()

    if not args.skip_download:
        print(f"Источник: {args.url}")
        print(f"Файл: {args.output}")
        download(args.url, args.output)

    if not args.output.is_file():
        print(f"Архив не найден: {args.output}", file=sys.stderr)
        return 1

    print("Проверка SHA-256...")
    actual = sha256_file(args.output)
    print(f"SHA-256: {actual}")

    if actual.lower() != EXPECTED_SHA256:
        print(
            "ОШИБКА: контрольная сумма не совпала. "
            "Не распаковывайте файл; проверьте источник/загрузку.",
            file=sys.stderr,
        )
        return 2

    print("Архив успешно загружен и прошёл проверку SHA-256.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
