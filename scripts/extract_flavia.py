from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path, PurePosixPath


DEFAULT_ARCHIVE = Path("data/downloads/Leaves.tar.bz2")
DEFAULT_DESTINATION = Path("data/raw/Leaves")


def safe_extract(archive_path: Path, destination: Path) -> int:
    """Извлекает только обычные файлы/директории, блокируя path traversal и ссылки."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    extracted = 0

    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive:
            # Ссылки, устройства, FIFO и прочие специальные типы не извлекаем.
            if not (member.isfile() or member.isdir()):
                continue

            member_path = PurePosixPath(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"Небезопасный путь в архиве: {member.name}")

            target = (destination_resolved / Path(*member_path.parts)).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Путь выходит за пределы каталога: {member.name}")

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1

    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(description="Безопасная распаковка Flavia.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(
            f"Архив не найден: {args.archive}. "
            "Сначала запустите scripts/download_flavia.py"
        )

    count = safe_extract(args.archive, args.destination)
    print(f"Распаковка завершена. Извлечено файлов: {count}")
    print(f"Директория: {args.destination.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
