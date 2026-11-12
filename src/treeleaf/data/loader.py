"""Загрузчик и проверка исходных изображений Flavia."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

from .classes import FlaviaClass, get_class_by_image_id


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_ID_PATTERN = re.compile(r"^(\d{4})")


@dataclass(frozen=True, slots=True)
class FlaviaImage:
    path: Path
    relative_path: str
    image_id: int
    class_info: FlaviaClass
    width: int
    height: int
    image_format: str
    size_bytes: int

    @property
    def class_index(self) -> int:
        return self.class_info.class_index

    @property
    def class_slug(self) -> str:
        return self.class_info.slug


@dataclass(frozen=True, slots=True)
class InvalidImage:
    path: Path
    reason: str


class FlaviaLoader:
    """Сканирует Flavia и отдаёт однородные объекты FlaviaImage."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        if not self.root.exists():
            raise FileNotFoundError(
                f"Директория датасета не найдена: {self.root}. "
                "Сначала скачайте и распакуйте Flavia."
            )
        if not self.root.is_dir():
            raise NotADirectoryError(f"Ожидалась директория: {self.root}")

    @staticmethod
    def _parse_image_id(path: Path) -> int | None:
        match = IMAGE_ID_PATTERN.match(path.stem)
        return int(match.group(1)) if match else None

    def iter_images(
        self,
        *,
        verify: bool = True,
    ) -> Iterator[FlaviaImage]:
        """Возвращает только файлы с известным ID, классом и читаемым изображением."""
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            image_id = self._parse_image_id(path)
            if image_id is None:
                continue

            class_info = get_class_by_image_id(image_id)
            if class_info is None:
                continue

            try:
                with Image.open(path) as image:
                    width, height = image.size
                    image_format = image.format or path.suffix.lstrip(".").upper()
                    if width <= 0 or height <= 0:
                        continue
                    if verify:
                        image.verify()

                yield FlaviaImage(
                    path=path,
                    relative_path=path.relative_to(self.root).as_posix(),
                    image_id=image_id,
                    class_info=class_info,
                    width=width,
                    height=height,
                    image_format=image_format,
                    size_bytes=path.stat().st_size,
                )
            except (UnidentifiedImageError, OSError, ValueError):
                continue

    def iter_invalid_images(self) -> Iterator[InvalidImage]:
        """Отдельный проход для поиска битых файлов с известным Flavia ID."""
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = self._parse_image_id(path)
            if image_id is None or get_class_by_image_id(image_id) is None:
                continue
            try:
                with Image.open(path) as image:
                    image.verify()
            except (UnidentifiedImageError, OSError, ValueError) as exc:
                yield InvalidImage(path=path, reason=str(exc))

    def load_rgb(self, item: FlaviaImage) -> Image.Image:
        """Открывает изображение и возвращает независимую RGB-копию."""
        with Image.open(item.path) as image:
            return image.convert("RGB").copy()
