"""Image processing: simple load + basic metadata."""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .scanner import FileObject


@dataclass
class ImageInfo:
    width: int
    height: int
    mode: str
    path: str


def process_image(file_obj: FileObject) -> ImageInfo:
    with Image.open(file_obj.path) as img:
        return ImageInfo(width=img.width, height=img.height, mode=img.mode, path=str(file_obj.path))
