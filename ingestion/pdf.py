"""PDF processing: per-page text extraction and page image rendering."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from .config import Config
from .scanner import FileObject


@dataclass
class PdfPage:
    page_number: int
    text: str
    image_path: str


def process_pdf(file_obj: FileObject, cfg: Config, zoom: float = 2.0) -> list[PdfPage]:
    out_dir = cfg.pages_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = file_obj.path.stem

    pages: list[PdfPage] = []
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(file_obj.path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            pix = page.get_pixmap(matrix=mat)
            image_path = out_dir / f"{stem}_p{page_number:04d}.png"
            pix.save(str(image_path))
            pages.append(
                PdfPage(page_number=page_number, text=text, image_path=str(image_path))
            )
    return pages
