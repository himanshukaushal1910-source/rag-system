from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import fitz  # pymupdf
import pdfplumber
import structlog
from PIL import Image

from api.exceptions import PDFParseError
from config import get_settings

logger: structlog.BoundLogger = structlog.get_logger(__name__)

ContentType = Literal["text", "table", "image", "chart"]


@dataclass
class ParsedPage:
    """All extracted content for a single PDF page.

    Attributes:
        page_number: 1-indexed page number.
        raw_bytes: Raw page bytes used for SHA-256 fingerprinting.
        text_blocks: Plain text blocks extracted via fitz.
        tables: List of tables, each table is a list of row-dicts.
        images: List of (content_type, base64_str, ocr_text) tuples.
    """

    page_number: int
    raw_bytes: bytes
    text_blocks: list[str] = field(default_factory=list)
    tables: list[list[dict[str, str | None]]] = field(default_factory=list)
    images: list[tuple[ContentType, str, str]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Aggregated parsing result for a full PDF.

    Attributes:
        filename: Original PDF filename (not full path).
        doc_id: UUID string assigned by the caller (ingestion orchestrator).
        pages: Ordered list of ParsedPage objects.
    """

    filename: str
    doc_id: str
    pages: list[ParsedPage] = field(default_factory=list)


def _compress_image(img: Image.Image, max_bytes: int) -> bytes:
    """Compress a PIL image to JPEG until it fits within ``max_bytes``.

    Args:
        img: PIL Image to compress.
        max_bytes: Target maximum byte size.

    Returns:
        JPEG-encoded bytes at an appropriate quality level.
    """
    buf = io.BytesIO()
    quality = 85
    img.save(buf, format="JPEG", quality=quality)
    while buf.tell() > max_bytes and quality > 20:
        quality -= 10
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _image_to_b64(
    image_bytes: bytes,
    max_bytes: int,
) -> str:
    """Convert raw image bytes to a base64 string, compressing if needed.

    Args:
        image_bytes: Raw image bytes as extracted by fitz.
        max_bytes: If the image exceeds this size, compress before encoding.

    Returns:
        Base64-encoded string (no data-URI prefix).
    """
    if len(image_bytes) > max_bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_bytes = _compress_image(img, max_bytes)
    return base64.b64encode(image_bytes).decode("utf-8")


def _classify_image(img: Image.Image) -> ContentType:
    """Heuristically classify an image as 'chart' or 'image'.

    Charts tend to have low unique-colour counts relative to their area.
    This is a fast approximation — replace with a classifier if needed.

    Args:
        img: PIL Image to inspect.

    Returns:
        'chart' if the image looks like a plot/diagram, 'image' otherwise.
    """
    small = img.resize((64, 64)).convert("RGB")
    unique_colors = len(set(small.getdata()))
    # Empirically: charts ≤ ~400 unique colours at 64×64; photos >> 1000
    return "chart" if unique_colors < 500 else "image"


def _extract_tables_pdfplumber(
    pdf_path: Path,
    page_number: int,
) -> list[list[dict[str, str | None]]]:
    """Extract structured tables from a single page using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.
        page_number: 1-indexed page number.

    Returns:
        A list of tables; each table is a list of row-dicts with header keys.
    """
    tables: list[list[dict[str, str | None]]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # pdfplumber pages are 0-indexed
            page = pdf.pages[page_number - 1]
            raw_tables = page.extract_tables()
            for raw_table in raw_tables:
                if not raw_table or len(raw_table) < 2:
                    continue
                headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(raw_table[0])]
                rows = []
                for row in raw_table[1:]:
                    rows.append(
                        {
                            headers[i]: (str(cell).strip() if cell is not None else None)
                            for i, cell in enumerate(row)
                        }
                    )
                if rows:
                    tables.append(rows)
    except Exception as exc:
        logger.warning(
            "pdfplumber table extraction failed",
            page=page_number,
            error=str(exc),
        )
    return tables


def _extract_images_fitz(
    fitz_page: fitz.Page,
    fitz_doc: fitz.Document,
    page_number: int,
    max_bytes: int,
) -> list[tuple[ContentType, str, str]]:
    """Extract and classify all images on a fitz page.

    Args:
        fitz_page: The fitz Page object.
        fitz_doc: The parent fitz Document (needed for xref lookup).
        page_number: 1-indexed page number (for logging only).
        max_bytes: Compression threshold in bytes.

    Returns:
        List of (content_type, base64_str, placeholder_ocr_text) tuples.
        OCR text is currently empty — wire up pytesseract here if needed.
    """
    results: list[tuple[ContentType, str, str]] = []
    image_list = fitz_page.get_images(full=True)

    for img_info in image_list:
        xref = img_info[0]
        try:
            base_image = fitz_doc.extract_image(xref)
            img_bytes = base_image["image"]
            if len(img_bytes) < 1024:
                # Skip tiny images (icons, decorators)
                continue
            pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            content_type: ContentType = _classify_image(pil_img)
            b64 = _image_to_b64(img_bytes, max_bytes)
            results.append((content_type, b64, ""))
        except Exception as exc:
            logger.warning(
                "Failed to extract image",
                xref=xref,
                page=page_number,
                error=str(exc),
            )
    return results


def parse_pdf(pdf_path: Path, doc_id: str) -> ParsedDocument:
    """Parse a PDF file into structured per-page content.

    Combines pdfplumber (table extraction) with fitz (text blocks + images).
    Each page is parsed independently so failures are isolated.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        doc_id: UUID string assigned by the ingestion orchestrator.

    Returns:
        A :class:`ParsedDocument` with all extracted content.

    Raises:
        PDFParseError: If the file cannot be opened or is fundamentally broken.
    """
    settings = get_settings()
    log = logger.bind(filename=pdf_path.name, doc_id=doc_id)

    if not pdf_path.exists():
        raise PDFParseError(
            f"PDF file not found: {pdf_path}",
            detail=str(pdf_path),
        )

    try:
        fitz_doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise PDFParseError(
            f"Cannot open PDF with fitz: {pdf_path.name}",
            detail=str(exc),
        ) from exc

    parsed = ParsedDocument(filename=pdf_path.name, doc_id=doc_id)

    total_pages = fitz_doc.page_count
    log.info("Starting PDF parse", total_pages=total_pages)

    for page_idx in range(total_pages):
        page_number = page_idx + 1
        page_log = log.bind(page=page_number)

        try:
            fitz_page = fitz_doc[page_idx]

            # --- Raw bytes for fingerprinting ---
            raw_bytes: bytes = fitz_page.get_pixmap(dpi=72).tobytes("png")

            # --- Text blocks via fitz ---
            text_blocks: list[str] = []
            blocks = fitz_page.get_text("blocks")  # returns list of (x0,y0,x1,y1,text,...)
            for block in blocks:
                text = block[4].strip()
                if text:
                    text_blocks.append(text)

            # --- Tables via pdfplumber ---
            tables = _extract_tables_pdfplumber(pdf_path, page_number)

            # --- Images via fitz ---
            images = _extract_images_fitz(
                fitz_page,
                fitz_doc,
                page_number,
                settings.image_b64_size_threshold_bytes,
            )

            parsed_page = ParsedPage(
                page_number=page_number,
                raw_bytes=raw_bytes,
                text_blocks=text_blocks,
                tables=tables,
                images=images,
            )
            parsed.pages.append(parsed_page)

            page_log.debug(
                "Page parsed",
                text_blocks=len(text_blocks),
                tables=len(tables),
                images=len(images),
            )

        except PDFParseError:
            raise
        except Exception as exc:
            page_log.error("Page parse failed — skipping", error=str(exc))
            # Soft-fail: skip bad pages rather than aborting the entire document.
            continue

    fitz_doc.close()
    log.info("PDF parse complete", pages_extracted=len(parsed.pages))
    return parsed
