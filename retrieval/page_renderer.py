"""
retrieval/page_renderer.py

On-demand page rendering for figure/chart retrieval.

When a query asks about a figure, chart, or graph, we:
1. Identify which pages contain relevant figures from retrieved chunks
2. Render those pages as high-res images using fitz (PyMuPDF)
3. Return base64-encoded page images for GPT-4o vision

This works for ALL figure types including vector graphics (matplotlib,
LaTeX, R plots) that are NOT stored as embedded image objects in the PDF
and therefore cannot be extracted by pdfimages or fitz extract_image().

No re-ingestion required — renders from source PDFs at query time.
"""

from __future__ import annotations

import base64
import functools
import re
from pathlib import Path

import structlog

from config import get_settings

logger = structlog.get_logger(__name__)

# Keywords that indicate a query is asking about a visual element
_FIGURE_QUERY_RE = re.compile(
    r"\b(figure|fig|chart|graph|plot|diagram|visualization|heatmap|"
    r"curve|image|illustration|t-sne|tsne|kaplan|scatter|bar chart|"
    r"histogram|attention map)\b",
    re.I,
)

# Caption patterns in chunk text that indicate a figure reference
_CAPTION_RE = re.compile(
    r"\b(figure|fig\.?|chart|graph|plot)\s*(\d+)",
    re.I,
)


def _pdf_dir() -> Path:
    """Return PDF directory from settings (never hardcoded)."""
    return Path(get_settings().pdf_dir)


def _is_figure_query(query: str) -> bool:
    """Return True if query is asking about a visual element."""
    return bool(_FIGURE_QUERY_RE.search(query))


def _find_pdf_path(filename: str) -> Path | None:
    """Find the PDF file path for a given filename.

    Searches the configured papers directory and adjacent subdirectories.
    """
    pdf_dir = _pdf_dir()
    candidates = [
        pdf_dir / filename,
        pdf_dir.parent / "test files" / filename,
        pdf_dir.parent / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


try:
    import fitz as _fitz
    _FITZ_AVAILABLE = True
except ImportError:
    _fitz = None  # type: ignore[assignment]
    _FITZ_AVAILABLE = False


@functools.lru_cache(maxsize=32)
def _open_fitz_doc(pdf_path_str: str):
    """Return a cached fitz.Document for the given path.

    LRU cache holds up to 32 open document handles, evicting the least-recently
    used when full. Avoids reopening and re-parsing the cross-reference table on
    every render call (L-8 fix).
    """
    try:
        return _fitz.open(pdf_path_str)
    except Exception as exc:
        logger.warning("page_renderer.open_failed", path=pdf_path_str, error=str(exc))
        return None


def render_page_as_base64(
    filename: str,
    page_number: int,
    dpi: int = 150,
) -> str | None:
    """Render a single PDF page as a base64-encoded PNG image.

    Uses PyMuPDF (fitz) to render the full page including vector
    graphics, charts, and all visual content. PDF handles are cached
    via LRU so repeated renders of the same document avoid re-parsing.

    Args:
        filename:    PDF filename (basename only).
        page_number: 1-indexed page number.
        dpi:         Render resolution (150 = good quality, ~1MB per page).

    Returns:
        Base64-encoded PNG string, or None if rendering fails.
    """
    if not _FITZ_AVAILABLE:
        logger.warning("page_renderer.fitz_not_available")
        return None

    pdf_path = _find_pdf_path(filename)
    if pdf_path is None:
        logger.warning("page_renderer.pdf_not_found", filename=filename)
        return None

    doc = _open_fitz_doc(str(pdf_path))
    if doc is None:
        return None

    try:
        page_idx = page_number - 1  # convert to 0-indexed

        if page_idx < 0 or page_idx >= len(doc):
            logger.warning(
                "page_renderer.page_out_of_range",
                filename=filename,
                page=page_number,
                total=len(doc),
            )
            return None

        page = doc[page_idx]
        mat = _fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pixmap.tobytes("png")

        b64 = base64.b64encode(img_bytes).decode()
        logger.debug(
            "page_renderer.rendered",
            filename=filename,
            page=page_number,
            size_kb=len(img_bytes) // 1024,
        )
        return b64

    except Exception as exc:
        logger.warning(
            "page_renderer.render_failed",
            filename=filename,
            page=page_number,
            error=str(exc),
        )
        return None


def get_figure_pages(
    query: str,
    chunks: list,
    max_pages: int = 3,
) -> list[dict]:
    """Identify and render pages that likely contain figures relevant to query.

    Strategy:
    1. Check if query mentions figures/charts/graphs
    2. Find chunks whose text mentions figure captions (Figure N, Fig. N)
    3. Also include chunks with content_type image/chart
    4. Render those pages as high-res images
    5. Return up to max_pages rendered page images

    Args:
        query:     User query string.
        chunks:    Reranked RetrievedChunk objects.
        max_pages: Maximum pages to render (cost control).

    Returns:
        List of dicts with filename, page, image_b64, caption.
    """
    if not _is_figure_query(query):
        return []

    # Collect candidate pages — deduplicated by (filename, page)
    candidate_pages: dict[tuple[str, int], str] = {}  # key → caption

    for chunk in chunks:
        # Direct image/chart chunks
        if chunk.content_type in ("image", "chart"):
            key = (chunk.filename, chunk.page_number)
            if key not in candidate_pages:
                candidate_pages[key] = chunk.text or ""
            continue

        # Text chunks that mention a figure caption
        caption_matches = _CAPTION_RE.findall(chunk.text)
        if caption_matches:
            key = (chunk.filename, chunk.page_number)
            if key not in candidate_pages:
                match = _CAPTION_RE.search(chunk.text)
                if match:
                    start = max(0, match.start() - 20)
                    end = min(len(chunk.text), match.end() + 80)
                    candidate_pages[key] = chunk.text[start:end].strip()

    if not candidate_pages:
        return []

    # Render up to max_pages
    results: list[dict] = []
    for (filename, page_number), caption in list(candidate_pages.items())[:max_pages]:
        b64 = render_page_as_base64(filename, page_number, dpi=150)
        if b64:
            results.append({
                "filename": filename,
                "page": page_number,
                "image_b64": b64,
                "caption": caption,
                "is_page_render": True,
            })
            logger.info(
                "page_renderer.page_added",
                filename=filename,
                page=page_number,
            )

    return results
