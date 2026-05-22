"""
ingestion/figure_describer.py

Option B: GPT-4o vision figure description at ingestion time.

For every page that contains figures/charts/plots:
1. Render the full page at 150 DPI
2. Send to GPT-4o vision with a focused description prompt
3. Return rich text description stored as a searchable chunk

This runs ONCE at ingestion time and stores descriptions in Qdrant.
No page rendering needed at query time — descriptions are pre-computed
and fully searchable via dense + sparse retrieval.

Cost: ~$0.01-0.02 per figure page (GPT-4o vision input tokens)
For 200 PDFs × ~30% figure pages × 15 pages = ~900 pages ≈ $18 total

Controlled via config:
  figure_description_enabled: bool = True
  figure_description_model: str = "gpt-4o"
  figure_description_max_per_doc: int = 20  (safety cap)
"""

from __future__ import annotations

import asyncio
import base64
import re
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

# Patterns that indicate a page likely contains a figure
_FIGURE_CAPTION_RE = re.compile(
    r"\b(figure|fig\.?|chart|graph|plot|diagram|heatmap|"
    r"visualization|t-sne|tsne|kaplan|scatter|curve)\s*\d*",
    re.I,
)

# System prompt for figure description
_FIGURE_SYSTEM = """You are a scientific figure analyst. You will be shown a page from a research paper.

Your task:
1. Identify ALL figures, charts, graphs, plots, or diagrams on this page
2. For each figure, provide a detailed description that includes:
   - What TYPE of visualization it is (bar chart, line plot, scatter plot, heatmap, t-SNE, Kaplan-Meier curve, etc.)
   - What the AXES represent (x-axis label, y-axis label, units)
   - What DATA is shown (specific values, ranges, trends)
   - What COLORS or markers represent different groups/conditions
   - The KEY FINDINGS visible in the figure (which group is highest/lowest, what trend exists, statistical significance if shown)
   - The FIGURE NUMBER and CAPTION if visible

Be specific and include all numbers, percentages, and values visible in the figure.
If no figures are present, respond with: NO_FIGURES

Format: For each figure write "Figure [N]: [description]" """

_FIGURE_HUMAN = "Describe all figures on this page of the research paper."


def _page_likely_has_figure(
    page_text: str,
    has_image_blocks: bool,
) -> bool:
    """Check if a page likely contains a figure worth describing.

    Args:
        page_text:        Text extracted from the page.
        has_image_blocks: Whether fitz found embedded image blocks.

    Returns:
        True if page likely has a describable figure.
    """
    # Has embedded image blocks
    if has_image_blocks:
        return True
    # Text mentions a figure caption
    if _FIGURE_CAPTION_RE.search(page_text):
        return True
    return False


async def describe_figure_page(
    fitz_page: object,
    page_number: int,
    filename: str,
    openai_client: object,
    model: str = "gpt-4o",
    dpi: int = 150,
) -> str | None:
    """Render a page and get GPT-4o vision description of its figures.

    Args:
        fitz_page:     PyMuPDF page object.
        page_number:   1-indexed page number (for logging).
        filename:      PDF filename (for logging).
        openai_client: AsyncOpenAI client instance.
        model:         GPT-4o model name.
        dpi:           Render resolution (150 = good quality).

    Returns:
        Text description of figures on the page, or None if no figures
        found or description fails.
    """
    import fitz as _fitz

    try:
        # Render full page as PNG
        mat = _fitz.Matrix(dpi / 72, dpi / 72)
        pixmap = fitz_page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pixmap.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()

        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _FIGURE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": _FIGURE_HUMAN},
                    ],
                },
            ],
            max_tokens=800,
            temperature=0,
        )

        description = (response.choices[0].message.content or "").strip()

        # Skip pages with no figures
        if description.upper().startswith("NO_FIGURES") or not description:
            return None

        logger.info(
            "figure_describer.described",
            filename=filename,
            page=page_number,
            length=len(description),
        )
        return description

    except Exception as exc:
        logger.warning(
            "figure_describer.failed",
            filename=filename,
            page=page_number,
            error=str(exc),
        )
        return None


async def describe_figure_pages_batch(
    fitz_doc: object,
    page_texts: dict[int, str],       # page_idx → text
    page_has_images: dict[int, bool],  # page_idx → bool
    filename: str,
    openai_client: object,
    semaphore: asyncio.Semaphore,
    model: str = "gpt-4o",
    max_per_doc: int = 20,
) -> dict[int, str]:
    """Describe all figure pages in a document concurrently.

    Runs GPT-4o vision calls concurrently but rate-limited via semaphore.

    Args:
        fitz_doc:        Open PyMuPDF document.
        page_texts:      Dict of page_idx → extracted text.
        page_has_images: Dict of page_idx → whether fitz found image blocks.
        filename:        PDF filename for logging.
        openai_client:   AsyncOpenAI client.
        semaphore:       Shared semaphore for rate limiting.
        model:           GPT-4o model name.
        max_per_doc:     Max pages to describe per document (cost control).

    Returns:
        Dict of page_idx → description string (only for pages with figures).
    """
    # Find candidate pages
    candidates: list[int] = []
    for page_idx in range(len(fitz_doc)):
        text = page_texts.get(page_idx, "")
        has_imgs = page_has_images.get(page_idx, False)
        if _page_likely_has_figure(text, has_imgs):
            candidates.append(page_idx)

    # Cap at max_per_doc
    candidates = candidates[:max_per_doc]

    if not candidates:
        return {}

    logger.info(
        "figure_describer.batch_start",
        filename=filename,
        candidate_pages=len(candidates),
    )

    async def _describe_one(page_idx: int) -> tuple[int, str | None]:
        async with semaphore:
            desc = await describe_figure_page(
                fitz_page=fitz_doc[page_idx],
                page_number=page_idx + 1,
                filename=filename,
                openai_client=openai_client,
                model=model,
            )
        return page_idx, desc

    results = await asyncio.gather(
        *[_describe_one(idx) for idx in candidates],
        return_exceptions=True,
    )

    descriptions: dict[int, str] = {}
    for result in results:
        if isinstance(result, Exception):
            logger.warning("figure_describer.gather_error", error=str(result))
            continue
        page_idx, desc = result
        if desc:
            descriptions[page_idx] = desc

    logger.info(
        "figure_describer.batch_done",
        filename=filename,
        described=len(descriptions),
        of=len(candidates),
    )
    return descriptions
