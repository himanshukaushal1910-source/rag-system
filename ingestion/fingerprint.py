from __future__ import annotations

import hashlib


def compute_page_fingerprint(raw_bytes: bytes) -> str:
    """Compute a SHA-256 fingerprint for a PDF page's raw bytes.

    Used to detect duplicate pages on re-ingestion. The fingerprint is stored
    as a Qdrant payload field and checked before every upsert — if a point
    with the same ``page_fingerprint`` already exists, the page is skipped.

    Args:
        raw_bytes: Raw bytes representing the page (e.g. pixmap PNG bytes
            from fitz, or the raw page byte stream). Must be deterministic
            across runs for the same page content.

    Returns:
        Lowercase hex digest string (64 characters).

    Example::

        fp = compute_page_fingerprint(page.get_pixmap(dpi=72).tobytes("png"))
        # "3b4c1a9e..."
    """
    return hashlib.sha256(raw_bytes).hexdigest()


def fingerprints_match(fp_a: str, fp_b: str) -> bool:
    """Compare two fingerprints in constant time to avoid timing attacks.

    Args:
        fp_a: First fingerprint hex string.
        fp_b: Second fingerprint hex string.

    Returns:
        True if both fingerprints are identical.
    """
    return hashlib.compare_digest(fp_a.lower(), fp_b.lower())
