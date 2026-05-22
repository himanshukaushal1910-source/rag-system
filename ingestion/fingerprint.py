"""
ingestion/fingerprint.py

SPEED-1: Page fingerprint from raw bytes (text bytes from fitz).

Previously rendered a 72dpi pixmap (~50ms per page) just to get bytes
for SHA-256. Now pdf_parser.py passes text bytes directly — instant.

The fingerprint function signature is unchanged so ingestor.py works
without modification.
"""

from __future__ import annotations

import hashlib


def compute_page_fingerprint(raw_bytes: bytes) -> str:
    """Compute SHA-256 fingerprint of page bytes for dedup.

    Args:
        raw_bytes: Raw bytes representing the page — now text bytes
                   from fitz.get_text().encode() instead of pixmap bytes.
                   Fingerprint is stable as long as page text doesn't change.

    Returns:
        SHA-256 hex digest string.
    """
    return hashlib.sha256(raw_bytes).hexdigest()
