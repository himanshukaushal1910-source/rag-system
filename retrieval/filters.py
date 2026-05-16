from __future__ import annotations

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)


def build_filter(
    doc_ids: list[str] | None = None,
    content_types: list[str] | None = None,
    page_number_gte: int | None = None,
    page_number_lte: int | None = None,
) -> Filter | None:
    """Build a Qdrant :class:`Filter` from optional metadata constraints.

    All provided conditions are combined with AND (``must``). Returns
    ``None`` if no constraints are specified, which tells Qdrant to search
    the entire collection without filtering.

    Args:
        doc_ids: Restrict results to these document UUIDs. If a single
            doc_id is provided it is matched exactly; multiple use
            ``MatchAny``.
        content_types: Restrict by content modality. Valid values:
            ``"text"``, ``"table"``, ``"image"``, ``"chart"``.
        page_number_gte: Minimum page number (inclusive).
        page_number_lte: Maximum page number (inclusive).

    Returns:
        A :class:`Filter` object or ``None`` if no constraints given.

    Examples::

        # Only text chunks from two specific documents
        f = build_filter(
            doc_ids=["abc", "def"],
            content_types=["text"],
        )

        # Pages 3–10 of any document
        f = build_filter(page_number_gte=3, page_number_lte=10)

        # No filter — search everything
        f = build_filter()  # returns None
    """
    must: list[FieldCondition] = []

    if doc_ids:
        if len(doc_ids) == 1:
            must.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_ids[0]),
                )
            )
        else:
            must.append(
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=doc_ids),
                )
            )

    if content_types:
        if len(content_types) == 1:
            must.append(
                FieldCondition(
                    key="content_type",
                    match=MatchValue(value=content_types[0]),
                )
            )
        else:
            must.append(
                FieldCondition(
                    key="content_type",
                    match=MatchAny(any=content_types),
                )
            )

    if page_number_gte is not None or page_number_lte is not None:
        must.append(
            FieldCondition(
                key="page_number",
                range=Range(
                    gte=page_number_gte,
                    lte=page_number_lte,
                ),
            )
        )

    if not must:
        return None

    return Filter(must=must)
