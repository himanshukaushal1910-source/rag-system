"""
config.py

Pydantic BaseSettings — all configuration from environment variables.
All field names are lowercase (pydantic-settings auto-lowercases).
get_settings() is cached via functools.lru_cache — single instance per process.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All application settings loaded from environment / .env file.

    pydantic-settings lowercases all field names automatically, so
    ``OPENAI_API_KEY`` in the env maps to ``settings.openai_api_key``.
    Use snake_case everywhere when accessing settings attributes.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # silently ignore .env keys not declared as fields
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model — never change, must be 3072-dim.",
    )
    embedding_dim: int = Field(
        default=3072,
        description="Embedding dimensionality — must match embedding_model.",
    )
    llm_model: str = Field(default="gpt-4o", description="OpenAI chat model for generation.")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant REST URL.")
    qdrant_collection_name: str = Field(default="pdf_rag", description="Qdrant collection name.")
    qdrant_api_key: str = Field(default="", description="Qdrant API key (empty = no auth).")
    use_grpc: bool = Field(default=True, description="Use gRPC instead of REST for Qdrant (2-3x faster).")
    qdrant_grpc_port: int = Field(default=6334, description="Qdrant gRPC port.")

    # ── FastAPI ───────────────────────────────────────────────────────────────
    api_key: str = Field(description="X-API-Key header value required on all requests.")
    host: str = Field(default="0.0.0.0", description="FastAPI bind host.")
    port: int = Field(default=8000, description="FastAPI bind port.")
    rate_limit_enabled: bool = Field(default=True, description="Enable per-key rate limiting.")
    rate_limit_requests: int = Field(default=60, description="Max requests per window.")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit sliding window.")
    streaming_enabled: bool = Field(default=True, description="Enable SSE streaming endpoint.")

    # ── Retrieval ────────────────────────────────────────────────────────────
    max_retrieval_chunks: int = Field(
        default=40,
        description="Raw candidates fetched per sub-query before reranking. "
                    "Doubled automatically for complex queries.",
    )
    top_k_final: int = Field(
        default=10,
        description="Chunks passed to generator after reranking. "
                    "Doubled automatically for complex queries.",
    )
    reranker_model: str = Field(
        default="BAAI/bge-reranker-large",
        description="HuggingFace cross-encoder model for reranking.",
    )

    # ── HyDE ────────────────────────────────────────────────────────────────
    hyde_enabled: bool = Field(default=True, description="Enable Hypothetical Document Embeddings.")
    hyde_model: str = Field(default="gpt-4o-mini", description="LLM used for HyDE generation.")

    # ── MMR ──────────────────────────────────────────────────────────────────
    mmr_enabled: bool = Field(default=True, description="Enable Maximal Marginal Relevance diversification.")
    mmr_lambda: float = Field(default=0.5, description="MMR lambda — 0=max diversity, 1=max relevance.")

    # ── NLI faithfulness ────────────────────────────────────────────────────
    nli_model: str = Field(
        default="cross-encoder/nli-deberta-v3-small",
        description="NLI model for faithfulness scoring.",
    )
    use_nli_faithfulness: bool = Field(
        default=True,
        description="Use NLI model instead of LLM judge for faithfulness.",
    )

    # ── Hallucination guard ──────────────────────────────────────────────────
    faithfulness_threshold: float = Field(
        default=0.75,
        description="Minimum faithfulness score before re-retrieval is triggered.",
    )
    consistency_samples: int = Field(
        default=3,
        description="Number of parallel generations for self-consistency check.",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    min_chunk_chars: int = Field(
        default=80,
        description="Minimum character length for prose chunks. "
                    "Tables, images, and closing-section chunks bypass this.",
    )
    breakpoint_threshold_amount: float = Field(
        default=85.0,
        description="Percentile breakpoint for SemanticChunker.",
    )
    chunk_overlap_ratio: float = Field(
        default=0.10,
        description="Fraction of chunk content shared with adjacent chunk (boundary overlap).",
    )
    child_chunk_size: int = Field(
        default=300,
        description="Token size for child chunks in parent-child chunking.",
    )
    parent_chunk_size: int = Field(
        default=1000,
        description="Token size for parent chunks in parent-child chunking.",
    )

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_enabled: bool = Field(
        default=True,
        description="Run pytesseract OCR on pages with fewer than ocr_min_words_threshold words.",
    )
    ocr_min_words_threshold: int = Field(
        default=50,
        description="Pages with fewer extracted words trigger OCR fallback.",
    )

    # ── Embedding / ingestion ────────────────────────────────────────────────
    openai_embed_batch_size: int = Field(
        default=512,
        description="Batch size for OpenAI embedding API calls during ingestion.",
    )
    embedding_cache_enabled: bool = Field(
        default=True,
        description="Cache query embeddings in memory to avoid redundant API calls.",
    )
    embedding_cache_size: int = Field(
        default=1000,
        description="LRU cache capacity for query embeddings.",
    )

    # ── Figure description (Option B) ────────────────────────────────────────
    figure_description_enabled: bool = Field(
        default=True,
        description="Use GPT-4o vision to describe figures at ingestion time.",
    )
    figure_description_model: str = Field(
        default="gpt-4o",
        description="Model used for figure description (gpt-4o recommended).",
    )
    figure_description_max_per_doc: int = Field(
        default=20,
        description="Max figure pages to describe per PDF (cost control).",
    )
    tesseract_cmd: str = Field(
        default="",
        description="Full path to tesseract.exe on Windows. Leave empty on Linux/Mac.",
    )

    # ── Paths ─────────────────────────────────────────────────────────────────
    pdf_dir: str = Field(
        default="data/pdfs/papers",
        description="Directory where source PDFs are stored (relative or absolute).",
    )
    upload_dir: str = Field(
        default="data/pdfs/papers",
        description="Directory where uploaded PDFs are saved.",
    )
    allowed_ingest_roots: list[str] = Field(
        default_factory=list,
        description=(
            "Absolute path prefixes allowed for POST /api/ingest directory field. "
            "Empty list = only upload_dir is allowed."
        ),
    )

    # ── RAG Fusion ────────────────────────────────────────────────────────────
    rag_fusion_enabled: bool = Field(
        default=True,
        description="Generate N paraphrase variants of query → retrieve → client-side RRF merge.",
    )
    rag_fusion_num_queries: int = Field(
        default=3,
        description="Number of paraphrase variants to generate for RAG Fusion.",
    )

    # ── Step-Back Prompting ───────────────────────────────────────────────────
    step_back_enabled: bool = Field(
        default=True,
        description="Generate an abstract step-back query alongside the specific query for broader retrieval.",
    )

    # ── Sentence Window Retrieval ─────────────────────────────────────────────
    sentence_window_enabled: bool = Field(
        default=True,
        description="Expand each retrieved chunk to include neighboring chunks for richer context.",
    )
    sentence_window_size: int = Field(
        default=2,
        description="Number of neighboring chunks to include on each side of a retrieved chunk.",
    )

    # ── Contextual Compression ────────────────────────────────────────────────
    contextual_compression_enabled: bool = Field(
        default=True,
        description="Use gpt-4o-mini to extract only query-relevant sentences from each chunk. "
                    "Adds ~0.3-0.5s per query but reduces context noise.",
    )
    contextual_compression_model: str = Field(
        default="gpt-4o-mini",
        description="Model used for contextual compression.",
    )

    # ── Query Routing ─────────────────────────────────────────────────────────
    query_routing_enabled: bool = Field(
        default=True,
        description="Classify query type (factual/analytical/visual/table/code) and route accordingly.",
    )

    # ── App mode ──────────────────────────────────────────────────────────────
    debug: bool = Field(
        default=False,
        description="Enable debug mode — exposes full exception details in API responses.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", description="structlog log level.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached via lru_cache — Settings() is only constructed once per process.
    All code must import and call get_settings() — never instantiate Settings()
    directly elsewhere.

    Returns:
        Application :class:`Settings` instance.
    """
    return Settings()
