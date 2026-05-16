from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration loaded from environment / .env file.

    All secrets and tunables are defined here. No hardcoded values anywhere
    else in the codebase — import ``get_settings()`` instead.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # OpenAI
    # ------------------------------------------------------------------ #
    openai_api_key: str = Field(..., description="OpenAI API key (sk-...)")
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model name",
    )
    embedding_dim: int = Field(
        default=3072,
        description="Embedding dimensionality — must match the model",
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="Chat completion model used for generation & verification",
    )

    # ------------------------------------------------------------------ #
    # Qdrant
    # ------------------------------------------------------------------ #
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Qdrant base URL",
    )
    qdrant_api_key: str = Field(
        default="",
        description="Qdrant API key — empty string for local Docker",
    )
    qdrant_collection_name: str = Field(
        default="pdf_rag",
        description="Name of the Qdrant collection",
    )

    # ------------------------------------------------------------------ #
    # FastAPI / Auth
    # ------------------------------------------------------------------ #
    api_key: str = Field(..., description="Secret for X-API-Key header auth")
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    # ------------------------------------------------------------------ #
    # Retrieval & Re-ranking
    # ------------------------------------------------------------------ #
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Sentence-Transformers cross-encoder model",
    )
    max_retrieval_chunks: int = Field(
        default=20,
        description="Number of candidates fetched from Qdrant before re-ranking",
    )
    top_k_final: int = Field(
        default=5,
        description="Number of chunks kept after re-ranking for generation",
    )

    # ------------------------------------------------------------------ #
    # Hallucination Guard
    # ------------------------------------------------------------------ #
    faithfulness_threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum faithfulness score; below this triggers re-retrieval",
    )
    consistency_samples: int = Field(
        default=3,
        ge=1,
        description="Number of parallel generations for self-consistency check",
    )

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    openai_embed_batch_size: int = Field(
        default=512,
        le=2048,
        description="Max inputs per OpenAI embeddings API call (hard cap: 2048)",
    )
    image_b64_size_threshold_bytes: int = Field(
        default=250_000,
        description="Images larger than this are compressed before Qdrant storage",
    )
    min_chunk_size: int = Field(
        default=50,
        description="Minimum character length for a chunk to be indexed",
    )

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @field_validator("top_k_final")
    @classmethod
    def _top_k_lte_max_chunks(cls, v: int, info: object) -> int:  # noqa: ANN001
        # Can't cross-validate with max_retrieval_chunks easily via field_validator
        # (that requires model_validator); keep a sanity floor here.
        if v < 1:
            raise ValueError("top_k_final must be >= 1")
        return v


# Module-level singleton — import this everywhere.
_settings: Settings | None = None


def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Instantiated lazily on first call so tests can monkey-patch env vars
    before the object is created.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
