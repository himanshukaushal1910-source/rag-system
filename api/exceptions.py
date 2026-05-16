from __future__ import annotations

from http import HTTPStatus


class RagException(Exception):
    """Base exception for the entire RAG system.

    All domain exceptions inherit from this class so callers can catch at
    any granularity they need::

        except RagException:          # catch everything
        except RetrievalError:        # catch only retrieval failures
        except RetrievalError as e:
            if e.status_code == 503:  # inspect HTTP semantics
                ...
    """

    #: Default HTTP status code — subclasses override as needed.
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR.value
    #: Machine-readable error code surfaced in API responses.
    error_code: str = "rag_error"

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or message

    def to_dict(self) -> dict[str, object]:
        """Serialise to a shape suitable for a JSON error response body."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# Ingestion errors
# --------------------------------------------------------------------------- #


class IngestionError(RagException):
    """Raised when the ingestion pipeline fails."""

    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "ingestion_error"


class PDFParseError(IngestionError):
    """Raised when pdfplumber / fitz cannot parse a PDF page."""

    error_code = "pdf_parse_error"


class EmbeddingError(IngestionError):
    """Raised when the OpenAI embeddings API call fails."""

    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "embedding_error"


class ChunkingError(IngestionError):
    """Raised when SemanticChunker produces invalid output."""

    error_code = "chunking_error"


# --------------------------------------------------------------------------- #
# Retrieval errors
# --------------------------------------------------------------------------- #


class RetrievalError(RagException):
    """Raised when Qdrant search or re-ranking fails."""

    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "retrieval_error"


class QdrantConnectionError(RetrievalError):
    """Raised when the async Qdrant client cannot connect."""

    status_code = HTTPStatus.SERVICE_UNAVAILABLE.value
    error_code = "qdrant_connection_error"


class RerankerError(RetrievalError):
    """Raised when the cross-encoder inference step fails."""

    error_code = "reranker_error"


# --------------------------------------------------------------------------- #
# Generation errors
# --------------------------------------------------------------------------- #


class GenerationError(RagException):
    """Raised when GPT-4o generation fails."""

    status_code = HTTPStatus.BAD_GATEWAY.value
    error_code = "generation_error"


class PromptBuildError(GenerationError):
    """Raised when a ChatPromptTemplate cannot be rendered."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "prompt_build_error"


# --------------------------------------------------------------------------- #
# Verification errors
# --------------------------------------------------------------------------- #


class VerificationError(RagException):
    """Raised when the hallucination verifier node fails to execute."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR.value
    error_code = "verification_error"


class FaithfulnessError(VerificationError):
    """Raised when faithfulness scoring itself errors (not: score too low)."""

    error_code = "faithfulness_error"


class CitationVerificationError(VerificationError):
    """Raised when citation extraction / cross-check logic fails."""

    error_code = "citation_verification_error"


# --------------------------------------------------------------------------- #
# API / auth errors
# --------------------------------------------------------------------------- #


class AuthenticationError(RagException):
    """Raised when the X-API-Key header is missing or invalid."""

    status_code = HTTPStatus.UNAUTHORIZED.value
    error_code = "authentication_error"


class ValidationError(RagException):
    """Raised for request payload validation failures (beyond Pydantic)."""

    status_code = HTTPStatus.UNPROCESSABLE_ENTITY.value
    error_code = "validation_error"


class NotFoundError(RagException):
    """Raised when a requested resource (doc_id, job_id) does not exist."""

    status_code = HTTPStatus.NOT_FOUND.value
    error_code = "not_found"
