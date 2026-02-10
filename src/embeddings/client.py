"""Vertex AI embedding client.

Uses Vertex AI TextEmbeddingModel for computing embeddings
of page chunks and keyword text.
"""

import logging
import time
from typing import Optional

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

logger = logging.getLogger(__name__)

# Retry settings for rate limit (429) errors
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2.0


class EmbeddingClient:
    """Client for Vertex AI text embeddings."""

    def __init__(
        self,
        project_id: str,
        region: str,
        model_name: str = "text-embedding-004",
        batch_size: int = 250,
    ):
        """Initialize the embedding client.

        Args:
            project_id: GCP project ID.
            region: GCP region for Vertex AI endpoint.
            model_name: Embedding model name.
            batch_size: Max texts per API call (Vertex AI limit).
        """
        vertexai.init(project=project_id, location=region)
        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.batch_size = batch_size
        logger.info(
            "EmbeddingClient initialized: model=%s, region=%s", model_name, region
        )

    def get_embeddings(
        self,
        texts: list[str],
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Automatically batches requests to respect API limits.
        Uses task_type to optimize embeddings for retrieval.

        Args:
            texts: List of text strings to embed.
            task_type: Vertex AI task type. Use "RETRIEVAL_DOCUMENT" for
                       page chunks and "RETRIEVAL_QUERY" for keyword queries.

        Returns:
            List of embedding vectors (list of floats).
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch = texts[batch_start:batch_end]

            inputs = [
                TextEmbeddingInput(text=t, task_type=task_type) for t in batch
            ]

            embeddings = self._call_with_retry(inputs)
            all_embeddings.extend([e.values for e in embeddings])

            logger.debug(
                "Embedded batch %d-%d of %d texts",
                batch_start,
                batch_end,
                len(texts),
            )

        return all_embeddings

    def get_query_embedding(self, text: str) -> list[float]:
        """Get embedding for a query/keyword string.

        Uses RETRIEVAL_QUERY task type for optimal retrieval performance.

        Args:
            text: Query text to embed.

        Returns:
            Embedding vector.
        """
        results = self.get_embeddings([text], task_type="RETRIEVAL_QUERY")
        return results[0]

    def _call_with_retry(
        self, inputs: list[TextEmbeddingInput]
    ) -> list:
        """Call the embedding API with retry on rate limit errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                return self.model.get_embeddings(inputs)
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource exhausted" in error_str:
                    wait_time = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "Rate limited (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1,
                        _MAX_RETRIES,
                        wait_time,
                        e,
                    )
                    time.sleep(wait_time)
                else:
                    raise
        # Final attempt without catching
        return self.model.get_embeddings(inputs)
