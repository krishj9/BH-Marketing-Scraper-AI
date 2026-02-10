"""Text chunking for embeddings and LLM inputs.

Splits text into chunks at sentence boundaries with configurable
overlap for context continuity.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Sentence-ending patterns
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')
_CLAUSE_BREAK = re.compile(r'(?<=[,;:])\s+')


def chunk_text(
    text: str,
    max_chars: int = 1500,
    overlap: int = 100,
) -> list[str]:
    """Split text into chunks with overlap at sentence boundaries.

    Strategy:
    1. Split on sentence boundaries.
    2. Accumulate sentences up to max_chars.
    3. If a single sentence exceeds max_chars, split on clause boundaries.
    4. Apply overlap between consecutive chunks.

    Args:
        text: Input text to chunk.
        max_chars: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text or not text.strip():
        return []

    text = text.strip()

    # Short text: return as single chunk
    if len(text) <= max_chars:
        return [text]

    # Split into sentences
    sentences = _SENTENCE_END.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk_parts: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If single sentence is too long, split on clause boundaries
        if sentence_len > max_chars:
            # Flush current chunk first
            if current_chunk_parts:
                chunks.append(" ".join(current_chunk_parts))
                current_chunk_parts = []
                current_len = 0

            sub_chunks = _split_long_sentence(sentence, max_chars)
            chunks.extend(sub_chunks)
            continue

        # Check if adding this sentence exceeds limit
        new_len = current_len + sentence_len + (1 if current_chunk_parts else 0)
        if new_len > max_chars and current_chunk_parts:
            # Flush current chunk
            chunks.append(" ".join(current_chunk_parts))

            # Start new chunk with overlap from previous
            overlap_text = _get_overlap_text(current_chunk_parts, overlap)
            current_chunk_parts = [overlap_text] if overlap_text else []
            current_len = len(overlap_text) if overlap_text else 0

        current_chunk_parts.append(sentence)
        current_len += sentence_len + (1 if len(current_chunk_parts) > 1 else 0)

    # Flush remaining
    if current_chunk_parts:
        chunk = " ".join(current_chunk_parts)
        if chunk.strip():
            chunks.append(chunk.strip())

    logger.debug("Split text (%d chars) into %d chunks", len(text), len(chunks))
    return chunks


def _split_long_sentence(sentence: str, max_chars: int) -> list[str]:
    """Split a long sentence on clause boundaries or hard breaks."""
    clauses = _CLAUSE_BREAK.split(sentence)
    if len(clauses) <= 1:
        # No clause boundaries; hard-split
        return _hard_split(sentence, max_chars)

    chunks = []
    current_parts: list[str] = []
    current_len = 0

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        clause_len = len(clause)

        if clause_len > max_chars:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_len = 0
            chunks.extend(_hard_split(clause, max_chars))
            continue

        new_len = current_len + clause_len + (1 if current_parts else 0)
        if new_len > max_chars and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(clause)
        current_len += clause_len + (1 if len(current_parts) > 1 else 0)

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Hard-split text into chunks of max_chars, breaking at word boundaries."""
    words = text.split()
    chunks = []
    current_parts: list[str] = []
    current_len = 0

    for word in words:
        word_len = len(word)
        new_len = current_len + word_len + (1 if current_parts else 0)

        if new_len > max_chars and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_len = 0

        current_parts.append(word)
        current_len += word_len + (1 if len(current_parts) > 1 else 0)

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _get_overlap_text(parts: list[str], overlap_chars: int) -> str:
    """Get the last `overlap_chars` of text from accumulated parts."""
    if not parts or overlap_chars <= 0:
        return ""
    full_text = " ".join(parts)
    if len(full_text) <= overlap_chars:
        return full_text
    # Find a word boundary near the overlap point
    overlap_start = len(full_text) - overlap_chars
    space_pos = full_text.find(" ", overlap_start)
    if space_pos != -1 and space_pos < len(full_text):
        return full_text[space_pos + 1:]
    return full_text[overlap_start:]
