"""Exact keyword substring matcher.

Simple binary matching: if any keyword appears as a case-insensitive
substring in the page text, it's a match. No scoring or weighting.
"""

import logging
from typing import Optional

from src.models import ExactMatchResult

logger = logging.getLogger(__name__)


class ExactMatcher:
    """Case-insensitive exact substring matcher for keywords."""

    def __init__(self, keywords: list[str]):
        """Initialize with a list of keyword phrases.

        Args:
            keywords: List of keyword phrases to match against.
        """
        # Store lowercased keywords for case-insensitive matching
        self.keywords = [kw.strip().lower() for kw in keywords if kw.strip()]
        logger.info("ExactMatcher initialized with %d keywords", len(self.keywords))

    def match(self, text: str) -> Optional[ExactMatchResult]:
        """Check if any keyword appears in the text.

        Args:
            text: Normalized page text to search.

        Returns:
            ExactMatchResult with matched=True and the keyword if found,
            None if no match.
        """
        if not text:
            return None

        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword in text_lower:
                logger.debug("Exact match found: '%s'", keyword)
                return ExactMatchResult(matched=True, keyword=keyword)

        return None

    def match_all(self, text: str) -> list[ExactMatchResult]:
        """Find all keyword matches in the text.

        Args:
            text: Normalized page text to search.

        Returns:
            List of ExactMatchResult for each matched keyword.
        """
        if not text:
            return []

        text_lower = text.lower()
        results = []
        for keyword in self.keywords:
            if keyword in text_lower:
                results.append(ExactMatchResult(matched=True, keyword=keyword))

        return results
