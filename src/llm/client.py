"""Gemini LLM client for borderline case analysis.

Uses Vertex AI Generative Models (Gemini) for RAG-style analysis
of page content when vector similarity is borderline.
"""

import json
import logging
import time
from typing import Any

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

from src.llm.prompts import SYSTEM_PROMPT, build_analysis_prompt

logger = logging.getLogger(__name__)

# Retry settings
_MAX_RETRIES = 3
_BASE_BACKOFF_SECONDS = 2.0

# Default LLM response when parsing fails
_DEFAULT_RESPONSE = {
    "has_childcare_benefit": False,
    "confidence": 0.0,
    "detected_phrases": [],
    "explanation": "LLM response could not be parsed.",
}


class LLMClient:
    """Client for Vertex AI Gemini generative model."""

    def __init__(
        self,
        project_id: str,
        region: str,
        model_name: str = "gemini-1.5-flash",
    ):
        """Initialize the LLM client.

        Args:
            project_id: GCP project ID.
            region: GCP region for Vertex AI endpoint.
            model_name: Gemini model name.
        """
        vertexai.init(project=project_id, location=region)
        self.model = GenerativeModel(
            model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        self.generation_config = GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=1024,
        )
        logger.info(
            "LLMClient initialized: model=%s, region=%s", model_name, region
        )

    def analyze_content(self, url: str, content_chunk: str) -> dict[str, Any]:
        """Analyze page content for childcare benefits.

        Args:
            url: The page URL being analyzed.
            content_chunk: The text chunk to analyze.

        Returns:
            Dict with keys: has_childcare_benefit (bool), confidence (float),
            detected_phrases (list), explanation (str).
        """
        prompt = build_analysis_prompt(url, content_chunk)

        for attempt in range(_MAX_RETRIES):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config,
                )
                result = self._parse_response(response.text)
                logger.debug(
                    "LLM analysis for %s: benefit=%s, confidence=%.2f",
                    url,
                    result.get("has_childcare_benefit"),
                    result.get("confidence", 0),
                )
                return result

            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "resource exhausted" in error_str:
                    wait_time = _BASE_BACKOFF_SECONDS * (2 ** attempt)
                    logger.warning(
                        "LLM rate limited (attempt %d/%d), retrying in %.1fs",
                        attempt + 1,
                        _MAX_RETRIES,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    logger.error("LLM analysis failed for %s: %s", url, e)
                    return _DEFAULT_RESPONSE.copy()

        logger.error("LLM analysis exhausted retries for %s", url)
        return _DEFAULT_RESPONSE.copy()

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Parse the JSON response from the LLM.

        Args:
            response_text: Raw response text from the model.

        Returns:
            Parsed dict, or default response if parsing fails.
        """
        try:
            # Strip markdown code fences if present
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first and last lines (code fences)
                lines = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                text = "\n".join(lines)

            result = json.loads(text)

            # Validate required fields
            if "has_childcare_benefit" not in result:
                logger.warning("LLM response missing 'has_childcare_benefit'")
                result["has_childcare_benefit"] = False
            if "confidence" not in result:
                result["confidence"] = 0.0

            return result

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse LLM response: %s", e)
            return _DEFAULT_RESPONSE.copy()
