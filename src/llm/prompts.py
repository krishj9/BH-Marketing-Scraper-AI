"""Prompt templates for LLM analysis of borderline cases.

Based on the LLM Prompt Template defined in project-prompt.md.
"""

SYSTEM_PROMPT = (
    "You are an expert at identifying employer provided childcare benefits "
    "in web content."
)

USER_PROMPT_TEMPLATE = """Analyze the following page content and determine whether it describes employer provided childcare support including on site childcare, childcare subsidies or stipends, backup childcare, dependent care benefits, partnerships, or other employer childcare programs. Return a JSON object:

{{
  "has_childcare_benefit": bool,
  "confidence": float,
  "detected_phrases": [{{"text": str, "type": str, "confidence": float}}],
  "explanation": str
}}

**INPUT**

- URL: {url}
- CONTENT: {content_chunk}

**GUIDANCE**

- Use the provided keyword concepts to guide detection.
- Ignore job postings for childcare workers unless they describe employee benefits.
- Focus on benefits, perks, compensation sections, press releases, and partnership announcements.
- Classify detected phrases as one of: on_site, subsidy, backup, dependent_care, partnership, other.
- Return ONLY the JSON object, no additional text."""


def build_analysis_prompt(url: str, content_chunk: str) -> str:
    """Build the user prompt for LLM analysis.

    Args:
        url: The page URL being analyzed.
        content_chunk: The text chunk to analyze.

    Returns:
        Formatted prompt string.
    """
    return USER_PROMPT_TEMPLATE.format(url=url, content_chunk=content_chunk)
