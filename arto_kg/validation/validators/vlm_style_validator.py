"""Zero-shot style classification with the VLM (validation gate 1).

Uses the five labels, descriptions and synonym table from `style_validator`.
A reply that does not resolve to exactly one label is reported as
unparseable; no gate decision is made for it.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from .style_validator import ART_STYLES, STYLE_PROMPTS, STYLE_SYNONYMS

logger = logging.getLogger(__name__)

MAX_TOKENS = 24  # Enough for a single style label.


def _build_prompt() -> str:
    """Build a closed-set prompt listing each style with its description.

    Descriptions help separate adjacent movements such as Impressionism and
    Post-Impressionism.
    """
    options = "\n".join(
        f"- {style}: {STYLE_PROMPTS[style]}" for style in ART_STYLES
    )
    return (
        "Identify the art style of this painting.\n\n"
        f"Choose exactly one of these five styles:\n{options}\n\n"
        "Answer with the style name only, exactly as written above. "
        "Do not explain, and do not name any style outside the list."
    )


def parse_style(response: str) -> Optional[str]:
    """Resolve a reply to one of the five labels, or None.

    Tries exact labels, then synonyms, both on word boundaries so
    "impressionist" does not match inside "post-impressionist". Returns None
    when the reply names no label or more than one.
    """
    if not response:
        return None
    text = response.strip().lower().replace("_", " ")
    # "post impressionism" and "post-impressionism" name the same style; joined
    # with a hyphen, "impressionism" does not match inside it.
    text = re.sub(r"\bpost\s+(?=impression)", "post-", text)

    matched: List[str] = []
    for style in ART_STYLES:
        if re.search(rf"(?<![\w-]){re.escape(style.lower())}(?![\w-])", text):
            matched.append(style)
    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        return None

    # Longest synonyms first, so "post-impressionist" is tested before
    # "impressionist".
    candidates = sorted(
        ((syn, style) for style, syns in STYLE_SYNONYMS.items() for syn in syns),
        key=lambda pair: len(pair[0]), reverse=True,
    )
    for synonym, style in candidates:
        if re.search(rf"(?<![\w-]){re.escape(synonym)}(?![\w-])", text):
            return style
    return None


class VLMStyleValidator:
    """Zero-shot five-way style classification."""

    def __init__(self, vlm_wrapper):
        if vlm_wrapper is None:
            raise ValueError(
                "VLMStyleValidator needs a VLM wrapper: style is classified "
                "by the VLM."
            )
        self.vlm = vlm_wrapper
        self.art_styles = list(ART_STYLES)

    def classify(self, image_path: str) -> Dict[str, Any]:
        """Classify one image into one of the five styles."""
        result = self.vlm.generate_response(_build_prompt(), image_path,
                                            max_tokens=MAX_TOKENS)
        if not result.get("success"):
            return {
                "predicted_style": None,
                "status": "error",
                "raw_response": result.get("response"),
                "error": result.get("error"),
                "method": "vlm_zero_shot",
            }

        raw = result.get("response", "")
        predicted = parse_style(raw)
        return {
            "predicted_style": predicted,
            "status": "checked" if predicted else "unparseable",
            "raw_response": raw.strip()[:200],
            "candidates": self.art_styles,
            "method": "vlm_zero_shot",
            "model_used": result.get("model_used"),
            "substitute_model": result.get("substitute_model", False),
        }

    def main_evaluation(self, json_data: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """Classify the image and report whether it matches the requested style.

        The gate decision is left to the caller.
        """
        expected = self._expected_style(json_data)
        outcome = self.classify(image_path)

        predicted = outcome.get("predicted_style")
        outcome["expected_style"] = expected
        outcome["matched"] = bool(
            predicted and expected
            and predicted.strip().lower() == str(expected).strip().lower()
        )
        # A forced five-way choice needs no cutoff.
        outcome["threshold"] = None
        return outcome

    @staticmethod
    def _expected_style(json_data: Dict[str, Any]) -> Optional[str]:
        """Return the style requested by the specification, if any."""
        style = json_data.get("style")
        if isinstance(style, str) and style.strip():
            return style.strip()
        metadata = json_data.get("metadata") or {}
        if isinstance(metadata, dict):
            style = metadata.get("style")
            if isinstance(style, str) and style.strip():
                return style.strip()
        return None
