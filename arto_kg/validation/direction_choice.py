"""Forced choice between the two directions of a relation.

Asked whether "A <relation> B" holds, the VLM tends to answer yes when the
painting shows "B <relation> A": the relation is checked as a pairing, not as
a direction. So a relation is put as a choice between the stated direction,
the reverse, and neither. The two directions are ordered by a hash of the
relation, so a preference for the first option cannot pass for an answer and
the same relation is always asked the same way.

A symmetric relation ("A opposite B" says the same as "B opposite A") has no
direction to choose: it is asked as one claim about the pair, or neither.
"""

import json
import re
import zlib
from typing import Any, Dict, Optional, Tuple

AS_WRITTEN = "as_written"
REVERSED = "reversed"
NEITHER = "neither"

# Symmetric relations and how the pair claim is phrased.
SYMMETRIC_PHRASING = {
    "opposite": "opposite each other",
    "touching": "touching each other",
    "kissing": "kissing each other",
    "connected to": "connected to each other",
    "beside": "beside each other",
    "near": "near each other",
    "next to": "next to each other",
}


def _normalise(relation: Any) -> str:
    return " ".join(str(relation or "").lower().replace("_", " ").split())


def is_symmetric(relation: Any) -> bool:
    """Whether the relation says the same thing in both directions."""
    return _normalise(relation) in SYMMETRIC_PHRASING


def options(subject: str, object_: str, phrasing: str, key: str,
            relation: Optional[str] = None) -> Tuple[str, Dict[str, str]]:
    """Numbered options and {number: meaning}, for "the <subject> is <phrasing> the <object>".

    For a symmetric `relation` the options are the pair claim and neither.
    """
    if relation is not None and is_symmetric(relation):
        pair = SYMMETRIC_PHRASING[_normalise(relation)]
        stated = [(AS_WRITTEN, f"The {subject} and the {object_} are {pair}.")]
    else:
        stated = [(AS_WRITTEN, f"The {subject} is {phrasing} the {object_}."),
                  (REVERSED, f"The {object_} is {phrasing} the {subject}.")]
        if zlib.crc32(key.encode("utf-8")) % 2:
            stated.reverse()
    stated.append((NEITHER, "Neither, or it cannot be told."))
    text = "\n".join(f"{number}) {sentence}" for number, (_, sentence) in enumerate(stated, 1))
    return text, {str(number): meaning for number, (meaning, _) in enumerate(stated, 1)}


def parse_choice(response: str, meaning: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """The chosen direction from a reply, or None if no option can be read.

    `holds` is true only when the stated direction was chosen.
    """
    if not response:
        return None

    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None

    number = re.match(r"\s*(\d+)", str(parsed.get("choice", "")))
    verdict = meaning.get(number.group(1)) if number else None
    if verdict is None:
        return None

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "observed": str(parsed.get("observed", ""))[:300],
        "verdict": verdict,
        "holds": verdict == AS_WRITTEN,
        "confidence": max(0.0, min(1.0, confidence)),
    }
