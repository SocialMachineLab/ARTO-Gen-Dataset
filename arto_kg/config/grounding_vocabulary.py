"""
Concepts the knowledge graph aligns to Wikidata and Getty AAT.

Concepts come in eight families (object, place, weather, period, time, style,
size, medium), each with a URI and its approved external links, read from
grounding_concepts.json. The URIs are in the project's `resource/` namespace,
except the medium concept, `arto:DigitalMedium`. An artwork names its style by
the style's Getty AAT concept; the style concepts here carry that AAT concept
and the Wikidata item. Size concepts are the five steps of an object's
relative size, ranked by `notation`, with no external equivalent.
"""

import json
import os
import re
import threading
from typing import Any, Dict, List, Optional

VOCABULARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "grounding_concepts.json")

# Places365 names look like "/l/living_room" or "/b/bedroom/indoor": a letter
# bucket (dropped), the category, then an optional qualifier.
_TRAILING_QUALIFIERS = {
    "indoor": "Indoor", "outdoor": "Outdoor", "wild": "Wild",
    "natural": "Natural", "urban": "Urban", "snowy": "Snowy",
    "cultivated": "Cultivated", "broadleaf": "Broadleaf",
    "sand": "Sandy", "vegetation": "Vegetated",
}

_lock = threading.Lock()
_cache: Optional[Dict[str, Any]] = None


def normalize(value: Any) -> str:
    """The registry's own key form: lowercase, underscores and hyphens to spaces."""
    text = str(value or "").strip().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", text).lower().strip()


def _load() -> Dict[str, Any]:
    with open(VOCABULARY_PATH, encoding="utf-8") as handle:
        document = json.load(handle)

    by_family: Dict[str, Dict[str, Dict[str, Any]]] = {}
    by_uri: Dict[str, Dict[str, Any]] = {}
    for concept in document.get("concepts", []):
        family = concept.get("family")
        key = concept.get("normalized") or normalize(concept.get("label"))
        if not family or not key:
            continue
        by_family.setdefault(family, {}).setdefault(key, concept)
        by_uri[concept["arto_uri"]] = concept

    return {"by_family": by_family, "by_uri": by_uri,
            "families": sorted(by_family), "document": document}


def _vocabulary() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load()
    return _cache


def families() -> List[str]:
    return list(_vocabulary()["families"])


def concepts_in(family: str) -> List[Dict[str, Any]]:
    return list(_vocabulary()["by_family"].get(family, {}).values())


def lookup(family: str, label: Any) -> Optional[Dict[str, Any]]:
    """The concept a label names within one family, or None.

    Families are kept apart so a label never matches a concept of another kind.
    """
    return _vocabulary()["by_family"].get(family, {}).get(normalize(label))


def place_label(place365_name: Any) -> str:
    """The label a Places365 category maps to: "/l/living_room" -> "living room".

    A trailing qualifier moves to the front: "/b/bedroom/indoor" -> "indoor bedroom".
    """
    text = str(place365_name or "").strip().strip("/")
    if not text:
        return ""
    parts = [part for part in text.split("/") if part]
    if not parts:
        return ""
    if len(parts) > 1:
        parts = parts[1:]          # drop the letter bucket

    if len(parts) >= 2:
        qualifier = normalize(parts[-1])
        if qualifier in _TRAILING_QUALIFIERS:
            base = " ".join(parts[:-1])
            return normalize(f"{_TRAILING_QUALIFIERS[qualifier]} {base}")
    return normalize(" ".join(parts))


def external_links(concept: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The approved external mappings on a concept, or an empty list."""
    if not isinstance(concept, dict):
        return []
    return list(concept.get("external") or [])
