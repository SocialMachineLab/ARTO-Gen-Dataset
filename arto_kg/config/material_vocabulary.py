"""
Material vocabulary grounded in the Getty AAT Materials facet.

ARTO types `material` as an AAT concept, so objects select from the concepts in
aat_materials.json and the graph records an AAT id (e.g. `aat:300011914`).
Aliases map common synonyms ("fabric", "oak") to their canonical label.

Body coverings (`fur`, `flesh`, `hide`, `skin`, `feathers`) are deliberately
absent; whether an object is made of a material is decided per object
(`is_made_object`). The `group` field lays out the menu but never narrows it.

Entries marked `lookup_only` (modern materials such as plastic) resolve to
their AAT concept for lookups but are left out of the menu, `labels()` and
`resolve_all()`, so they are neither offered to the model nor accepted from it.
"""

import json
import random
import os
import threading
from typing import Dict, List, Optional

VOCABULARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "aat_materials.json")

# Getty AAT prefix, matching the one the mapper declares for style concepts.
AAT_PREFIX = "aat:"

_lock = threading.Lock()
_cache: Optional[Dict[str, object]] = None


def _load() -> Dict[str, object]:
    """Read the JSON once. Returns {by_label, by_group, groups, order, meta}."""
    with open(VOCABULARY_PATH, encoding="utf-8") as handle:
        document = json.load(handle)

    by_label: Dict[str, Dict[str, object]] = {}
    by_group: Dict[str, List[str]] = {}
    order: List[str] = []

    for record in document.get("materials", []):
        label = str(record.get("label", "")).strip()
        aat_id = str(record.get("aat_id", "")).strip()
        if not label or not aat_id:
            continue

        entry = {
            "label": label,
            "aat_id": aat_id,
            "aat_label": str(record.get("aat_label", "")).strip() or label,
            "group": str(record.get("group", "")).strip() or "other",
            "offered": not record.get("lookup_only", False),
        }

        by_label[label.lower()] = entry
        if entry["offered"]:
            order.append(label)
            by_group.setdefault(entry["group"], []).append(label)

        # Aliases resolve to the same entry but never override a real label.
        for alias in record.get("aliases", []):
            key = str(alias).strip().lower()
            if key and key not in by_label:
                by_label[key] = entry

    groups = [g for g in document.get("groups", []) if g in by_group]
    groups.extend(sorted(g for g in by_group if g not in groups))

    return {
        "by_label": by_label,
        "by_group": by_group,
        "groups": groups,
        "order": order,
        "semantic_rule": document.get("semantic_rule", ""),
        "composite_rule": document.get("composite_rule", ""),
    }


def _vocabulary() -> Dict[str, object]:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load()
    return _cache


def labels() -> List[str]:
    """Every material label offered to the model, in the file's own order."""
    return list(_vocabulary()["order"])


def groups() -> List[str]:
    """The material families, used to lay out the menu."""
    return list(_vocabulary()["groups"])


def labels_in(group: str) -> List[str]:
    return list(_vocabulary()["by_group"].get(group, []))


def is_known(name: str) -> bool:
    """True for a label or any of its aliases, lookup-only ones included."""
    return str(name or "").strip().lower() in _vocabulary()["by_label"]


def is_offered(name: str) -> bool:
    """True for a label or alias the model may choose (not lookup-only)."""
    entry = _vocabulary()["by_label"].get(str(name or "").strip().lower())
    return bool(entry and entry["offered"])


def canonical(name: str) -> Optional[str]:
    """The vocabulary's own label for a name or alias, or None if unknown."""
    entry = _vocabulary()["by_label"].get(str(name or "").strip().lower())
    return entry["label"] if entry else None


def aat_id_of(name: str) -> Optional[str]:
    """The Getty AAT concept id, or None."""
    entry = _vocabulary()["by_label"].get(str(name or "").strip().lower())
    return entry["aat_id"] if entry else None


def aat_uri_of(name: str) -> Optional[str]:
    """The prefixed URI to write into Turtle, e.g. `aat:300011914`."""
    aat_id = aat_id_of(name)
    return f"{AAT_PREFIX}{aat_id}" if aat_id else None


def aat_label_of(name: str) -> Optional[str]:
    """Getty's own disambiguated term, e.g. `tin (metal)`.

    Shown in prompts because some labels (`tin`, `bamboo`, `paper`) are also
    object variety names.
    """
    entry = _vocabulary()["by_label"].get(str(name or "").strip().lower())
    return entry["aat_label"] if entry else None


def group_of(name: str) -> Optional[str]:
    entry = _vocabulary()["by_label"].get(str(name or "").strip().lower())
    return entry["group"] if entry else None


def menu(shuffle: bool = False) -> str:
    """The vocabulary as prompt text, grouped and showing Getty's qualifier.

    `shuffle` randomises the order of groups and of labels within them, to
    remove list-position bias in the model's choice.
    """
    ordered_groups = list(groups())
    if shuffle:
        random.shuffle(ordered_groups)

    lines = []
    for group in ordered_groups:
        lines.append(f"  [{group}]")
        group_labels = labels_in(group)
        if shuffle:
            random.shuffle(group_labels)
        for label in group_labels:
            lines.append(f"    - {label}  ({aat_label_of(label)})")
    return "\n".join(lines)


def resolve_all(names: List[str]) -> Dict[str, List[str]]:
    """Split a model's names into offered canonical labels and the rest.

    Returns {"accepted": [...], "rejected": [...]}. Synonyms collapse to one
    accepted label; lookup-only materials are rejected like unknown ones.
    """
    accepted: List[str] = []
    rejected: List[str] = []
    for name in names or []:
        label = canonical(name) if is_offered(name) else None
        if label is None:
            rejected.append(str(name))
        elif label not in accepted:
            accepted.append(label)
    return {"accepted": accepted, "rejected": rejected}
