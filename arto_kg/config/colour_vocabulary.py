"""
Named-colour vocabulary shared by generation and validation.

Generation selects colour names from color_vocabulary.csv and validation looks
the same names up, so every reference colour comes from the CSV. Selection
is two-step: a colour family (colour_group) first, then names within it.
"""

import csv
import os
import threading
from typing import Dict, List, Optional, Tuple

VOCABULARY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "color_vocabulary.csv")

# Warm hue arcs, consistent with the CSV's Warm/Cool labels; everything else
# is cool. See validation/colour_temperature.py.
_WARM_ARCS = ((0.0, 120.0), (300.0, 360.0))

_lock = threading.Lock()
_cache: Optional[Dict[str, object]] = None


def _row_value(row: Dict[str, str], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value:
            return value.strip()
    return ""


def _load() -> Dict[str, object]:
    """Read the CSV once. Returns {by_name, by_group, groups}."""
    by_name: Dict[str, Dict[str, object]] = {}
    by_group: Dict[str, List[str]] = {}

    with open(VOCABULARY_PATH, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = _row_value(row, "color_name", "colour_name")
            hex_string = _row_value(row, "hex_string", "hex")
            if not name or not hex_string:
                continue

            key = name.strip().lower()
            if key in by_name:          # the file repeats some names per source
                continue

            group = _row_value(row, "colour_group", "color_group") or "Other"
            try:
                hue = float(_row_value(row, "hue") or 0.0)
            except ValueError:
                hue = 0.0

            by_name[key] = {
                "name": name.strip(),
                "hex": hex_string.lstrip("#").lower(),
                "group": group,
                "hue": hue,
                "temperature": _row_value(row, "temperature"),
            }
            by_group.setdefault(group, []).append(name.strip())

    return {
        "by_name": by_name,
        "by_group": {g: sorted(names) for g, names in by_group.items()},
        "groups": sorted(by_group),
    }


def _vocabulary() -> Dict[str, object]:
    global _cache
    if _cache is None:
        with _lock:
            if _cache is None:
                _cache = _load()
    return _cache


def groups() -> List[str]:
    """The colour families - the first step of selection."""
    return list(_vocabulary()["groups"])


def names_in(group: str) -> List[str]:
    """Colour names within one family - the second step."""
    return list(_vocabulary()["by_group"].get(group, []))


def names_in_groups(wanted: List[str]) -> List[str]:
    names: List[str] = []
    for group in wanted:
        names.extend(names_in(group))
    return names


def is_known(name: str) -> bool:
    return str(name or "").strip().lower() in _vocabulary()["by_name"]


def rgb_of(name: str) -> Optional[Tuple[int, int, int]]:
    """The CSV RGB for a vocabulary name, or None."""
    entry = _vocabulary()["by_name"].get(str(name or "").strip().lower())
    if entry is None:
        return None
    digits = entry["hex"]
    if len(digits) != 6:
        return None
    try:
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    except ValueError:
        return None


def canonical(name: str) -> Optional[str]:
    """The vocabulary's own spelling of a name, or None if unknown."""
    entry = _vocabulary()["by_name"].get(str(name or "").strip().lower())
    return entry["name"] if entry else None


def group_of(name: str) -> Optional[str]:
    entry = _vocabulary()["by_name"].get(str(name or "").strip().lower())
    return entry["group"] if entry else None


def warm_groups() -> List[str]:
    """Families whose colours sit mostly in the warm hue arcs.

    Derived from the CSV hue values, so it follows changes to the vocabulary.
    """
    return _groups_by_temperature("warm")


def cool_groups() -> List[str]:
    return _groups_by_temperature("cool")


def _groups_by_temperature(wanted: str) -> List[str]:
    tally: Dict[str, List[int]] = {}
    for entry in _vocabulary()["by_name"].values():
        hue = entry["hue"] % 360.0
        warm = any(low <= hue < high for low, high in _WARM_ARCS)
        counts = tally.setdefault(entry["group"], [0, 0])
        counts[0 if warm else 1] += 1

    chosen = []
    for group, (warm_count, cool_count) in tally.items():
        majority = "warm" if warm_count >= cool_count else "cool"
        if majority == wanted:
            chosen.append(group)
    return sorted(chosen)
