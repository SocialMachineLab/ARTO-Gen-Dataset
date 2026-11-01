"""
Size validation by rank agreement.

Size labels are ordinal (Dominant > Large > Medium > Small > Marginal). For each
pair of objects with different declared sizes, check that the observed areas
are ordered the same way.
"""

from typing import Any, Dict, List, Optional, Sequence

# Size vocabulary, largest first.
SIZE_ORDER = ["Dominant", "Large", "Medium", "Small", "Marginal"]
_RANK = {name.lower(): index for index, name in enumerate(SIZE_ORDER)}


def rank_of(size_label: Optional[str]) -> Optional[int]:
    """Position in the ordering; lower means larger. None if unrecognised."""
    if not isinstance(size_label, str):
        return None
    return _RANK.get(size_label.strip().lower())


def _area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def evaluate(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check declared size ordering against observed areas.

    objects: [{"instance_id", "name", "size", "bbox"}, ...]

    Score is the share of concordant pairs. Pairs sharing a declared size are
    skipped: the label asserts nothing about their relative area.
    """
    usable = []
    skipped = []

    for obj in objects:
        rank = rank_of(obj.get("size"))
        box = obj.get("bbox")
        if rank is None:
            skipped.append({"instance_id": obj.get("instance_id"),
                            "reason": f"unrecognised size label {obj.get('size')!r}"})
            continue
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            skipped.append({"instance_id": obj.get("instance_id"),
                            "reason": "no bounding box"})
            continue
        usable.append({
            "instance_id": obj.get("instance_id"),
            "name": obj.get("name"),
            "size": obj.get("size"),
            "rank": rank,
            "area": _area(box),
        })

    pairs = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            a, b = usable[i], usable[j]
            if a["rank"] == b["rank"]:
                continue  # same declared size: no ordering asserted
            larger, smaller = (a, b) if a["rank"] < b["rank"] else (b, a)
            concordant = larger["area"] > smaller["area"]
            pairs.append({
                "larger_declared": larger["instance_id"],
                "smaller_declared": smaller["instance_id"],
                "concordant": concordant,
            })

    concordant = [p for p in pairs if p["concordant"]]

    return {
        "metric": "rank_agreement",
        "objects_used": len(usable),
        "pairs": len(pairs),
        "concordant": len(concordant),
        "score": (100.0 * len(concordant) / len(pairs)) if pairs else None,
        "skipped": skipped,
        "details": pairs,
    }
