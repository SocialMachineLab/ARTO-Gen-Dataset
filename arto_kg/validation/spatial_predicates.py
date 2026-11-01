"""
Geometric predicates for spatial relations.

Each predicate checks whether a declared relation holds between two boxes.
Constants are relative to object size. Relations that need depth, occlusion,
part-whole or orientation reasoning are routed to the VLM (see VLM_RELATIONS).
"""

import math
from typing import Dict, List, Optional, Sequence

Box = Sequence[float]

# Fraction of the two objects' mean extent that centres must differ by before
# a directional relation counts. Below this they are treated as aligned.
DIRECTION_MARGIN = 0.25
# Horizontal overlap required before one object can be said to rest on another.
SUPPORT_OVERLAP = 0.30
# How much of A must lie inside B for containment.
CONTAINMENT = 0.90
# Multiples of the two objects' mean diagonal within which they count as near.
NEAR_FACTOR = 1.5
# Vertical slack around B's top edge: a fraction of A's height in `on`, of B's
# height in `over`.
CONTACT_SLACK = 0.10
# How far below B's top edge A's base may sit and still rest on it, as a
# fraction of B's height. A painted surface recedes, so a resting object's base
# can lie well below the far rim of its support.
SUPPORT_DEPTH = 1.0
# Minimum centre separation for `beside`, as a fraction of the narrower width.
BESIDE_OFFSET = 0.5

# Decided by the VLM, not by geometry: depth, occlusion, part-whole, orientation.
VLM_RELATIONS = {
    "in front of", "behind", "through", "against", "opposite", "part of", "in",
    "around",
}


# --- box helpers --------------------------------------------------------------

def _w(b: Box) -> float:
    return max(0.0, b[2] - b[0])


def _h(b: Box) -> float:
    return max(0.0, b[3] - b[1])


def _cx(b: Box) -> float:
    return (b[0] + b[2]) / 2.0


def _cy(b: Box) -> float:
    return (b[1] + b[3]) / 2.0


def _area(b: Box) -> float:
    return _w(b) * _h(b)


def _diag(b: Box) -> float:
    return math.hypot(_w(b), _h(b))


def _intersection(a: Box, b: Box) -> float:
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def x_overlap(a: Box, b: Box) -> float:
    """Horizontal overlap as a fraction of the narrower box."""
    span = min(a[2], b[2]) - max(a[0], b[0])
    denom = min(_w(a), _w(b))
    return max(0.0, span) / denom if denom > 0 else 0.0


def y_overlap(a: Box, b: Box) -> float:
    span = min(a[3], b[3]) - max(a[1], b[1])
    denom = min(_h(a), _h(b))
    return max(0.0, span) / denom if denom > 0 else 0.0


def inside_ratio(a: Box, b: Box) -> float:
    """How much of A lies within B."""
    area = _area(a)
    return _intersection(a, b) / area if area > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return inter / union if union > 0 else 0.0


def centre_distance(a: Box, b: Box) -> float:
    return math.hypot(_cx(a) - _cx(b), _cy(a) - _cy(b))


def _scale(a: Box, b: Box) -> float:
    return (_diag(a) + _diag(b)) / 2.0


# --- predicates ---------------------------------------------------------------
# Each takes (subject_box, object_box) and answers: does "subject REL object" hold?

def left_of(a: Box, b: Box) -> bool:
    return _cx(a) + DIRECTION_MARGIN * (_w(a) + _w(b)) / 2.0 < _cx(b)


def right_of(a: Box, b: Box) -> bool:
    return left_of(b, a)


def above(a: Box, b: Box) -> bool:
    return _cy(a) + DIRECTION_MARGIN * (_h(a) + _h(b)) / 2.0 < _cy(b)


def below(a: Box, b: Box) -> bool:
    return above(b, a)


def on(a: Box, b: Box) -> bool:
    """A rests on B: horizontally over it, with A's base within B's box.

    The base may lie from just above B's top edge (CONTACT_SLACK) down to
    SUPPORT_DEPTH of B's height; unlike `above`, a hanging object does not count.
    """
    if x_overlap(a, b) < SUPPORT_OVERLAP:
        return False
    base = a[3]
    top = b[1]
    return (top - CONTACT_SLACK * _h(a)) <= base <= (top + SUPPORT_DEPTH * _h(b))


def under(a: Box, b: Box) -> bool:
    """A is beneath B, whether or not they touch.

    The inverse of `on` or `over`, covering both resting and hanging cases.
    """
    return on(b, a) or over(b, a)


def over(a: Box, b: Box) -> bool:
    """Above B and clear of it - the hanging lamp, not the resting cup."""
    if x_overlap(a, b) < SUPPORT_OVERLAP:
        return False
    return a[3] < b[1] - CONTACT_SLACK * _h(b)


def inside(a: Box, b: Box) -> bool:
    return inside_ratio(a, b) >= CONTAINMENT


def contains(a: Box, b: Box) -> bool:
    return inside(b, a)


def outside(a: Box, b: Box) -> bool:
    return iou(a, b) <= 0.01 and inside_ratio(a, b) < 0.1 and inside_ratio(b, a) < 0.1


def surrounding(a: Box, b: Box) -> bool:
    if not contains(a, b):
        return False
    if _area(b) <= 0 or _area(a) < 3 * _area(b):
        return False
    # B sits near A's centre rather than hugging one edge.
    return (abs(_cx(a) - _cx(b)) < 0.25 * _w(a)
            and abs(_cy(a) - _cy(b)) < 0.25 * _h(a))


def near(a: Box, b: Box) -> bool:
    return centre_distance(a, b) < NEAR_FACTOR * _scale(a, b)


def beside(a: Box, b: Box) -> bool:
    """Side by side: close, vertically aligned, and offset horizontally.

    Uses centre separation rather than box disjointness, so the overlap that
    perspective produces is allowed.
    """
    if not near(a, b) or y_overlap(a, b) < 0.3:
        return False
    # Scaled by the narrower object so a wide neighbour does not raise the bar.
    centre_gap = abs(_cx(a) - _cx(b))
    narrower = min(_w(a), _w(b))
    return narrower > 0 and centre_gap >= BESIDE_OFFSET * narrower


def _reject_self(predicate):
    """Wrap a predicate so it is False when both boxes are the same.

    Containment and support predicates are otherwise trivially true of a box
    against itself.
    """
    def guarded(a, b):
        if a is b or list(a) == list(b):
            return False
        return predicate(a, b)

    guarded.__name__ = predicate.__name__
    guarded.__doc__ = predicate.__doc__
    return guarded


GEOMETRIC_PREDICATES = {
    name: _reject_self(function) for name, function in {
        "left of": left_of,
        "right of": right_of,
        "above": above,
        "below": below,
        "on": on,
        "under": under,
        "over": over,
        "inside": inside,
        "contains": contains,
        "outside": outside,
        "surrounding": surrounding,
        "near": near,
        "beside": beside,
    }.items()
}


def normalise(relation: str) -> str:
    return str(relation).strip().lower().replace("_", " ")


def channel_for(relation: str) -> str:
    """Which validation channel a relation belongs to: geometric, vlm, unsupported."""
    key = normalise(relation)
    if key in GEOMETRIC_PREDICATES:
        return "geometric"
    if key in VLM_RELATIONS:
        return "vlm"
    return "unsupported"


def evaluate(relation: str, subject_box: Optional[Box], object_box: Optional[Box]) -> Dict[str, object]:
    """Check one asserted relation against two boxes.

    Returns a record so that "could not check" stays distinct from "checked and
    false".
    """
    key = normalise(relation)
    channel = channel_for(key)
    result: Dict[str, object] = {"relation": key, "channel": channel}

    if channel == "unsupported":
        result["status"] = "unsupported"
        result["reason"] = f"'{key}' is not in the relation vocabulary"
        return result

    if channel == "vlm":
        result["status"] = "deferred"
        result["reason"] = "needs depth or semantic reasoning; VLM channel"
        return result

    if subject_box is None or object_box is None:
        result["status"] = "not_checked"
        result["reason"] = "missing bounding box for one or both objects"
        return result

    result["status"] = "checked"
    result["verified"] = bool(GEOMETRIC_PREDICATES[key](subject_box, object_box))
    return result


def merge_vlm_results(summary: Dict[str, object],
                      vlm_results: List[Dict[str, object]]) -> Dict[str, object]:
    """Fold VLM answers back into the summary and rescore over both channels.

    Records are matched on (subject, relation, object). An answered deferred
    relation becomes `checked`; an unanswered one stays deferred.
    """
    answered = {}
    for record in vlm_results:
        if record.get("status") != "checked":
            continue
        key = (record.get("subject"), normalise(record.get("relation", "")),
               record.get("object"))
        answered[key] = record

    merged: List[Dict[str, object]] = []
    for record in summary.get("details", []):
        if record.get("status") != "deferred":
            merged.append(record)
            continue
        key = (record.get("subject"), normalise(record.get("relation", "")),
               record.get("object"))
        answer = answered.get(key)
        if answer is None:
            merged.append(record)
            continue
        updated = dict(record)
        updated.update({
            "status": "checked",
            "verified": bool(answer.get("holds")),
            "confidence": answer.get("confidence"),
            "observed": answer.get("observed"),
            "reason": None,
        })
        merged.append(updated)

    combined = summarise(merged)
    # Per-channel figures (geometric vs VLM).
    checked = [r for r in merged if r.get("status") == "checked"]
    for channel in ("geometric", "vlm"):
        rows = [r for r in checked if r.get("channel") == channel]
        verified = [r for r in rows if r.get("verified")]
        combined[f"{channel}_checked"] = len(rows)
        combined[f"{channel}_verified"] = len(verified)
        combined[f"{channel}_score"] = (
            100.0 * len(verified) / len(rows) if rows else None)
    return combined


def summarise(results: List[Dict[str, object]]) -> Dict[str, object]:
    """Score = verified / checked. Deferred and unchecked relations are reported
    separately and never counted as failures."""
    checked = [r for r in results if r.get("status") == "checked"]
    verified = [r for r in checked if r.get("verified")]

    return {
        "total": len(results),
        "checked": len(checked),
        "verified": len(verified),
        "deferred_to_vlm": len([r for r in results if r.get("status") == "deferred"]),
        "not_checked": len([r for r in results if r.get("status") == "not_checked"]),
        "unsupported": len([r for r in results if r.get("status") == "unsupported"]),
        "score": (100.0 * len(verified) / len(checked)) if checked else None,
        "details": results,
    }
