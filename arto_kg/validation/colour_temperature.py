"""
Check the declared `color_harmony.color_temperature` (warm / cool / mixed)
against the image palette.

The warm/cool hue boundary follows the Warm/Cool labels in
`color_vocabulary.csv`. Tinted near-greys keep the temperature of their tint,
as the vocabulary labels them; colours with almost no saturation (grey, black,
white) have no hue to speak of and count as neutral. Warm and cool are weighed
over the coloured part of the palette only, and a palette that is almost all
neutral (an ink painting, say) is not judged.
"""

import colorsys
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Hue arcs (degrees) derived from the color_vocabulary.csv labels.
WARM_ARCS = ((0.0, 120.0), (300.0, 360.0))
COOL_ARC = (120.0, 300.0)

# Fraction of palette mass needed to call it warm or cool (a convention); the
# fraction itself is always reported.
WARM_MAJORITY = 0.60
COOL_MAJORITY = 0.60

# HSV saturation (the vocabulary's own `saturation` column) below which a
# colour is neutral; below it the vocabulary's temperature labels are not used.
NEUTRAL_SATURATION = 0.05

# Share of the palette that must be coloured for warm/cool to be judged.
MIN_COLOURED_SHARE = 0.15

DECLARED_VALUES = ("warm", "cool", "mixed")


def classify_hue(hue_degrees: float) -> str:
    """Classify a hue as warm or cool."""
    hue = float(hue_degrees) % 360.0
    for low, high in WARM_ARCS:
        if low <= hue < high:
            return "warm"
    return "cool"


def classify_rgb(rgb: Sequence[float]) -> str:
    red, green, blue = (max(0, min(255, int(c))) / 255.0 for c in rgb)
    hue, _, _ = colorsys.rgb_to_hls(red, green, blue)
    return classify_hue(hue * 360.0)


def measure(clusters: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Warm/cool balance of a K-means palette.

    `clusters` are K-means palette entries with `rgb` and `proportion`; each
    is weighted by proportion only.
    """
    warm_mass = 0.0
    cool_mass = 0.0
    neutral_mass = 0.0
    detail = []

    for cluster in clusters:
        rgb = cluster.get("rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            continue
        try:
            weight = float(cluster.get("proportion", 0.0) or 0.0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0:
            continue

        red, green, blue = (max(0, min(255, int(c))) / 255.0 for c in rgb)
        hue, _, _ = colorsys.rgb_to_hls(red, green, blue)
        hue_degrees = hue * 360.0
        _, hsv_saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
        if hsv_saturation < NEUTRAL_SATURATION:
            temperature = "neutral"
            neutral_mass += weight
        else:
            temperature = classify_hue(hue_degrees)
            if temperature == "warm":
                warm_mass += weight
            else:
                cool_mass += weight

        detail.append({
            "rgb": [int(c) for c in rgb],
            "hue": round(hue_degrees, 1),
            "saturation": round(hsv_saturation, 3),
            "proportion": round(weight, 2),
            "temperature": temperature,
        })

    coloured = warm_mass + cool_mass
    total = coloured + neutral_mass
    if total <= 0:
        return {"measured": None, "warm_fraction": None, "clusters": detail,
                "reason": "no usable colour clusters"}

    neutral_fraction = round(neutral_mass / total, 3)
    if coloured / total < MIN_COLOURED_SHARE:
        return {"measured": None, "warm_fraction": None,
                "neutral_fraction": neutral_fraction, "clusters": detail,
                "reason": (f"the palette is {neutral_fraction:.0%} neutral (grey, black, "
                           f"white); too little colour to judge warm or cool")}

    warm_fraction = warm_mass / coloured
    if warm_fraction >= WARM_MAJORITY:
        measured = "warm"
    elif (1.0 - warm_fraction) >= COOL_MAJORITY:
        measured = "cool"
    else:
        measured = "mixed"

    return {
        "measured": measured,
        "warm_fraction": round(warm_fraction, 3),
        "neutral_fraction": neutral_fraction,
        "neutral_saturation": NEUTRAL_SATURATION,
        "majority_threshold": WARM_MAJORITY,
        "hue_boundary": {"warm": list(WARM_ARCS), "cool": list(COOL_ARC)},
        "boundary_source": "color_vocabulary.csv Warm/Cool labels",
        "clusters": detail,
    }


def verify(declared: Optional[str], clusters: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Check a declared temperature against the palette.

    A declaration outside warm/cool/mixed is reported as invalid, not guessed at.
    """
    result = measure(clusters)
    result["declared"] = declared

    normalised = str(declared or "").strip().lower()
    if normalised not in DECLARED_VALUES:
        result["holds"] = None
        result["status"] = "undeclared" if not normalised else "invalid_declaration"
        result["reason"] = (
            f"'{declared}' is not one of {'/'.join(DECLARED_VALUES)}"
            if normalised else "no colour temperature declared"
        )
        return result

    if result["measured"] is None:
        result["holds"] = None
        result["status"] = "not_checked"
        return result

    result["holds"] = result["measured"] == normalised
    result["status"] = "checked"
    return result
