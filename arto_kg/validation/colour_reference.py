"""
Resolve a colour phrase (e.g. "muted teal") to a reference RGB.

The base colour comes from the vocabulary; each modifier word then shifts it
along the axis it names. The modifier coefficients are fixed heuristic
values.
"""

import colorsys
import re
from typing import Dict, List, Optional, Sequence, Tuple

RGB = Tuple[int, int, int]

# Multiplier applied to HLS saturation. Below 1 desaturates.
SATURATION = {
    "muted": 0.55, "faded": 0.50, "dusty": 0.60, "dull": 0.60,
    "weathered": 0.65, "aged": 0.70, "subdued": 0.55, "washed": 0.50,
    "greyish": 0.45, "grayish": 0.45, "desaturated": 0.45,
    "vibrant": 1.30, "rich": 1.20, "brilliant": 1.35, "vivid": 1.35,
    "intense": 1.30, "saturated": 1.30, "bold": 1.20,
}

# Multiplier applied to HLS lightness. Below 1 darkens.
LIGHTNESS = {
    "deep": 0.75, "dark": 0.70, "shadowed": 0.70, "shadowy": 0.72,
    "midnight": 0.55, "charcoal": 0.60,
    "pale": 1.35, "light": 1.25, "soft": 1.12, "creamy": 1.20,
    "milky": 1.30, "bright": 1.15, "luminous": 1.20, "glowing": 1.18,
    "sunlit": 1.15,
}

# Hue target (degrees) a temperature word pulls towards, and the fraction of
# the way it moves: "warm gray" is a grey nudged towards orange.
TEMPERATURE = {
    "warm": (30.0, 0.18),
    "cool": (210.0, 0.18),
    "cold": (210.0, 0.22),
}

# Words that are part of a colour name, never treated as modifiers.
NOT_MODIFIERS = {"golden", "silver", "rose", "olive", "sky", "sea", "forest",
                 "navy", "royal", "burnt", "ocean", "wine", "blood", "steel"}


def _split_phrase(phrase: str) -> Tuple[List[str], str]:
    """(modifiers, remaining phrase) for a colour description."""
    words = [w for w in re.split(r"[^a-z]+", str(phrase or "").lower()) if w]
    modifiers = []
    rest = []
    for word in words:
        if word in NOT_MODIFIERS:
            rest.append(word)
        elif word in SATURATION or word in LIGHTNESS or word in TEMPERATURE:
            modifiers.append(word)
        else:
            rest.append(word)
    return modifiers, " ".join(rest)


def modifiers_in(phrase: str) -> List[str]:
    """Which modifier words a phrase carries."""
    return _split_phrase(phrase)[0]


def base_term(phrase: str) -> str:
    """The phrase with its modifiers removed - what to look up."""
    return _split_phrase(phrase)[1]


def apply_modifiers(rgb: RGB, modifiers: Sequence[str]) -> RGB:
    """Shift a base colour in HLS along the axes its modifiers name.

    Modifiers compose multiplicatively ("deep muted teal" darkens and desaturates).
    """
    if not modifiers:
        return rgb

    red, green, blue = (max(0, min(255, int(c))) / 255.0 for c in rgb)
    hue, lightness, saturation = colorsys.rgb_to_hls(red, green, blue)

    for word in modifiers:
        if word in SATURATION:
            saturation *= SATURATION[word]
        if word in LIGHTNESS:
            lightness *= LIGHTNESS[word]
        if word in TEMPERATURE:
            target, weight = TEMPERATURE[word]
            # Move along the shorter way round the hue circle.
            current = hue * 360.0
            delta = ((target - current + 180.0) % 360.0) - 180.0
            hue = ((current + delta * weight) % 360.0) / 360.0

    saturation = max(0.0, min(1.0, saturation))
    lightness = max(0.0, min(1.0, lightness))

    red, green, blue = colorsys.hls_to_rgb(hue, lightness, saturation)
    return (int(round(red * 255)), int(round(green * 255)), int(round(blue * 255)))


def resolve(phrase: str, lookup, provenance=None) -> Optional[Dict[str, object]]:
    """Reference colour for a phrase: base term looked up, modifiers applied.

    `lookup` maps a colour name to an RGB tuple or None. `provenance`, if given,
    reports which tier answered and whether it states the colour verbatim; a
    non-authoritative verbatim hit is then set aside when the phrase has modifiers.
    Returns a record describing how the reference was obtained.
    """
    text = str(phrase or "").strip()

    # A hex value is used directly, without the phrase parser.
    if text.startswith("#"):
        direct = lookup(text)
        if direct is not None:
            return {
                "phrase": phrase,
                "base_term": text,
                "base_rgb": [int(c) for c in direct],
                "modifiers": [],
                "rgb": [int(c) for c in direct],
                "adjusted": False,
                "source": "hex",
            }
        return None

    # Look up the whole phrase first, so a vocabulary name such as "Dark Brown"
    # keeps its stated RGB instead of being rebuilt from "brown".
    exact = lookup(text.lower())
    verbatim_source = provenance(text.lower()) if provenance and exact is not None else None

    # Keep the verbatim answer unless it is a reconstruction that dropped a
    # modifier this phrase carries.
    if exact is not None:
        rejected = (verbatim_source is not None
                    and not verbatim_source.get("authoritative")
                    and modifiers_in(phrase))
        if not rejected:
            record = {
                "phrase": phrase,
                "base_term": text,
                "base_rgb": [int(c) for c in exact],
                "modifiers": [],
                "rgb": [int(c) for c in exact],
                "adjusted": False,
                "method": "exact",
            }
            if verbatim_source is not None:
                record["source_tier"] = verbatim_source.get("tier")
                record["source_method"] = verbatim_source.get("method")
            return record

    # Fallback for free colour phrases: look up the base term, apply modifiers.
    modifiers, base = _split_phrase(phrase)

    base_rgb = lookup(base) if base else None
    if base_rgb is None:
        # Base term unknown: return the set-aside verbatim hit, marked as such.
        if exact is not None:
            return {
                "phrase": phrase,
                "base_term": text,
                "base_rgb": [int(c) for c in exact],
                "modifiers": [],
                "rgb": [int(c) for c in exact],
                "adjusted": False,
                "method": "exact_reconstructed",
                "source_tier": (verbatim_source or {}).get("tier"),
                "source_method": (verbatim_source or {}).get("method"),
                "note": ("verbatim hit is a reconstruction and the base term "
                         "does not resolve, so the modifiers could not be applied"),
            }
        return None

    adjusted = apply_modifiers(tuple(base_rgb), modifiers)
    record = {
        "phrase": phrase,
        "base_term": base,
        "base_rgb": [int(c) for c in base_rgb],
        "modifiers": modifiers,
        "rgb": list(adjusted),
        "adjusted": adjusted != tuple(int(c) for c in base_rgb),
        "method": "modifiers" if modifiers else "base_term",
    }
    if exact is not None and verbatim_source is not None:
        # Record the verbatim answer that was set aside.
        record["verbatim_rejected"] = {
            "rgb": [int(c) for c in exact],
            "tier": verbatim_source.get("tier"),
            "method": verbatim_source.get("method"),
        }
    return record
