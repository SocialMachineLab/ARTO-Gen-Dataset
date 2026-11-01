"""
Perceptual colour difference: sRGB -> CIELAB -> CIEDE2000.

Thresholds follow the standard CIE reading of the CIEDE2000 scale.
"""

import math
from typing import Dict, Sequence, Tuple

RGB = Sequence[float]

# D65, 2 degree observer.
_WHITE = (95.047, 100.000, 108.883)

# Standard reading of the CIEDE2000 scale.
JND = 1.0            # below this a difference is not perceptible
CLOSE_INSPECTION = 2.0   # perceptible only on close inspection
SIMILAR = 10.0       # upper bound of the 'similar' band

# Matched when Delta E <= MATCH_THRESHOLD (upper bound of the 'similar' band).
MATCH_THRESHOLD = SIMILAR


def srgb_to_xyz(rgb: RGB) -> Tuple[float, float, float]:
    def linear(channel: float) -> float:
        c = max(0.0, min(1.0, channel / 255.0))
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = (linear(v) for v in rgb[:3])
    x = (r * 0.4124564 + g * 0.3575761 + b * 0.1804375) * 100.0
    y = (r * 0.2126729 + g * 0.7151522 + b * 0.0721750) * 100.0
    z = (r * 0.0193339 + g * 0.1191920 + b * 0.9503041) * 100.0
    return x, y, z


def xyz_to_lab(xyz: Sequence[float]) -> Tuple[float, float, float]:
    def f(t: float) -> float:
        return t ** (1.0 / 3.0) if t > 0.008856 else (7.787 * t) + (16.0 / 116.0)

    fx, fy, fz = (f(v / w) for v, w in zip(xyz, _WHITE))
    return (116.0 * fy) - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


def rgb_to_lab(rgb: RGB) -> Tuple[float, float, float]:
    return xyz_to_lab(srgb_to_xyz(rgb))


def ciede2000(lab1: Sequence[float], lab2: Sequence[float]) -> float:
    """CIEDE2000 colour difference (CIE 142-2001), kL = kC = kH = 1."""
    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    c1 = math.hypot(a1, b1)
    c2 = math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0

    g = 0.5 * (1 - math.sqrt((c_bar ** 7) / (c_bar ** 7 + 25.0 ** 7))) if c_bar > 0 else 0.0
    a1p, a2p = (1 + g) * a1, (1 + g) * a2

    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)

    def hue(ap: float, bp: float) -> float:
        if ap == 0 and bp == 0:
            return 0.0
        angle = math.degrees(math.atan2(bp, ap))
        return angle + 360.0 if angle < 0 else angle

    h1p, h2p = hue(a1p, b1), hue(a2p, b2)

    dlp = l2 - l1
    dcp = c2p - c1p

    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360.0
    else:
        dhp = h2p - h1p + 360.0
    dHp = 2.0 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp) / 2.0)

    lp_bar = (l1 + l2) / 2.0
    cp_bar = (c1p + c2p) / 2.0

    if c1p * c2p == 0:
        hp_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hp_bar = (h1p + h2p) / 2.0
    elif h1p + h2p < 360:
        hp_bar = (h1p + h2p + 360.0) / 2.0
    else:
        hp_bar = (h1p + h2p - 360.0) / 2.0

    t = (1
         - 0.17 * math.cos(math.radians(hp_bar - 30))
         + 0.24 * math.cos(math.radians(2 * hp_bar))
         + 0.32 * math.cos(math.radians(3 * hp_bar + 6))
         - 0.20 * math.cos(math.radians(4 * hp_bar - 63)))

    d_theta = 30.0 * math.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    rc = 2.0 * math.sqrt((cp_bar ** 7) / (cp_bar ** 7 + 25.0 ** 7)) if cp_bar > 0 else 0.0
    rt = -rc * math.sin(2 * math.radians(d_theta))

    sl = 1 + (0.015 * ((lp_bar - 50) ** 2)) / math.sqrt(20 + ((lp_bar - 50) ** 2))
    sc = 1 + 0.045 * cp_bar
    sh = 1 + 0.015 * cp_bar * t

    return math.sqrt(
        (dlp / sl) ** 2
        + (dcp / sc) ** 2
        + (dHp / sh) ** 2
        + rt * (dcp / sc) * (dHp / sh)
    )


def delta_e(rgb1: RGB, rgb2: RGB) -> float:
    """CIEDE2000 difference between two sRGB colours."""
    return ciede2000(rgb_to_lab(rgb1), rgb_to_lab(rgb2))


def describe(value: float) -> str:
    """Map a Delta E value to its CIE interpretation."""
    if value < JND:
        return "imperceptible"
    if value < CLOSE_INSPECTION:
        return "perceptible on close inspection"
    if value < SIMILAR:
        return "similar"
    return "different"


def compare(observed: RGB, expected: RGB) -> Dict[str, object]:
    value = delta_e(observed, expected)
    return {
        "delta_e": round(value, 3),
        "metric": "CIEDE2000",
        "interpretation": describe(value),
        "matched": value <= MATCH_THRESHOLD,
    }
