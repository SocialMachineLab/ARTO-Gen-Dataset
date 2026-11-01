"""
Draw labelled boxes on an image so a VLM question can refer to a specific object.

Used by state validation (one box) and by the VLM checks of spatial and semantic
relations (boxes A and B), since two instances may share a name. `relative_area`
is reported per mark so that "too small to see" can be told apart from "the VLM
disagreed".
"""

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# BGR colours with their names; saturated primaries stand out on paintings.
# Prompts use the name, so it is kept next to the colour actually drawn.
MARK_COLOURS = [
    ((0, 0, 255), "red"),
    ((255, 0, 0), "blue"),
    ((0, 255, 255), "yellow"),
    ((255, 0, 255), "magenta"),
]

# Fractions of the image's short side, so annotation scales with resolution
# instead of vanishing on a large canvas.
LINE_FRACTION = 0.004
FONT_FRACTION = 0.030
MIN_LINE = 2
MIN_FONT_SCALE = 0.5

# Below this the mark is drawn but the object is flagged as probably illegible.
TINY_AREA = 0.01


def _clamp_box(bbox: Sequence[float], width: int, height: int) -> Optional[List[int]]:
    try:
        x1, y1, x2, y2 = (int(round(float(v))) for v in bbox)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
    y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def relative_area(bbox: Sequence[float], width: int, height: int) -> Optional[float]:
    """Box area as a fraction of the image."""
    box = _clamp_box(bbox, width, height)
    if box is None or width <= 0 or height <= 0:
        return None
    return ((box[2] - box[0]) * (box[3] - box[1])) / float(width * height)


def annotate(image_path: str, marks: Sequence[Dict[str, Any]],
             out_path: str) -> Optional[Dict[str, Any]]:
    """Draw labelled boxes on a copy of the image.

    `marks` is a sequence of {"bbox": [x1, y1, x2, y2], "label": "A"}.
    Returns None if the image is unreadable or no box is usable.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("Could not read image for annotation: %s", image_path)
        return None

    height, width = image.shape[:2]
    canvas = image.copy()
    short_side = min(width, height)
    thickness = max(MIN_LINE, int(short_side * LINE_FRACTION))
    font_scale = max(MIN_FONT_SCALE, short_side * FONT_FRACTION / 30.0)

    drawn: List[Dict[str, Any]] = []
    for index, mark in enumerate(marks):
        box = _clamp_box(mark.get("bbox") or [], width, height)
        if box is None:
            continue
        colour, colour_name = MARK_COLOURS[index % len(MARK_COLOURS)]
        label = str(mark.get("label", chr(ord("A") + index)))

        cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), colour, thickness)

        # Label plate above the box, or inside it when the box touches the top edge.
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        plate_h = text_h + baseline + 4
        top = box[1] - plate_h if box[1] - plate_h >= 0 else box[1]
        cv2.rectangle(canvas, (box[0], top), (box[0] + text_w + 8, top + plate_h),
                      colour, -1)
        cv2.putText(canvas, label, (box[0] + 4, top + text_h + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

        area = ((box[2] - box[0]) * (box[3] - box[1])) / float(width * height)
        drawn.append({
            "label": label,
            "colour_name": colour_name,
            "instance_id": mark.get("instance_id"),
            "name": mark.get("name"),
            "bbox": box,
            "relative_area": round(area, 5),
            "possibly_illegible": area < TINY_AREA,
        })

    if not drawn:
        return None

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not cv2.imwrite(out_path, canvas):
        logger.warning("Could not write annotated image: %s", out_path)
        return None

    return {
        "path": out_path,
        "marks": drawn,
        "image_size": [width, height],
    }
