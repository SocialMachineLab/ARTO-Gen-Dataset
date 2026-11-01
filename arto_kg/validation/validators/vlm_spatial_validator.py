"""VLM channel for spatial relations that 2D boxes cannot decide.

Answers the relations `spatial_predicates` marks `deferred` (depth, occlusion,
part-whole, orientation). Both endpoints are drawn and labelled, and the model
chooses between the stated direction, its reverse and neither
(`direction_choice`).
"""

import logging
import os
from typing import Any, Dict

from .. import box_annotation
from .. import direction_choice as dc
from .. import instance_assignment as ia
from .. import spatial_predicates as sp

logger = logging.getLogger(__name__)

MAX_TOKENS = 320

CHECKED = "checked"
NOT_LOCALISED = "not_localised"
ERROR = "error"

# Explicit wording for deferred relations, since bare predicates such as
# "through" or "against" are ambiguous.
RELATION_PHRASING = {
    "behind": "further from the viewer than",
    "in front of": "closer to the viewer than",
    "through": "passing through",
    "against": "touching and leaning on",
    "opposite": "on the opposite side, facing",
    "part of": "a part of",
    "in": "contained in or set within",
    "around": "encircling or wrapped around",
}


def _build_prompt(subject_name: str, object_name: str, relation: str,
                  subject_mark: Dict[str, Any], object_mark: Dict[str, Any],
                  key: str):
    """The prompt and {option number: meaning} for one relation."""
    a_label, a_colour = subject_mark["label"], subject_mark["colour_name"]
    b_label, b_colour = object_mark["label"], object_mark["colour_name"]
    phrasing = RELATION_PHRASING.get(sp.normalise(relation), relation)
    choices, meaning = dc.options(f"{subject_name} in box {a_label}",
                                  f"{object_name} in box {b_label}", phrasing, key,
                                  relation=relation)
    directions = ("" if dc.is_symmetric(relation) else
                  "The two directions are different claims; choose the one the painting "
                  "shows. ")

    prompt = (
        "Two rectangles have been drawn on top of this painting as "
        "annotations. They are not part of the artwork.\n\n"
        f"- Box {a_label} ({a_colour}) marks the {subject_name}.\n"
        f"- Box {b_label} ({b_colour}) marks the {object_name}.\n\n"
        "Which of these is true in the painting?\n"
        f"{choices}\n\n"
        f"{directions}Use depth cues in the painting such as occlusion, overlap, "
        "scale and perspective. Where the boxes sit on the flat image does "
        "not settle this.\n\n"
        "Answer in this JSON format and nothing else:\n"
        '{"observed": "<the actual spatial arrangement of the two marked '
        'objects>", "choice": <the option number>, "confidence": <0.0 to 1.0>}\n\n'
        "If either box is too small or unclear to judge, set observed to "
        '"unclear" and choose the last option.'
    )
    return prompt, meaning


class VLMSpatialValidator:
    """Answers spatial relations deferred by the geometric pass."""

    def __init__(self, vlm_wrapper, work_dir: str = "annotations"):
        if vlm_wrapper is None:
            raise ValueError(
                "VLMSpatialValidator needs a VLM wrapper. Depth relations are "
                "not decidable from 2D boxes, which is why they were deferred."
            )
        self.vlm = vlm_wrapper
        self.work_dir = work_dir

    def resolve_deferred(self, json_data: Dict[str, Any], image_path: str,
                         measurement_block: Dict[str, Any],
                         geometric_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Answer every relation the geometric pass marked `deferred`."""
        deferred = [r for r in geometric_summary.get("details", [])
                    if r.get("status") == "deferred"]
        if not deferred:
            return {"status": "not_applicable", "relations": [], "checked": 0,
                    "score": None, "method": "vlm_marked_pair",
                    "reason": "no relation needed the VLM channel"}

        # Fallback lookups for records that carry no boxes.
        names = ia.names_by_instance(json_data)
        boxes = {entry["instance_id"]: entry.get("bbox")
                 for entry in measurement_block.get("objects", [])}

        results = [self._check_one(image_path, record, names, boxes, index)
                   for index, record in enumerate(deferred)]

        checked = [r for r in results if r["status"] == CHECKED]
        held = [r for r in checked if r.get("holds")]

        return {
            "status": "checked" if checked else "not_checked",
            "relations": results,
            "total": len(results),
            "checked": len(checked),
            "held": len(held),
            "not_localised": len([r for r in results if r["status"] == NOT_LOCALISED]),
            "errors": len([r for r in results if r["status"] == ERROR]),
            "score": (100.0 * len(held) / len(checked)) if checked else None,
            "method": "vlm_marked_pair",
        }

    def _check_one(self, image_path: str, record: Dict[str, Any],
                   names: Dict[str, str], boxes: Dict[str, Any],
                   index: int) -> Dict[str, Any]:
        subject_id = record.get("subject")
        object_id = record.get("object")
        predicate = str(record.get("relation") or "").strip()

        out: Dict[str, Any] = {
            "subject": subject_id, "relation": predicate, "object": object_id,
            "channel": "vlm",
        }

        # Prefer boxes resolved by the geometric pass (instance ids or object
        # names), else the instance-id boxes.
        subject_box = record.get("subject_box") or boxes.get(subject_id)
        object_box = record.get("object_box") or boxes.get(object_id)
        if not subject_box or not object_box:
            out.update({"status": NOT_LOCALISED,
                        "reason": "endpoint without a bounding box"})
            return out

        subject_name = record.get("subject_name") or names.get(subject_id, "object")
        object_name = record.get("object_name") or names.get(object_id, "object")

        out_path = os.path.join(
            self.work_dir, f"spatial_{index}_{subject_id}_{object_id}.png")
        annotated = box_annotation.annotate(image_path, [
            {"bbox": subject_box, "label": "A", "instance_id": subject_id},
            {"bbox": object_box, "label": "B", "instance_id": object_id},
        ], out_path)

        if annotated is None or len(annotated["marks"]) != 2:
            out.update({"status": ERROR, "reason": "could not mark both endpoints"})
            return out

        subject_mark, object_mark = annotated["marks"]
        out.update({
            "subject_name": subject_name,
            "object_name": object_name,
            "possibly_illegible": (subject_mark["possibly_illegible"]
                                   or object_mark["possibly_illegible"]),
        })

        prompt, meaning = _build_prompt(subject_name, object_name, predicate,
                                        subject_mark, object_mark,
                                        f"{subject_id}|{predicate}|{object_id}")
        result = self.vlm.generate_response(prompt, annotated["path"],
                                            max_tokens=MAX_TOKENS)

        if not result.get("success"):
            out.update({"status": ERROR, "reason": result.get("error")})
            return out

        verdict = dc.parse_choice(result.get("response", ""), meaning)
        if verdict is None:
            out.update({"status": ERROR, "reason": "unparseable reply",
                        "raw_response": str(result.get("response", ""))[:200]})
            return out

        out.update({"status": CHECKED, **verdict})
        return out
