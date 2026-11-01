"""Semantic relation validation with the VLM, one directed relation at a time.

Both endpoints are drawn and labelled A and B, since two instances can share a
name, and the model chooses between the stated direction, its reverse and
neither (`direction_choice`). A specification with no semantic relations is
`not_applicable`.
"""

import logging
import os
from typing import Any, Dict, Optional

from .. import box_annotation
from .. import direction_choice as dc
from .. import instance_assignment as ia

logger = logging.getLogger(__name__)

MAX_TOKENS = 320

CHECKED = "checked"
NOT_LOCALISED = "not_localised"
ERROR = "error"


def _build_prompt(subject_name: str, object_name: str, relation: str,
                  subject_mark: Dict[str, Any], object_mark: Dict[str, Any],
                  key: str):
    """State what each letter marks, then offer both directions (or, for a
    symmetric relation, the pair) and neither.

    Returns the prompt and {option number: meaning}.
    """
    a_label, a_colour = subject_mark["label"], subject_mark["colour_name"]
    b_label, b_colour = object_mark["label"], object_mark["colour_name"]
    choices, meaning = dc.options(f"{subject_name} in box {a_label}",
                                  f"{object_name} in box {b_label}", relation, key,
                                  relation=relation)
    directions = ("" if dc.is_symmetric(relation) else
                  "The two directions are different claims; choose the one the painting "
                  "shows. ")

    prompt = (
        f"Two rectangles have been drawn on top of this painting as "
        "annotations. They are not part of the artwork.\n\n"
        f"- Box {a_label} ({a_colour}) marks the {subject_name}.\n"
        f"- Box {b_label} ({b_colour}) marks the {object_name}.\n\n"
        "Which of these is true in the painting?\n"
        f"{choices}\n\n"
        f"{directions}Base your answer on what the two marked objects are doing in "
        "the painting, not on what is plausible.\n\n"
        "Answer in this JSON format and nothing else:\n"
        '{"observed": "<what the two marked objects are actually doing>", '
        '"choice": <the option number>, '
        '"confidence": <0.0 to 1.0>}\n\n'
        'If either box is too small or unclear to judge, set observed to '
        '"unclear" and choose the last option.'
    )
    return prompt, meaning


class VLMSemanticValidator:
    """Directed semantic relations, checked one at a time on a marked image."""

    def __init__(self, vlm_wrapper, work_dir: str = "annotations"):
        if vlm_wrapper is None:
            raise ValueError(
                "VLMSemanticValidator needs a VLM wrapper: semantic relations "
                "cannot be decided from geometry."
            )
        self.vlm = vlm_wrapper
        self.work_dir = work_dir

    def main_evaluation(self, json_data: Dict[str, Any], image_path: str,
                        measurement_block: Dict[str, Any]) -> Dict[str, Any]:
        relations = (json_data.get("composition") or {}).get("semantic_relations") or []
        if not relations:
            return {"status": "not_applicable", "relations": [], "score": None,
                    "checked": 0,
                    "reason": "specification declares no semantic relations",
                    "method": "vlm_marked_pair"}

        names = ia.names_by_instance(json_data)
        boxes = {
            entry["instance_id"]: entry.get("bbox")
            for entry in measurement_block.get("objects", [])
        }
        # Triple-shaped relations cite the numeric object_id, which must be
        # resolved to an instance_id before the box lookup.
        by_object_id = ia.instance_by_object_id(json_data)

        results = [self._check_one(image_path, relation, names, boxes, index,
                                   by_object_id)
                   for index, relation in enumerate(relations)]

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
            # Only answered questions are scored.
            "score": (100.0 * len(held) / len(checked)) if checked else None,
            "method": "vlm_marked_pair",
        }

    @staticmethod
    def _endpoints(relation: Any) -> tuple:
        """Return (subject, relation, object) from a dict or a bare triple."""
        if isinstance(relation, dict):
            return (relation.get("subject"), relation.get("relation"),
                    relation.get("object"))
        if isinstance(relation, (list, tuple)) and len(relation) >= 3:
            return (relation[0], relation[1], relation[2])
        return (None, None, None)

    def _check_one(self, image_path: str, relation: Any,
                   names: Dict[str, str], boxes: Dict[str, Any],
                   index: int,
                   by_object_id: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        subject_id, raw_predicate, object_id = self._endpoints(relation)
        predicate = str(raw_predicate or "").strip()

        # Map numeric object_ids from the triple shape to instance_ids.
        if by_object_id:
            if subject_id not in boxes:
                subject_id = by_object_id.get(str(subject_id), subject_id)
            if object_id not in boxes:
                object_id = by_object_id.get(str(object_id), object_id)

        record: Dict[str, Any] = {
            "subject": subject_id,
            "relation": predicate,
            "object": object_id,
        }

        if not (subject_id and object_id and predicate):
            record.update({"status": ERROR, "reason": "incomplete relation record"})
            return record
        if subject_id == object_id:
            record.update({"status": ERROR,
                           "reason": "subject and object are the same instance"})
            return record

        subject_box, object_box = boxes.get(subject_id), boxes.get(object_id)
        if not subject_box or not object_box:
            missing = [i for i, b in ((subject_id, subject_box), (object_id, object_box))
                       if not b]
            record.update({"status": NOT_LOCALISED,
                           "reason": f"no bounding box for {', '.join(missing)}"})
            return record

        subject_name = names.get(subject_id, "object")
        object_name = names.get(object_id, "object")

        out_path = os.path.join(self.work_dir, f"semantic_{index}_{subject_id}_{object_id}.png")
        annotated = box_annotation.annotate(image_path, [
            {"bbox": subject_box, "label": "A", "instance_id": subject_id,
             "name": subject_name},
            {"bbox": object_box, "label": "B", "instance_id": object_id,
             "name": object_name},
        ], out_path)

        if annotated is None or len(annotated["marks"]) != 2:
            record.update({"status": ERROR,
                           "reason": "could not mark both endpoints"})
            return record

        subject_mark, object_mark = annotated["marks"]
        record.update({
            "subject_name": subject_name,
            "object_name": object_name,
            "subject_bbox": subject_mark["bbox"],
            "object_bbox": object_mark["bbox"],
            "possibly_illegible": (subject_mark["possibly_illegible"]
                                   or object_mark["possibly_illegible"]),
        })

        prompt, meaning = _build_prompt(subject_name, object_name, predicate,
                                        subject_mark, object_mark,
                                        f"{subject_id}|{predicate}|{object_id}")
        result = self.vlm.generate_response(prompt, annotated["path"],
                                            max_tokens=MAX_TOKENS)

        if not result.get("success"):
            record.update({"status": ERROR, "reason": result.get("error")})
            return record

        verdict = dc.parse_choice(result.get("response", ""), meaning)
        if verdict is None:
            record.update({"status": ERROR, "reason": "unparseable reply",
                           "raw_response": str(result.get("response", ""))[:200]})
            return record

        record.update({"status": CHECKED, **verdict})
        return record
