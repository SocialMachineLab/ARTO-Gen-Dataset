"""Per-instance object state validation with the VLM.

Each instance is asked about separately, with its own box drawn on the image,
and keyed by `instance_id`. Instances without a usable box are recorded as
`not_localised` and do not enter the score.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from .. import box_annotation
from .. import instance_assignment as ia

logger = logging.getLogger(__name__)

MAX_TOKENS = 320

# Outcome vocabulary, matching the rest of Stage 3.
CHECKED = "checked"
NOT_LOCALISED = "not_localised"
NOT_DECLARED = "not_declared"
ERROR = "error"


def _build_prompt(name: str, expected_state: str, colour_name: str = "red",
                  label: str = "A") -> str:
    """Build a prompt about the marked object only, asking for a JSON verdict.

    `colour_name` comes from the drawn annotation. The prompt states the box is
    an annotation so the model does not treat it as part of the artwork.
    """
    return (
        f"A {colour_name} rectangle labelled {label} has been drawn on top of "
        "this painting as an annotation. It is not part of the artwork.\n\n"
        f"Box {label} marks one {name}.\n"
        f"Expected state of the {name} inside box {label}: {expected_state}\n\n"
        f"Look only at the object inside box {label}. Ignore every other "
        f"object, including any other {name} elsewhere in the painting, and "
        "ignore the rectangle itself when describing what you see.\n\n"
        "Answer in this JSON format and nothing else:\n"
        '{"observed_state": "<what you actually see>", '
        '"matches": true or false, '
        '"confidence": <0.0 to 1.0>}\n\n'
        "Set matches to false if the object's state differs from the expected "
        f"one, and say what you see either way. If box {label} is too small or "
        'unclear to judge, set observed_state to "unclear" and matches to false.'
    )


def parse_verdict(response: str) -> Optional[Dict[str, Any]]:
    """Extract the JSON verdict from a reply, or None if it cannot be read."""
    if not response:
        return None

    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None

    try:
        parsed = json.loads(candidate)
    except (ValueError, TypeError):
        return None
    if not isinstance(parsed, dict) or "matches" not in parsed:
        return None

    matches = parsed.get("matches")
    if isinstance(matches, str):
        matches = matches.strip().lower() in ("true", "yes")
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return {
        "observed_state": str(parsed.get("observed_state", ""))[:300],
        "matches": bool(matches),
        "confidence": max(0.0, min(1.0, confidence)),
    }


class VLMStateValidator:
    """Per-instance state checking against a marked image."""

    def __init__(self, vlm_wrapper, work_dir: str = "annotations"):
        if vlm_wrapper is None:
            raise ValueError(
                "VLMStateValidator needs a VLM wrapper; state cannot be judged "
                "geometrically."
            )
        self.vlm = vlm_wrapper
        # Annotated images are scratch files, one per instance.
        self.work_dir = work_dir

    def main_evaluation(self, json_data: Dict[str, Any], image_path: str,
                        measurement_block: Dict[str, Any]) -> Dict[str, Any]:
        """One verdict per instance that declares a state and has a box."""
        instances = dict(ia.iter_objects(json_data))
        boxes = {
            entry["instance_id"]: entry.get("bbox")
            for entry in measurement_block.get("objects", [])
        }

        results: List[Dict[str, Any]] = []
        for instance_id, instance in instances.items():
            expected = str(instance.get("state") or "").strip()
            name = str(instance.get("name") or "object")

            if not expected:
                results.append({"instance_id": instance_id, "name": name,
                                "status": NOT_DECLARED})
                continue

            bbox = boxes.get(instance_id)
            if not bbox:
                # Without a box the question cannot target a specific object.
                results.append({"instance_id": instance_id, "name": name,
                                "expected_state": expected,
                                "status": NOT_LOCALISED,
                                "reason": "no bounding box for this instance"})
                continue

            results.append(self._check_one(image_path, instance_id, name, expected, bbox))

        checked = [r for r in results if r["status"] == CHECKED]
        matched = [r for r in checked if r.get("matches")]

        return {
            "objects": results,
            "total": len(results),
            "checked": len(checked),
            "matched": len(matched),
            "not_localised": len([r for r in results if r["status"] == NOT_LOCALISED]),
            "not_declared": len([r for r in results if r["status"] == NOT_DECLARED]),
            "errors": len([r for r in results if r["status"] == ERROR]),
            # Only answered questions are scored; unlocatable and undeclared
            # instances never count as failures.
            "score": (100.0 * len(matched) / len(checked)) if checked else None,
            "method": "vlm_marked_instance",
        }

    def _check_one(self, image_path: str, instance_id: str, name: str,
                   expected: str, bbox: List[float]) -> Dict[str, Any]:
        out_path = os.path.join(self.work_dir, f"state_{instance_id}.png")
        annotated = box_annotation.annotate(
            image_path, [{"bbox": bbox, "label": "A", "instance_id": instance_id}],
            out_path)

        if annotated is None:
            return {"instance_id": instance_id, "name": name,
                    "expected_state": expected, "status": ERROR,
                    "reason": "could not annotate the image"}

        mark = annotated["marks"][0]
        result = self.vlm.generate_response(
            _build_prompt(name, expected, mark["colour_name"], mark["label"]),
            annotated["path"], max_tokens=MAX_TOKENS)

        record: Dict[str, Any] = {
            "instance_id": instance_id,
            "name": name,
            "expected_state": expected,
            "bbox": mark["bbox"],
            "relative_area": mark["relative_area"],
            # Flags boxes under 1% of the canvas, which may be too small to judge.
            "possibly_illegible": mark["possibly_illegible"],
        }

        if not result.get("success"):
            record.update({"status": ERROR, "reason": result.get("error")})
            return record

        verdict = parse_verdict(result.get("response", ""))
        if verdict is None:
            record.update({"status": ERROR, "reason": "unparseable reply",
                           "raw_response": str(result.get("response", ""))[:200]})
            return record

        record.update({"status": CHECKED, **verdict})
        return record
