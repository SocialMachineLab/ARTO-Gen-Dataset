"""
The measurements block: per-instance image measurements used by Stage 4 (knowledge graph assembly).

  bbox               from the instance assignment (instance_assignment)
  representative_rgb from colour analysis (color_validator.extract_colors_from_region)

Keyed by `instance_id`, so instances sharing a name stay separate.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence

from . import instance_assignment as ia


def _representative_rgb(palette: Optional[Sequence[Dict[str, Any]]]) -> Optional[List[int]]:
    """The largest cluster of a K-means palette, as plain ints.

    The palette is sorted by proportion. Returns None when nothing usable is
    present.
    """
    if not palette:
        return None
    first = palette[0]
    if not isinstance(first, dict):
        return None
    rgb = first.get("rgb")
    if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
        return None
    try:
        return [int(channel) for channel in rgb]
    except (TypeError, ValueError):
        return None


def build(instances: Sequence[Dict[str, Any]],
          detections: Sequence[Dict[str, Any]],
          palette_of: Optional[Callable[[List[float]], Any]] = None) -> Dict[str, Any]:
    """Assemble the measurements block.

    `palette_of` maps a bbox to a K-means palette; None skips colour.
    Instances detected but not tied to a box keep their entry with `bbox: null`.
    """
    assignment = ia.assign(instances, detections)

    objects: List[Dict[str, Any]] = []
    for record in assignment["objects"]:
        bbox = record["bbox"]
        entry: Dict[str, Any] = {
            "instance_id": record["instance_id"],
            "bbox": bbox,
            "detection_relation": record["relation"],
            "detection_status": record["status"],
            "representative_rgb": None,
        }
        if bbox is not None and palette_of is not None:
            try:
                entry["representative_rgb"] = _representative_rgb(palette_of(bbox))
            except Exception:
                # An unmeasurable colour stays None; the rest of the block is kept.
                entry["representative_rgb"] = None
        objects.append(entry)

    return {
        "objects": objects,
        "summary": {
            "total": assignment["total"],
            "detected": assignment["detected"],
            "not_localised": assignment["not_localised"],
            "missing": assignment["missing"],
            "exact": assignment["exact"],
            "parent_of_gt": assignment["parent_of_gt"],
            # Each object is queried by name and COCO category, so raw
            # detections can exceed distinct objects.
            "raw_detections": assignment["raw_detections"],
            "distinct_objects": assignment["distinct_objects"],
        },
        # Object-presence gate: presence, not localisation.
        "all_present": assignment["passed"],
    }


def bboxes_by_instance(measurements: Dict[str, Any]) -> Dict[str, List[float]]:
    """{instance_id: bbox} for the instances that have one.

    Instances without a box are absent, so `.get()` returns None (`not_checked`).
    """
    return {entry["instance_id"]: entry["bbox"]
            for entry in measurements.get("objects", [])
            if entry.get("bbox") is not None}


def duplicate_names(instances: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    """Names shared by more than one instance in the same artwork.

    Reported because prompts that refer to objects by name become ambiguous.
    """
    seen: Dict[str, List[str]] = {}
    for instance in instances:
        name = str(instance.get("name") or "").strip().lower()
        if name:
            seen.setdefault(name, []).append(instance.get("instance_id"))
    return {name: ids for name, ids in seen.items() if len(ids) > 1}
