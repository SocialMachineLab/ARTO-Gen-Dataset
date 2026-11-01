"""
Read object-detection result files in any of their supported layouts.
"""

from typing import Any, Dict, List, Tuple


def _detection_list(od_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Find the detection array, whichever layout this file uses."""
    if not isinstance(od_data, dict):
        return []

    # Layout produced by the object detection stage.
    detections = od_data.get("detected_objects")
    if isinstance(detections, list):
        return detections

    # Fused result wrapper.
    detailed = od_data.get("detailed_results")
    if isinstance(detailed, dict):
        detections = detailed.get("detection", {}).get("detected_objects")
        if isinstance(detections, list):
            return detections

    # Flat layout.
    detections = od_data.get("detections")
    if isinstance(detections, list):
        return detections

    return []


def detection_list(od_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The raw detection records of a result file, whichever layout it uses."""
    return list(_detection_list(od_data))


def _box_of(detection: Dict[str, Any]):
    box = detection.get("box") or detection.get("bbox")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return [float(v) for v in box]
        except (TypeError, ValueError):
            return None
    return None


def _key_of(detection: Dict[str, Any]):
    """Which specification object this detection stands for.

    ``mapped_gt`` (the specification's word, e.g. "urn") is preferred over the
    detector's ``label`` (e.g. "vase"), since validators look objects up by it.
    """
    for field in ("instance_id", "mapped_gt", "class_name", "label", "text_label"):
        value = detection.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_detections(od_data: Dict[str, Any]) -> Tuple[Dict[str, List[float]], List[Dict[str, Any]]]:
    """Return ({key: bbox}, normalised detection records).

    Keys are lowercased. The dict keeps only the first detection per key; the
    list keeps every detection.
    """
    bboxes: Dict[str, List[float]] = {}
    records: List[Dict[str, Any]] = []

    for detection in _detection_list(od_data):
        if not isinstance(detection, dict):
            continue
        box = _box_of(detection)
        key = _key_of(detection)
        if box is None or key is None:
            continue

        record = {
            "key": key.lower(),
            "box": box,
            "instance_id": detection.get("instance_id"),
            "mapped_gt": detection.get("mapped_gt"),
            "label": detection.get("label") or detection.get("class_name"),
            "relation": detection.get("relation"),
            # Query term that produced the detection; defaults to the spec's word.
            "query": detection.get("query") or detection.get("mapped_gt") or key,
            "score": detection.get("score") or detection.get("confidence") or 0.0,
        }
        records.append(record)
        bboxes.setdefault(record["key"], box)

    return bboxes, records


def duplicate_keys(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Keys carrying more than one detection.

    These keys are ambiguous in the flat {key: bbox} view; instance_assignment
    resolves them by instance.
    """
    counts: Dict[str, int] = {}
    for record in records:
        counts[record["key"]] = counts.get(record["key"], 0) + 1
    return {key: count for key, count in counts.items() if count > 1}
