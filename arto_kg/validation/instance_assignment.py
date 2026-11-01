"""
Assign detection boxes to specification instances.

Each instance has a variety name (`lamb`) and a COCO category (`sheep`). Each
box is bound to at most one instance, in three rounds:

  1. variety name matched exactly              -> exact
  2. COCO fallback, only when unambiguous      -> parent_of_gt
  3. several same-category instances that
     cannot be told apart                      -> ambiguous_category (no box)

Declared relations and sizes are never used here, since they are verified later.
"""

from typing import Dict, List, Optional, Sequence

# Per-instance outcomes.
EXACT = "exact"                            # variety name detected directly
PARENT_OF_GT = "parent_of_gt"              # only the COCO category was detected
AMBIGUOUS = "ambiguous_category"           # present, but which box is unknowable
NOT_DETECTED = "not_detected"

# Per-instance status, mirroring the vocabulary the other validators use.
DETECTED = "detected"
NOT_LOCALISED = "not_localised"            # present, no usable box - not a failure
MISSING = "missing"


def _norm(text: object) -> str:
    return str(text or "").strip().lower()


def _instance_queries(instance: Dict[str, object]) -> Dict[str, str]:
    """The two query terms for one instance, in order of evidence strength."""
    return {
        EXACT: _norm(instance.get("name")),
        PARENT_OF_GT: _norm(instance.get("coco_category")),
    }


# Two boxes overlapping this much are taken to show the same physical object.
SAME_OBJECT_IOU = 0.5


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if dx <= 0 or dy <= 0:
        return 0.0
    intersection = dx * dy
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def query_categories(instances: Sequence[Dict[str, object]]) -> Dict[str, set]:
    """Query term -> the COCO categories it was asked for.

    Both of an instance's terms (variety name and COCO category) map to its
    category, so "sailboat" and "boat" share "boat".
    """
    categories: Dict[str, set] = {}
    for instance in instances:
        category = _norm(instance.get("coco_category"))
        for term in _instance_queries(instance).values():
            if term and category:
                categories.setdefault(term, set()).add(category)
    return categories


def group_detections(detections: Sequence[Dict[str, object]],
                     categories: Optional[Dict[str, set]] = None) -> List[Dict[str, object]]:
    """Collapse detections of one physical object into a single candidate.

    Boxes with IoU >= SAME_OBJECT_IOU are merged (e.g. "sailboat" and "boat").
    With `categories` (from query_categories), only detections whose terms
    share a COCO category are merged, so a coat over a person stays apart from
    the person. Each group records every query term that found it and the
    detections it merged (`merged_from`).
    """
    groups: List[Dict[str, object]] = []

    def categories_of(term: str) -> set:
        return categories.get(term) or {term}

    # Strongest detections first, so each group's box and score are its best.
    order = sorted(range(len(detections)),
                   key=lambda i: (-_score_of(detections, i), i))

    for index in order:
        detection = detections[index]
        box = detection.get("box")
        if not box:
            continue
        term = _norm(detection.get("query"))

        score = _score_of(detections, index)
        for group in groups:
            same_kind = (categories is None
                         or bool(categories_of(term) & group["categories"]))
            if same_kind and _iou(box, group["box"]) >= SAME_OBJECT_IOU:
                # Keep the best score per term; it ranks instances in binding.
                if term:
                    group["terms"][term] = max(group["terms"].get(term, 0.0), score)
                    if categories is not None:
                        group["categories"] |= categories_of(term)
                group["merged_from"].append(index)
                break
        else:
            groups.append({
                "box": box,
                "terms": {term: score} if term else {},
                "categories": set(categories_of(term)) if categories is not None else set(),
                "score": score,
                "merged_from": [index],
                "source_index": index,
            })

    return groups


def _score_of(detections: Sequence[Dict[str, object]], index: int) -> float:
    try:
        return float(detections[index].get("score", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _candidates(groups: Sequence[Dict[str, object]], term: str,
                claimed: set) -> List[int]:
    """Indices of unclaimed groups that the given term found."""
    if not term:
        return []
    return [i for i, group in enumerate(groups)
            if i not in claimed and term in group["terms"]]


def _greedy_bind(pairs: List[tuple], claimed: set, taken: set,
                 ) -> List[tuple]:
    """Highest-confidence first, one object per instance and one instance per object.

    `pairs` is (score, instance_id, group_index); ties break on ids for
    deterministic output.
    """
    bound = []
    for score, instance_id, index in sorted(
            pairs, key=lambda p: (-p[0], str(p[1]), p[2])):
        if instance_id in taken or index in claimed:
            continue
        taken.add(instance_id)
        claimed.add(index)
        bound.append((instance_id, index, score))
    return bound


def assign(instances: Sequence[Dict[str, object]],
           detections: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Bind detections to instances.

    `instances` carry `instance_id`, `name` (variety) and `coco_category`.
    `detections` carry `box`, `query` (the term that produced the detection) and
    optionally `score`.

    Returns one record per instance plus a summary; each grouped object is
    bound to at most one instance.
    """
    groups = group_detections(detections, query_categories(instances))

    results: Dict[str, Dict[str, object]] = {}
    claimed: set = set()          # group indices already bound
    taken: set = set()            # instance ids already bound

    # --- Round 1: exact variety-name match ---------------------------------
    # Ranked by the score of this instance's own term, not the group's best.
    round1_pairs = []
    for instance in instances:
        term = _instance_queries(instance)[EXACT]
        for index in _candidates(groups, term, claimed):
            round1_pairs.append(
                (groups[index]["terms"].get(term, 0.0), instance["instance_id"], index))

    for instance_id, index, score in _greedy_bind(round1_pairs, claimed, taken):
        results[instance_id] = {
            "instance_id": instance_id,
            "bbox": groups[index]["box"],
            "relation": EXACT,
            "status": DETECTED,
            "score": score,
            "detection_index": groups[index]["source_index"],
            "merged_detections": len(groups[index]["merged_from"]),
        }

    # --- Round 2: COCO fallback, only where the binding is forced ----------
    # Unmatched instances grouped by fallback category. A lone instance is bound;
    # a larger group cannot be told apart and is deferred.
    pending: Dict[str, List[Dict[str, object]]] = {}
    for instance in instances:
        if instance["instance_id"] in taken:
            continue
        pending.setdefault(_instance_queries(instance)[PARENT_OF_GT], []).append(instance)

    deferred: List[Dict[str, object]] = []
    for category, group in pending.items():
        available = _candidates(groups, category, claimed)
        if not available:
            deferred.extend(group)
            continue
        if len(group) > 1:
            deferred.extend(group)
            continue

        instance = group[0]
        # Rank by the fallback term's score for each object.
        pairs = [(groups[i]["terms"].get(category, 0.0), instance["instance_id"], i)
                 for i in available]
        bound = _greedy_bind(pairs, claimed, taken)
        if bound:
            instance_id, index, score = bound[0]
            results[instance_id] = {
                "instance_id": instance_id,
                "bbox": groups[index]["box"],
                "relation": PARENT_OF_GT,
                "status": DETECTED,
                "score": score,
                "detection_index": groups[index]["source_index"],
                "merged_detections": len(groups[index]["merged_from"]),
            }
        else:
            deferred.append(instance)

    # --- Round 3: count arbitration ---------------------------------------
    # k indistinguishable instances against j unclaimed boxes of that category.
    # min(j, k) are present but unlocatable; the remainder were never detected.
    by_category: Dict[str, List[Dict[str, object]]] = {}
    for instance in deferred:
        by_category.setdefault(_instance_queries(instance)[PARENT_OF_GT], []).append(instance)

    for category, group in by_category.items():
        available = _candidates(groups, category, claimed)
        group = sorted(group, key=lambda inst: str(inst["instance_id"]))
        confirmed = min(len(available), len(group))

        for position, instance in enumerate(group):
            instance_id = instance["instance_id"]
            if position < confirmed:
                # Claim an object but record no box: which one is unknown.
                claimed.add(available[position])
                results[instance_id] = {
                    "instance_id": instance_id,
                    "bbox": None,
                    "relation": AMBIGUOUS,
                    "status": NOT_LOCALISED,
                    "score": None,
                    "detection_index": None,
                }
            else:
                results[instance_id] = {
                    "instance_id": instance_id,
                    "bbox": None,
                    "relation": NOT_DETECTED,
                    "status": MISSING,
                    "score": None,
                    "detection_index": None,
                }

    ordered = [results[inst["instance_id"]] for inst in instances]
    detected = [r for r in ordered if r["status"] == DETECTED]
    not_localised = [r for r in ordered if r["status"] == NOT_LOCALISED]
    missing = [r for r in ordered if r["status"] == MISSING]

    return {
        "objects": ordered,
        "total": len(ordered),
        "detected": len(detected),
        "not_localised": len(not_localised),
        "missing": len(missing),
        "exact": len([r for r in ordered if r["relation"] == EXACT]),
        "parent_of_gt": len([r for r in ordered if r["relation"] == PARENT_OF_GT]),
        # Raw detections vs grouped physical objects.
        "raw_detections": len(detections),
        "distinct_objects": len(groups),
        # The gate is presence, not localisation. A specification with no
        # objects has nothing to confirm in the image and does not pass.
        "passed": bool(ordered) and len(missing) == 0,
    }


def instances_from_spec(json_data: Dict[str, object]) -> List[Dict[str, object]]:
    """The instance list a specification carries.

    Read from `objects.enhanced_objects`. A missing instance_id defaults to
    `obj_<position>`; a missing coco_category defaults to the name, and such
    records are flagged `coco_category_defaulted`.
    """
    objects = json_data.get("objects") or {}
    if not isinstance(objects, dict):
        return []

    adapted: List[Dict[str, object]] = []
    for position, obj in enumerate(objects.get("enhanced_objects") or []):
        if not isinstance(obj, dict):
            continue
        name = _norm(obj.get("name"))
        if not name:
            continue
        record: Dict[str, object] = {
            "instance_id": obj.get("instance_id") or f"obj_{position}",
            "name": obj.get("name"),
            "coco_category": obj.get("coco_category") or obj.get("name"),
            "coco_id": obj.get("coco_id", obj.get("object_id")),
        }
        if not obj.get("coco_category"):
            record["coco_category_defaulted"] = True
        adapted.append(record)
    return adapted


def iter_objects(json_data: Dict[str, object]) -> List[tuple]:
    """(instance_id, object dict) for every object in a specification.

    Returns full `objects.enhanced_objects` records, with positional ids when
    instance_id is absent, so all validators key objects the same way.
    """
    objects = json_data.get("objects") or {}
    if not isinstance(objects, dict):
        return []

    return [(obj.get("instance_id") or f"obj_{position}", obj)
            for position, obj in enumerate(objects.get("enhanced_objects") or [])
            if isinstance(obj, dict)]


def names_by_instance(json_data: Dict[str, object]) -> Dict[str, str]:
    """{instance_id: name}, for prompts and for reporting."""
    return {instance_id: str(obj.get("name") or "object")
            for instance_id, obj in iter_objects(json_data)}


def instance_by_object_id(json_data: Dict[str, object]) -> Dict[str, str]:
    """{object_id as text: instance_id}, for relations that cite the number.

    Resolves triple-shaped relations such as `[20, "grazing in", 19]`. Keys are
    strings so 20 and "20" both match.
    """
    mapping = {}
    for instance_id, obj in iter_objects(json_data):
        object_id = obj.get("object_id")
        if object_id is not None:
            mapping[str(object_id)] = instance_id
    return mapping


def query_counts(instances: Sequence[Dict[str, object]]) -> Dict[str, int]:
    """Query term -> how many instances it can stand for.

    Two cats give "cat" a count of 2; a lion of category cat adds one more to
    "cat". A detector keeps up to this many distinct boxes per term.
    """
    counts: Dict[str, int] = {}
    for instance in instances:
        for term in set(_instance_queries(instance).values()):
            if term:
                counts[term] = counts.get(term, 0) + 1
    return counts


def build_queries(instances: Sequence[Dict[str, object]]) -> List[str]:
    """Query terms to send to the detectors: varieties first, categories as backup.

    Deduplicated with a stable order.
    """
    terms: List[str] = []
    for key in (EXACT, PARENT_OF_GT):
        for instance in instances:
            term = _instance_queries(instance)[key]
            if term and term not in terms:
                terms.append(term)
    return terms
