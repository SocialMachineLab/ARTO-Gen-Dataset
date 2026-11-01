"""
Canonical object instance model.

Each artwork object is identified by its ``instance_id`` through every stage;
the COCO id or name is never an identity, since two objects may share a category.
``coco_category`` (layer 1) and ``name`` (layer 2) always travel together.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

PRIMARY = "primary"
SECONDARY = "secondary"

# Each COCO category may appear at most twice in one artwork.
MAX_INSTANCES_PER_CATEGORY = 2


@dataclass
class ObjectInstance:
    """A single object in an artwork, identified by ``instance_id``."""

    instance_id: str
    coco_id: int
    coco_category: str
    name: str
    role: str = SECONDARY

    # Filled in by ObjectEnhancer; None until that stage has run.
    size: Optional[str] = None
    state: Optional[str] = None
    # One or two colour-vocabulary names, most dominant first.
    main_colours: List[str] = field(default_factory=list)
    # AAT material labels; an object may have several.
    materials: List[str] = field(default_factory=list)
    # Whether the object is manufactured, so an empty `materials` is unambiguous:
    #   False + []       has no material - a living cow, a rose, bread
    #   True  + [names]  made, and the materials resolved
    #   True  + []       made, but nothing it named was in the vocabulary
    # Spec annotation only; never emitted to the graph.
    is_made_object: Optional[bool] = None
    artistic_description: Optional[str] = None

    @property
    def is_primary(self) -> bool:
        return self.role == PRIMARY

    @property
    def is_enhanced(self) -> bool:
        return self.artistic_description is not None

    def prompt_label(self) -> str:
        """How the instance is shown to an LLM: id plus the style-specific name."""
        return f"{self.instance_id} ({self.name})"

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "instance_id": self.instance_id,
            "coco_id": self.coco_id,
            "coco_category": self.coco_category,
            "name": self.name,
            "role": self.role,
        }
        if self.is_enhanced:
            data.update({
                "size": self.size,
                "state": self.state,
                "main_colours": list(self.main_colours),
                "materials": list(self.materials),
                "is_made_object": self.is_made_object,
                "artistic_description": self.artistic_description,
            })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ObjectInstance":
        return cls(
            instance_id=data["instance_id"],
            coco_id=int(data["coco_id"]),
            coco_category=data["coco_category"],
            name=data["name"],
            role=data.get("role", SECONDARY),
            size=data.get("size"),
            state=data.get("state"),
            # Accept both the `main_colours` and `primary_colors` keys.
            main_colours=list(data.get("main_colours",
                                       data.get("primary_colors", []))),
            materials=materials_of(data),
            is_made_object=made_object_of(data),
            artistic_description=data.get("artistic_description"),
        )


def make_instance_id(index: int) -> str:
    """Instance ids are positional and stable within one artwork: obj_0, obj_1, ..."""
    return f"obj_{index}"


def build_instances(selections: List[Dict[str, Any]]) -> List[ObjectInstance]:
    """Turn raw selection records into instances, assigning ids in order.

    Each record needs ``coco_id``, ``coco_category``, ``name`` and ``role``.
    """
    return [
        ObjectInstance(
            instance_id=make_instance_id(index),
            coco_id=int(record["coco_id"]),
            coco_category=record["coco_category"],
            name=record["name"],
            role=record.get("role", SECONDARY),
        )
        for index, record in enumerate(selections)
    ]


def instances_by_id(instances: List[ObjectInstance]) -> Dict[str, ObjectInstance]:
    return {instance.instance_id: instance for instance in instances}


def category_counts(instances: List[ObjectInstance]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for instance in instances:
        counts[instance.coco_category] = counts.get(instance.coco_category, 0) + 1
    return counts


def primary_instance(instances: List[ObjectInstance]) -> Optional[ObjectInstance]:
    for instance in instances:
        if instance.is_primary:
            return instance
    return instances[0] if instances else None


def describe_instances(instances: List[ObjectInstance]) -> str:
    """One-line summary for logs: 'obj_0 lamb[sheep], obj_1 iris[potted plant]'."""
    return ", ".join(
        f"{i.instance_id} {i.name}[{i.coco_category}]" for i in instances
    )


# --- Derived views -------------------------------------------------------------
# object_ids / object_names / enhanced_objects are derived views over the
# instance list, never a second source of truth.

def to_object_ids(instances: List[ObjectInstance]) -> List[int]:
    return [instance.coco_id for instance in instances]


def to_object_names(instances: List[ObjectInstance]) -> List[str]:
    return [instance.name for instance in instances]


def to_instance_index(instances: List[ObjectInstance]) -> List[Dict[str, Any]]:
    """Compact two-layer index per object: id, COCO id/category and variety name.

    Kept for human readers of a specification. Accepts instances or their dicts.
    """
    keys = ("instance_id", "coco_id", "coco_category", "name")
    index = []
    for instance in instances:
        source = instance if isinstance(instance, dict) else instance.to_dict()
        index.append({key: source.get(key) for key in keys})
    return index


def to_enhanced_objects(instances: List[ObjectInstance]) -> List[Dict[str, Any]]:
    """The ``enhanced_objects`` records, including instance_id and coco_category.

    `colours_of` and `materials_of` read colours and materials from these records.
    """
    enhanced = []
    for instance in instances:
        enhanced.append({
            "instance_id": instance.instance_id,
            "object_id": instance.coco_id,
            "coco_category": instance.coco_category,
            "name": instance.name,
            "role": instance.role,
            "artistic_description": instance.artistic_description or "",
            "size": instance.size or "Medium",
            "state": instance.state or "",
            "main_colours": list(instance.main_colours),
            "materials": list(instance.materials),
            "is_made_object": instance.is_made_object,
        })
    return enhanced


def colours_of(obj: Dict[str, Any]) -> List[str]:
    """The colour names on an enhanced-object dict.

    Reads `main_colours`, or `primary_colors` when that key is absent.
    """
    if not isinstance(obj, dict):
        return []
    raw = obj.get("main_colours")
    if raw is None:
        raw = obj.get("primary_colors")
    if isinstance(raw, str):
        raw = [raw]
    return [str(name).strip() for name in raw or [] if str(name).strip()]


def materials_of(obj: Dict[str, Any]) -> List[str]:
    """The material names on an enhanced-object dict.

    Reads `materials`, or a prose `material` string split on ",", ";" and "and".
    Names are not mapped onto the vocabulary here.
    """
    if not isinstance(obj, dict):
        return []

    raw = obj.get("materials")
    if raw is None:
        raw = obj.get("material")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]

    parts: List[str] = []
    for item in raw:
        text = str(item or "").strip()
        if not text:
            continue
        for chunk in text.replace(";", ",").replace(" and ", ",").split(","):
            chunk = chunk.strip()
            if chunk and chunk not in parts:
                parts.append(chunk)
    return parts


def made_object_of(obj: Dict[str, Any]) -> Optional[bool]:
    """Whether the record says this object is a manufactured thing.

    Without an explicit `is_made_object`, inferred from whether any material is named.
    """
    if not isinstance(obj, dict):
        return None
    declared = obj.get("is_made_object")
    if isinstance(declared, bool):
        return declared
    return bool(materials_of(obj))
