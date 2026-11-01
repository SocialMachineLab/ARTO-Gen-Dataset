"""
Object Selection - two-layer vocabulary, conflict resolution and scene inference.

Layer 1 picks COCO categories; layer 2 gives each a style-specific variety from
STYLE_COCO_VARIETY. Objects are addressed by ``instance_id`` downstream.
A combination that fails compatibility or scene inference is resampled.
"""

import json
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from arto_kg.conceptualization.utils import setup_logger
from arto_kg.conceptualization.object_instance import (
    MAX_INSTANCES_PER_CATEGORY,
    PRIMARY,
    SECONDARY,
    ObjectInstance,
    build_instances,
    describe_instances,
    primary_instance,
    to_object_ids,
    to_object_names,
)
from arto_kg.config.model_config import (
    COCO_OBJECTS,
    COMPATIBILITY_INSTANCE_PROMPT,
    GENERATION_CONFIG,
    SCENE_INFERENCE_SYSTEM_PROMPT,
    STYLE_COCO_VARIETY,
)

# The prompt asks for 5-10 scene candidates, but neither bound is enforced:
# only zero candidates fails, and every candidate is kept for the random draw.
# Counts outside the range are logged.
TARGET_SCENE_CANDIDATES = 5
SUGGESTED_MAX_CANDIDATES = 10

# Retries for a single LLM call before the candidate combination is rejected.
LLM_MAX_RETRIES = 3


def _is_true(value: Any) -> bool:
    """A JSON boolean answer: true, or the string "true"; anything else is no."""
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return value is True


class ObjectSelectionError(RuntimeError):
    """Raised when no valid object combination could be produced."""


class ObjectSelector:

    def __init__(self, vllm_wrapper):
        self.logger = setup_logger("object_selector")
        self.vllm_wrapper = vllm_wrapper
        self.coco_objects = COCO_OBJECTS
        self.name_to_id = {name: coco_id for coco_id, name in COCO_OBJECTS.items()}
        self.style_objects = STYLE_COCO_VARIETY
        self.logger.info("Loaded style objects from STYLE_COCO_VARIETY config")

    # -- public API ------------------------------------------------------------

    def select_objects(self,
                       max_secondary_objects: Optional[int] = None,
                       style: str = "Baroque",
                       max_resample_attempts: int = 5) -> Dict[str, Any]:
        """Select one primary and several secondary object instances for a style.

        Each attempt draws a fresh combination; one with no plausible scene is
        discarded whole and redrawn. Raises ObjectSelectionError if every attempt
        fails, which callers treat as a rejected artwork.
        """
        if max_secondary_objects is None:
            max_secondary_objects = GENERATION_CONFIG["max_secondary_objects"]

        style_data = self._style_variety_table(style)
        self.logger.info(f"Selecting objects for style: {style}")

        last_error: Optional[Exception] = None
        for attempt in range(1, max_resample_attempts + 1):
            instances = self._sample_instances(style_data, max_secondary_objects)
            self.logger.info(
                f"Attempt {attempt}/{max_resample_attempts} initial selection: "
                f"{describe_instances(instances)}"
            )

            try:
                instances = self._resolve_conflicts(instances)
                candidates = self._infer_scenes(instances, style)
            except ObjectSelectionError as exc:
                last_error = exc
                self.logger.warning(f"Attempt {attempt} rejected: {exc}. Resampling.")
                continue

            selected_scene = random.choice(candidates).copy()
            selected_scene["scene_brief"] = self._generate_scene_brief(selected_scene, instances)
            self.logger.info(
                f"Selected scene from {len(candidates)} candidates: "
                f"{selected_scene.get('scene_type', 'unknown')}"
            )
            return self._build_result(instances, candidates, selected_scene)

        raise ObjectSelectionError(
            f"No valid object combination for style '{style}' after "
            f"{max_resample_attempts} resampling attempts: {last_error}"
        )

    # -- style pool ------------------------------------------------------------

    def _style_variety_table(self, style: str) -> Dict[str, List[str]]:
        """Return the two-layer table for a style, or raise."""
        style_data = self.style_objects.get(style)
        if not style_data:
            raise ObjectSelectionError(
                f"Style '{style}' has no STYLE_COCO_VARIETY table. "
                f"Available: {sorted(self.style_objects.keys())}"
            )

        usable = {
            category: variants
            for category, variants in style_data.items()
            if variants and category in self.name_to_id
        }
        if not usable:
            raise ObjectSelectionError(f"Style '{style}' has no usable COCO categories")
        return usable

    # -- sampling --------------------------------------------------------------

    def _sample_instances(self,
                          style_data: Dict[str, List[str]],
                          max_secondary_objects: int) -> List[ObjectInstance]:
        """Sample 1 primary + 1..max_secondary secondaries.

        A COCO category appears at most twice, and is repeated only when it has
        at least two distinct varieties.
        """
        categories = list(style_data.keys())

        main_category = random.choice(categories)
        main_variant = random.choice(style_data[main_category])
        records = [{
            "coco_id": self.name_to_id[main_category],
            "coco_category": main_category,
            "name": main_variant,
            "role": PRIMARY,
        }]

        counts: Counter = Counter({main_category: 1})
        used_variants: Dict[str, set] = {main_category: {main_variant}}

        target = random.randint(1, max(1, max_secondary_objects))
        max_draws = target * 20 + 50
        draws = 0
        while len(records) - 1 < target and draws < max_draws:
            draws += 1
            category = random.choice(categories)
            variants = style_data[category]
            taken = counts[category]

            if taken >= MAX_INSTANCES_PER_CATEGORY:
                continue
            # Repetition is only allowed when the category has >= 2 varieties.
            if taken >= 1 and len(variants) < 2:
                continue

            unused = [v for v in variants if v not in used_variants.get(category, set())]
            if not unused:
                continue

            variant = random.choice(unused)
            records.append({
                "coco_id": self.name_to_id[category],
                "coco_category": category,
                "name": variant,
                "role": SECONDARY,
            })
            counts[category] += 1
            used_variants.setdefault(category, set()).add(variant)

        return build_instances(records)

    # -- conflict resolution ---------------------------------------------------

    def _resolve_conflicts(self, instances: List[ObjectInstance]) -> List[ObjectInstance]:
        """Pairwise conflict filtering, resolved by instance id."""
        current = list(instances)

        for _ in range(3):
            if len(current) < 2:
                raise ObjectSelectionError("Conflict resolution left fewer than two objects")

            result = self._check_object_compatibility(current)
            if _is_true(result.get("compatible")):
                self.logger.info("Compatibility check passed")
                return current

            to_remove = self._decide_removal(current, result.get("conflicts", []))
            if not to_remove:
                # Declared incompatible but the conflicts are unusable; ask again.
                self.logger.warning("Incompatible set but no removable instance identified")
                continue

            removed = [i for i in current if i.instance_id in to_remove]
            current = [i for i in current if i.instance_id not in to_remove]
            self.logger.info(f"Removed due to conflict: {describe_instances(removed)}")

        raise ObjectSelectionError("Could not reach a compatible object set")

    def _check_object_compatibility(self, instances: List[ObjectInstance]) -> Dict[str, Any]:
        """Ask the LLM whether these instances can coexist.

        An unusable response after retries rejects the combination.
        """
        if len(instances) <= 1:
            return {"compatible": True}

        main = primary_instance(instances)
        payload = {
            "primary": self._instance_payload(main),
            "secondary": [self._instance_payload(i) for i in instances if i is not main],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                result = self.vllm_wrapper.generate_json_response(
                    COMPATIBILITY_INSTANCE_PROMPT, user_prompt
                )
            except Exception as exc:
                self.logger.warning(
                    f"Compatibility check error (attempt {attempt}/{LLM_MAX_RETRIES}): {exc}"
                )
                continue

            if isinstance(result, dict) and "error" not in result and "compatible" in result:
                return result

            self.logger.warning(
                f"Invalid compatibility response (attempt {attempt}/{LLM_MAX_RETRIES}): {result}"
            )

        raise ObjectSelectionError("Compatibility check failed after retries")

    @staticmethod
    def _instance_payload(instance: ObjectInstance) -> Dict[str, str]:
        return {
            "id": instance.instance_id,
            "name": instance.name,
            "category": instance.coco_category,
        }

    def _decide_removal(self,
                        instances: List[ObjectInstance],
                        conflicts: List[Dict[str, Any]]) -> set:
        """Choose which secondary instances to drop.

        Primary is protected.  Conflicts with the primary are resolved first by
        removing every secondary involved; otherwise the secondary appearing in
        the most conflicts is removed.
        """
        main = primary_instance(instances)
        valid_ids = {i.instance_id for i in instances}
        main_id = main.instance_id if main else None

        pairs: List[Tuple[str, str]] = []
        for conflict in conflicts or []:
            if not isinstance(conflict, dict):
                continue
            a, b = conflict.get("a"), conflict.get("b")
            if a in valid_ids and b in valid_ids and a != b:
                pairs.append((a, b))

        if not pairs:
            return set()

        # Step 1: conflicts against the primary object.
        against_main = {
            (b if a == main_id else a)
            for a, b in pairs
            if main_id in (a, b)
        }
        against_main.discard(main_id)
        if against_main:
            return against_main

        # Step 2: the secondary with the highest conflict count.
        counter: Counter = Counter()
        for a, b in pairs:
            counter[a] += 1
            counter[b] += 1
        counter.pop(main_id, None)
        if not counter:
            return set()

        highest = max(counter.values())
        candidates = [obj_id for obj_id, count in counter.items() if count == highest]
        return {random.choice(candidates)}

    # -- scene inference -------------------------------------------------------

    def _infer_scenes(self,
                      instances: List[ObjectInstance],
                      style: str) -> List[Dict[str, Any]]:
        """Request scene candidates for this combination.

        A combination with no plausible scene is rejected whole; the caller
        draws a new one instead of removing objects.
        """
        candidates = self._request_scene_candidates(instances, style)
        if candidates:
            return candidates

        raise ObjectSelectionError(
            f"No plausible scene for {describe_instances(instances)}"
        )

    def _request_scene_candidates(self,
                                  instances: List[ObjectInstance],
                                  style: str) -> Optional[List[Dict[str, Any]]]:
        """Scene-inference request, retried on invalid replies; returns the valid candidates or None."""
        payload = {
            "style": style,
            "objects": [
                dict(self._instance_payload(i), role=i.role) for i in instances
            ],
        }
        user_prompt = json.dumps(payload, ensure_ascii=False, indent=2)

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                response = self.vllm_wrapper.generate_json_response(
                    SCENE_INFERENCE_SYSTEM_PROMPT, user_prompt
                )
            except Exception as exc:
                self.logger.warning(
                    f"Scene inference error (attempt {attempt}/{LLM_MAX_RETRIES}): {exc}"
                )
                continue

            candidates = None
            if isinstance(response, dict):
                candidates = response.get("scene_candidates")

            if not isinstance(candidates, list):
                self.logger.warning(
                    f"Invalid scene inference response (attempt {attempt}/{LLM_MAX_RETRIES})"
                )
                continue

            candidates = [c for c in candidates if isinstance(c, dict) and c.get("scene_type")]

            # At least one candidate is required; fewer than five is accepted.
            if not candidates:
                self.logger.warning(
                    f"No usable scene candidates (attempt {attempt}/{LLM_MAX_RETRIES})"
                )
                continue

            if not (TARGET_SCENE_CANDIDATES <= len(candidates) <= SUGGESTED_MAX_CANDIDATES):
                self.logger.info(
                    f"{len(candidates)} scene candidates, outside the "
                    f"{TARGET_SCENE_CANDIDATES}-{SUGGESTED_MAX_CANDIDATES} asked "
                    "for - accepted as is"
                )

            return [self._normalise_candidate(c, instances) for c in candidates]

        return None

    def _normalise_candidate(self,
                             candidate: Dict[str, Any],
                             instances: List[ObjectInstance]) -> Dict[str, Any]:
        """Key object_interpretations by instance id and fill in missing entries."""
        raw = candidate.get("object_interpretations")
        raw = raw if isinstance(raw, dict) else {}

        by_name = {i.name.lower(): i for i in instances}
        normalised: Dict[str, Dict[str, Any]] = {}

        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            instance = None
            for i in instances:
                if i.instance_id == key:
                    instance = i
                    break
            if instance is None:
                instance = by_name.get(str(key).lower())
            if instance is None:
                continue
            normalised[instance.instance_id] = {
                "name": instance.name,
                "form": value.get("form", ""),
                "scale": value.get("scale", ""),
                "role": value.get("role", instance.role),
            }

        for instance in instances:
            normalised.setdefault(instance.instance_id, {
                "name": instance.name,
                "form": "",
                "scale": "",
                "role": instance.role,
            })

        candidate = dict(candidate)
        candidate["object_interpretations"] = normalised
        return candidate

    # -- result assembly -------------------------------------------------------

    def _build_result(self,
                      instances: List[ObjectInstance],
                      candidates: List[Dict[str, Any]],
                      selected_scene: Dict[str, Any]) -> Dict[str, Any]:
        main = primary_instance(instances)
        return {
            "instances": [i.to_dict() for i in instances],
            # Derived views over the instances.
            "object_ids": to_object_ids(instances),
            "object_names": to_object_names(instances),
            "primary_object": {
                "instance_id": main.instance_id,
                "id": main.coco_id,
                "name": main.coco_category,
                "specific_name": main.name,
            },
            "secondary_objects": [
                {
                    "instance_id": i.instance_id,
                    "id": i.coco_id,
                    "name": i.coco_category,
                    "specific_name": i.name,
                }
                for i in instances if not i.is_primary
            ],
            "scene_inference": {
                "all_candidates": candidates,
                "selected_scene": selected_scene,
            },
        }

    # -- scene brief -----------------------------------------------------------

    def _generate_scene_brief(self,
                              selected_scene: Dict[str, Any],
                              instances: List[ObjectInstance]) -> str:
        """Readable one-line scene description used by later stages."""
        scene_type = selected_scene.get("scene_type", "scene")
        interpretations = selected_scene.get("object_interpretations", {})

        if not instances:
            return f"A {scene_type}"

        main = primary_instance(instances)
        main_desc = self._describe_object_in_scene(
            main.name, interpretations.get(main.instance_id, {}), is_primary=True
        )

        secondary_descs = [
            self._describe_object_in_scene(
                i.name, interpretations.get(i.instance_id, {}), is_primary=False
            )
            for i in instances if i is not main
        ]

        if not secondary_descs:
            return f"A {scene_type} {main_desc}"

        if len(secondary_descs) == 1:
            secondary_part = secondary_descs[0]
        else:
            secondary_part = ", ".join(secondary_descs[:-1]) + ", and " + secondary_descs[-1]

        return f"A {scene_type} {main_desc}, with {secondary_part}"

    def _describe_object_in_scene(self,
                                  obj_name: str,
                                  interpretation: Dict[str, Any],
                                  is_primary: bool) -> str:
        form = str(interpretation.get("form", "")).lower()
        role = str(interpretation.get("role", "")).lower()

        # A toy is detected from the form only.
        if "toy" in form:
            obj_type = f"toy {obj_name}"
        elif "model" in form:
            obj_type = f"{obj_name} model"
        elif "decoration" in form or "decorative" in role:
            obj_type = f"decorative {obj_name}"
        else:
            obj_type = obj_name

        if is_primary:
            if "tool" in role:
                return f"where {obj_type} serves as the primary tool"
            if "functional" in role or "appliance" in role:
                return f"featuring {obj_type} as the main functional element"
            if "primary" in role or "main" in role:
                return f"centered around {obj_type}"
            return f"featuring {obj_type}"

        if "decoration" in form or "decorative" in role:
            return f"{obj_type} as decoration"
        if "shelf" in role or "display" in role:
            return f"{obj_type} displayed on shelves"
        if "background" in role:
            return f"{obj_type} in the background"
        if "toy" in form:
            return f"{obj_type} placed nearby"
        return obj_type
