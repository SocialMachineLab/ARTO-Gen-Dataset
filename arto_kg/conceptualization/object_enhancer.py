"""
Object Enhancer - environment-aware visual description of each object instance.

Addresses objects by ``instance_id`` and describes the style-specific name, not
the COCO category. Fills the four ARTO attributes (size, state, main colours,
materials); spatial relationships are out of scope.
"""

import json
from typing import Any, Dict, List, Optional, Union

from arto_kg.conceptualization.utils import setup_logger
from arto_kg.conceptualization.object_instance import (
    ObjectInstance, to_enhanced_objects, colours_of, materials_of)
from arto_kg.config import colour_vocabulary, material_vocabulary


# Retries for a single LLM call before enhancement is treated as failed.
LLM_MAX_RETRIES = 3

VALID_SIZES = ["Dominant", "Large", "Medium", "Small", "Marginal"]

# One dominant colour, optionally one secondary: more cannot be told apart from
# noise when validated against an object's bounding box.
MAX_OBJECT_COLOURS = 2

# A made object usually has one or two materials; three is the ceiling.
MAX_OBJECT_MATERIALS = 3


class ObjectEnhancementError(RuntimeError):
    """Raised when object enhancement could not produce usable attributes."""


class ObjectEnhancer:
    """Environment-aware visual enhancer for object instances."""

    def __init__(self, vllm_wrapper, output_manager: Optional = None):
        self.logger = setup_logger("object_enhancer")
        self.vllm_wrapper = vllm_wrapper
        self.output_manager = output_manager

        self.object_enhancement_system_prompt = """You are an expert object enhancer that creates environment-aware visual descriptions. Return ONLY valid JSON, no other text.

Your task is to describe each object's visual appearance considering the environment's lighting, atmosphere, and conditions.

Each object is given with a unique instance id, a style-specific name and its underlying COCO category. Describe the SPECIFIC NAME (for example describe a "lamb", not a generic "sheep"). You MUST return one entry per input object and echo its instance_id exactly.

Format:
{
    "enhanced_objects": [
        {
            "instance_id": "obj_0",
            "name": "the style-specific name you were given",
            "artistic_description": "clear artistic description of the object's appearance, considering environmental lighting and atmosphere",
            "size": "one of: Dominant, Large, Medium, Small, Marginal",
            "state": "current condition or state of the object (influenced by environment)",
            "main_colours": ["colour name", "colour name"],
            "is_made_object": true,
            "materials": ["material label", "material label"]
        }
    ]
}

IMPORTANT CONSIDERATIONS:
- Lighting affects color appearance: warm lighting (sunset, morning) adds golden/orange tones; cool lighting (overcast, night) adds blue/gray tones
- Weather affects object state: rainy weather may make surfaces wet/reflective; sunny weather creates strong shadows
- Time of day affects visibility and mood: morning light is fresh, evening light is warm, night emphasizes artificial lights
- Indoor vs outdoor affects how light interacts with materials
- Period/era influences the condition and aging of objects
- The primary object is the semantic centre of the artwork and should normally be scaled Dominant or Large relative to the secondary objects

Size options: Dominant, Large, Medium, Small, Marginal.

COLOURS: give one or two names for each object, copied EXACTLY from the colour list below. Do not invent names, do not write hex codes, and do not add adjectives - "warm golden brown" is not a valid answer, but "Cigar Brown" or "Gold" may be. Name the object's own colours as they appear under this lighting; the first name should be its most dominant colour.

IS_MADE_OBJECT: decide, for this artwork, whether the object is a manufactured thing.

Answer false for a living being, a real animal, a growing plant, or food - they are not made of anything, so also return "materials": []. Never answer with fur, flesh, hide, skin or feathers; those are not materials.

Answer true when the object is a made thing. Judge what THIS artwork depicts, not what the word usually means: a toy cow on a shelf, a marble statue of a cupid and a stuffed bird are all made objects, even though cow, cupid and bird normally are not. The scene's interpretation of each object is given above under "Object Forms" - use it.

MATERIALS: what the object is physically made of, copied EXACTLY from the material list below. Do not invent labels and do not add adjectives - a label with a describing word in front of it is not a valid answer, the plain label is. List each material separately when an object is made of several: ["material label", "material label"], never one string joined by "and". One or two is normal, three at most. Give at least one whenever is_made_object is true.

CRITICAL: Respond with ONLY the JSON object, no explanations or additional text."""

    # -- public API ------------------------------------------------------------

    def enhance_objects(self,
                        instances: List[ObjectInstance],
                        style: str,
                        environment: Optional[Dict[str, Any]] = None,
                        return_details: bool = False) -> Union[List[ObjectInstance], Dict[str, Any]]:
        """Fill the four ARTO attributes on each instance.

        Returns the same ObjectInstance objects, enhanced in place.  Raises
        ObjectEnhancementError if the LLM cannot describe every instance.
        """
        if not instances:
            self.logger.warning("No objects to enhance")
            return {"enhanced_objects": [], "instances": []} if return_details else []

        self._require_instances(instances)

        self.logger.info(
            "Enhancing visual descriptions for "
            f"{len(instances)} objects: {[i.name for i in instances]}"
        )

        user_prompt = self._build_enhancement_prompt(instances, style, environment)
        last_error = "unknown"

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                result = self.vllm_wrapper.generate_json_response(
                    self.object_enhancement_system_prompt, user_prompt
                )
            except Exception as exc:
                last_error = f"LLM error: {exc}"
                self.logger.warning(
                    f"Object enhancement error (attempt {attempt}/{LLM_MAX_RETRIES}): {exc}"
                )
                continue

            if not isinstance(result, dict) or "error" in result:
                # Record why generation ended, to tell truncation from a normal stop.
                if isinstance(result, dict):
                    last_error = (
                        f"LLM returned error: {result.get('error')} "
                        f"(finish_reason={result.get('finish_reason')!r}, "
                        f"stop_reason={result.get('stop_reason')!r}, "
                        f"tokens={result.get('generated_tokens')})"
                    )
                else:
                    last_error = f"LLM returned error: {result}"
                self.logger.warning(
                    f"Object enhancement rejected (attempt {attempt}/{LLM_MAX_RETRIES}): "
                    f"{last_error}"
                )
                continue

            enhanced_objects = result.get("enhanced_objects")
            if not isinstance(enhanced_objects, list) or not enhanced_objects:
                last_error = "no enhanced_objects in response"
                self.logger.warning(
                    f"No enhanced objects returned (attempt {attempt}/{LLM_MAX_RETRIES})"
                )
                continue

            try:
                self._apply_enhancements(enhanced_objects, instances)
            except ObjectEnhancementError as exc:
                last_error = str(exc)
                self.logger.warning(
                    f"Could not map enhancements (attempt {attempt}/{LLM_MAX_RETRIES}): {exc}"
                )
                continue

            self.logger.info(f"Successfully enhanced {len(instances)} objects")

            if return_details:
                return {
                    "instances": instances,
                    "enhanced_objects": to_enhanced_objects(instances),
                    "raw_output": result.get("raw_output", ""),
                    "parsing_details": {"status": "success", "objects_parsed": len(instances)},
                    "user_prompt": user_prompt,
                    "system_prompt": self.object_enhancement_system_prompt,
                }
            return instances

        raise ObjectEnhancementError(
            f"Object enhancement failed after {LLM_MAX_RETRIES} attempts ({last_error})"
        )

    @staticmethod
    def _require_instances(instances: Any) -> None:
        """Reject calls that pass COCO ids instead of ObjectInstance objects."""
        if instances and not isinstance(instances[0], ObjectInstance):
            raise TypeError(
                "enhance_objects() takes a list of ObjectInstance, not COCO ids. "
                "Passing ids would drop the style-specific object names."
            )

    # -- prompt ----------------------------------------------------------------

    @staticmethod
    def _colour_menu(environment: Optional[Dict[str, Any]]) -> str:
        """The colour names this artwork may use.

        Limited to the environment palette's colour families (the whole vocabulary
        if none are recorded), with the scene's own colours listed first.
        """
        scheme = {}
        if isinstance(environment, dict):
            scheme = (environment.get("colour_scheme")
                      or environment.get("color_scheme") or {})
        if not isinstance(scheme, dict):
            scheme = {}

        families = scheme.get("colour_families")
        names = (colour_vocabulary.names_in_groups(families)
                 if isinstance(families, list) and families
                 else colour_vocabulary.names_in_groups(colour_vocabulary.groups()))

        scene_colours = [name for name in (scheme.get("main_colours") or [])
                         if colour_vocabulary.is_known(name)]
        ordered = scene_colours + [n for n in names if n not in scene_colours]

        listing = "\n".join(f"  - {name}" for name in ordered)
        return f"COLOUR NAMES YOU MAY USE (copy exactly):\n{listing}"

    def _build_enhancement_prompt(self,
                                  instances: List[ObjectInstance],
                                  style: str,
                                  environment: Optional[Dict[str, Any]]) -> str:
        """Build the environment-aware enhancement prompt."""
        prompt_parts = []

        objects_payload = [
            {
                "instance_id": instance.instance_id,
                "name": instance.name,
                "category": instance.coco_category,
                "role": instance.role,
            }
            for instance in instances
        ]
        prompt_parts.append(
            "Objects to enhance:\n" + json.dumps(objects_payload, ensure_ascii=False, indent=2)
        )
        prompt_parts.append(f"Art Style: {style}")

        if environment:
            env_details = []

            if environment.get("scene_brief"):
                env_details.append(f"Scene: {environment['scene_brief']}")
            if environment.get("time_of_day"):
                env_details.append(f"Time of Day: {environment['time_of_day']}")
            if environment.get("weather"):
                env_details.append(f"Weather: {environment['weather']}")
            if environment.get("period"):
                env_details.append(f"Historical Period: {environment['period']}")
            if environment.get("is_indoor") is not None:
                location = "Indoor" if environment["is_indoor"] else "Outdoor"
                env_details.append(f"Location: {location}")

            lighting = environment.get("lighting")
            if isinstance(lighting, dict):
                lighting_desc = f"Lighting: {lighting.get('type', 'natural')} light"
                if lighting.get("quality"):
                    lighting_desc += f", {lighting['quality']} quality"
                if lighting.get("direction"):
                    lighting_desc += f", {lighting['direction']}"
                if lighting.get("mood"):
                    lighting_desc += f", {lighting['mood']} mood"
                env_details.append(lighting_desc)

            if environment.get("atmosphere"):
                env_details.append(f"Atmosphere: {environment['atmosphere']}")

            colour_scheme = environment.get("colour_scheme") or environment.get("color_scheme")
            if isinstance(colour_scheme, dict) and colour_scheme:
                env_details.append(
                    "Colour Scheme: " + json.dumps(colour_scheme, ensure_ascii=False)
                )

            # How the selected scene reinterprets each object (keyed by instance id).
            interpretations = environment.get("object_interpretations")
            if isinstance(interpretations, dict) and interpretations:
                hints = []
                for instance in instances:
                    hint = interpretations.get(instance.instance_id)
                    if isinstance(hint, dict):
                        form = hint.get("form") or "normal form"
                        scale = hint.get("scale") or ""
                        detail = f"{instance.instance_id} ({instance.name}): {form}"
                        if scale:
                            detail += f", {scale}"
                        hints.append(detail)
                if hints:
                    env_details.append("Object Forms: " + "; ".join(hints))

            if env_details:
                prompt_parts.append("ENVIRONMENT DETAILS:")
                prompt_parts.append("\n".join(f"- {detail}" for detail in env_details))

        # Colour menu limited to the scene palette's families.
        prompt_parts.append(self._colour_menu(environment))

        # The full material menu. Each label keeps its Getty qualifier because
        # some labels are also object variety names.
        prompt_parts.append(
            "MATERIAL LABELS YOU MAY USE (copy the label exactly, the "
            "parenthesis is only there to say which sense is meant):\n"
            + material_vocabulary.menu(shuffle=True))

        prompt_parts.append(f"""
Create rich, environment-aware visual descriptions for each object in {style} style.

Focus on:
- How the object appears under the specific lighting conditions
- Color variations caused by the environmental light (warm/cool tones)
- Surface appearance affected by weather (wet, dry, reflective, matte)
- State and condition appropriate to the historical period
- Material interactions with the environment (how light reflects, absorbs)
- Atmospheric effects on visibility and color saturation
- Style-specific artistic treatments that enhance the environmental mood

CRITICAL RULES:
- Return exactly one entry per input object, echoing its instance_id
- Describe ONLY visual and material qualities
- Do NOT mention spatial positioning or object relationships
- Consider how environment affects perceived colors and textures""")

        return "\n\n".join(prompt_parts)

    # -- response mapping ------------------------------------------------------

    def _apply_enhancements(self,
                            enhanced_objects: List[Dict[str, Any]],
                            instances: List[ObjectInstance]) -> None:
        """Map LLM entries onto instances by id, then by unique name.

        Never matches by position, which could attach attributes to the wrong object.
        """
        by_id: Dict[str, Dict[str, Any]] = {}
        by_name: Dict[str, List[Dict[str, Any]]] = {}

        for entry in enhanced_objects:
            if not isinstance(entry, dict):
                continue
            instance_id = entry.get("instance_id")
            if isinstance(instance_id, str):
                by_id.setdefault(instance_id, entry)
            name = entry.get("name")
            if isinstance(name, str):
                by_name.setdefault(name.strip().lower(), []).append(entry)

        name_counts: Dict[str, int] = {}
        for instance in instances:
            key = instance.name.strip().lower()
            name_counts[key] = name_counts.get(key, 0) + 1

        unmatched = []
        for instance in instances:
            entry = by_id.get(instance.instance_id)

            if entry is None:
                key = instance.name.strip().lower()
                matches = by_name.get(key, [])
                # Only trust name matching when the name is unambiguous on both sides.
                if len(matches) == 1 and name_counts.get(key, 0) == 1:
                    entry = matches[0]

            if entry is None:
                unmatched.append(instance.instance_id)
                continue

            self._assign(instance, entry)

        if unmatched:
            raise ObjectEnhancementError(
                f"No enhancement returned for instances: {unmatched}"
            )

    def _assign(self, instance: ObjectInstance, entry: Dict[str, Any]) -> None:
        """Copy attributes onto the instance. The name is never overwritten."""
        # Size feeds validation's size ordering, so an unknown label fails the attempt.
        size = str(entry.get("size") or "").strip()
        canonical_size = next((s for s in VALID_SIZES if s.lower() == size.lower()), None)
        if canonical_size is None:
            raise ObjectEnhancementError(
                f"{instance.instance_id} ({instance.name}): size {size or 'missing'!r} "
                f"is not one of {sorted(VALID_SIZES)}")
        instance.size = canonical_size

        # A missing state or description also fails the attempt; nothing is invented.
        state = entry.get("state")
        if not (isinstance(state, str) and state.strip()):
            raise ObjectEnhancementError(f"{instance.instance_id} ({instance.name}): no state")
        instance.state = state.strip()

        # Colours must be vocabulary names; unknown names are dropped because
        # they cannot be resolved to RGB later.
        raw = entry.get("main_colours", entry.get("primary_colors"))
        if isinstance(raw, str):
            raw = [raw]
        candidates = [str(c).strip() for c in raw or [] if str(c).strip()]

        accepted, rejected = [], []
        for name in candidates[:MAX_OBJECT_COLOURS]:
            canonical = colour_vocabulary.canonical(name)
            (accepted.append(canonical) if canonical else rejected.append(name))

        if rejected:
            self.logger.warning(
                f"{instance.instance_id}: dropped colours not in the vocabulary: {rejected}")

        # No vocabulary colour at all means the model ignored the list: retry.
        if not accepted:
            raise ObjectEnhancementError(
                f"{instance.instance_id} ({instance.name}): no colour from the "
                f"vocabulary; got {candidates or 'nothing'}"
            )

        instance.main_colours = accepted

        self._assign_materials(instance, entry)

        description = entry.get("artistic_description")
        if not (isinstance(description, str) and description.strip()):
            raise ObjectEnhancementError(
                f"{instance.instance_id} ({instance.name}): no artistic_description")
        instance.artistic_description = description.strip()

    def _assign_materials(self, instance: ObjectInstance, entry: Dict[str, Any]) -> None:
        """Resolve the declared materials against the AAT vocabulary.

        The model decides whether the object is made at all (`is_made_object`).
        Unresolvable labels leave `materials` empty but never fail the artwork.
        Resolved labels use Getty's spelling.
        """
        declared = materials_of(entry)
        resolved = material_vocabulary.resolve_all(declared)

        if resolved["rejected"]:
            self.logger.warning(
                f"{instance.instance_id}: materials not in the AAT vocabulary: "
                f"{resolved['rejected']}")

        made = entry.get("is_made_object")
        if not isinstance(made, bool):
            # No answer: infer it from whether anything usable was named.
            made = bool(resolved["accepted"])

        instance.is_made_object = made
        if not made:
            # The flag wins: a "not made" object keeps no materials.
            if resolved["accepted"]:
                self.logger.info(
                    f"{instance.instance_id} ({instance.name}): not a made object; "
                    f"dropped {resolved['accepted']}")
            instance.materials = []
            return

        instance.materials = resolved["accepted"][:MAX_OBJECT_MATERIALS]
        if not instance.materials:
            self.logger.info(
                f"{instance.instance_id} ({instance.name}): made object with no "
                f"resolvable material; left empty")

    # -- summaries over the enhanced_objects view -------------------------------

    def get_object_visual_summary(self, enhanced_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Visual summary over the enhanced_objects view."""
        if not enhanced_objects:
            return {}

        all_colors = []
        all_materials = []
        size_distribution = {"Dominant": 0, "Large": 0, "Medium": 0, "Small": 0, "Marginal": 0}

        for obj in enhanced_objects:
            all_colors.extend(colours_of(obj))

            all_materials.extend(materials_of(obj))

            size = obj.get("size", "Medium")
            if size in size_distribution:
                size_distribution[size] += 1

        return {
            "objects_count": len(enhanced_objects),
            "dominant_colors": list(dict.fromkeys(all_colors)),
            "material_variety": list(dict.fromkeys(all_materials)),
            "size_distribution": size_distribution,
            "visual_complexity": "high" if len(enhanced_objects) > 5 else "medium" if len(enhanced_objects) > 2 else "low"
        }

    def get_color_palette_summary(self, enhanced_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Overall colour palette summary."""
        if not enhanced_objects:
            return {}

        all_colors = []
        for obj in enhanced_objects:
            all_colors.extend(colours_of(obj))

        return {
            "overall_palette": list(dict.fromkeys(all_colors)),
            "color_diversity": len(set(all_colors))
        }
