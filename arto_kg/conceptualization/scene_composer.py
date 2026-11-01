"""
Scene Composer - spatial and semantic relationships between object instances.

Relations link ``instance_id`` values and use the RELATION_CONFIG vocabulary
(Visual Genome + GQA). If no usable composition is produced, the artwork fails.
"""

import json
from typing import Any, Dict, List, Optional

from arto_kg.conceptualization.utils import setup_logger
from arto_kg.conceptualization.object_instance import ObjectInstance, primary_instance
from arto_kg.config.model_config import OBJECT_CAPABILITIES, RELATION_CONFIG

LLM_MAX_RETRIES = 3


class SceneCompositionError(RuntimeError):
    """Raised when no usable composition could be produced."""


class SceneComposer:
    """Scene composer producing spatial and semantic relations."""

    def __init__(self, vllm_wrapper, output_manager: Optional = None):
        self.logger = setup_logger("scene_composer")
        self.vllm_wrapper = vllm_wrapper
        self.output_manager = output_manager

        # The strict menu, merged from Visual Genome and GQA.
        self.relation_rules = RELATION_CONFIG
        self.spatial_menu = list(self.relation_rules.get("spatial_relations", []))
        semantic = self.relation_rules.get("semantic_relations", {})
        self.semantic_actions = list(semantic.get("actions_and_interactions", []))
        self.semantic_states = list(semantic.get("pose_and_states", []))
        self.logger.info(
            f"Loaded relation vocabulary: {len(self.spatial_menu)} spatial, "
            f"{len(self.semantic_actions)} actions, {len(self.semantic_states)} states"
        )

        self.spatial_relationship_system_prompt = self._build_system_prompt()

    # -- system prompt ---------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build the system prompt with the relation vocabulary inlined.

        Uses concatenation, not str.format, because the prompt contains JSON braces.
        """
        return (
            "CRITICAL: Your entire response must be ONLY the JSON object. Start with { and end with }. "
            "Do not write any thinking, explanations, or text before or after the JSON.\n\n"
            "You are an expert scene composer creating artwork compositions with spatial and semantic "
            "relationships.\n\n"
            "Every object is given with a unique instance id. Refer to objects ONLY by their instance id "
            "(obj_0, obj_1, ...) in every relation. Two objects may share a category, so names are not "
            "unique and must not be used as identifiers.\n\n"
            "Your task is to design TWO types of relationships:\n\n"
            "1. SPATIAL RELATIONS (physical positioning - 2D bbox verifiable).\n"
            "   STRICT MENU - use ONLY these:\n   "
            + ", ".join(self.spatial_menu)
            + "\n\n2. SEMANTIC RELATIONS (meaningful interactions and actions).\n"
            "   STRICT MENU - use ONLY these:\n"
            "   - Actions: " + ", ".join(self.semantic_actions)
            + "\n   - States/Poses: " + ", ".join(self.semantic_states)
            + "\n\n"
            "DESIGN PRINCIPLES:\n\n"
            "For Spatial Relations:\n"
            "- Every object needs at least one spatial relation to be positioned\n"
            "- Use simple, clear positioning\n\n"
            "For Semantic Relations:\n"
            "- NOT all objects need semantic relationships; returning an empty list is acceptable\n"
            "- Only objects marked \"interactive\" in the input may be the subject of a semantic relation\n"
            "- The primary object is the focal point and should carry the richer interactions\n"
            "- Secondary and passive objects are positioned mainly by spatial relations\n"
            "- Prioritise relationships that create a visual narrative; avoid listing every possibility\n\n"
            "OUTPUT FORMAT (valid JSON only):\n"
            "{\n"
            "    \"primary_focus\": \"obj_0\",\n"
            "    \"spatial_relations\": [\n"
            "        {\"subject\": \"obj_1\", \"relation\": \"on\", \"object\": \"obj_3\"}\n"
            "    ],\n"
            "    \"semantic_relations\": [\n"
            "        {\"subject\": \"obj_0\", \"relation\": \"watching\", \"object\": \"obj_2\"}\n"
            "    ]\n"
            "}\n\n"
            "CRITICAL RULES:\n"
            "- Return ONLY the JSON object. Your response must start with { and end with }\n"
            "- subject and object must be instance ids that appear in the input\n"
            "- subject and object must be different\n"
            "- Every relation must come from the menus above\n\n"
            "REMINDER: Output ONLY the JSON. Nothing else."
        )

    # -- public API ------------------------------------------------------------

    def create_complete_composition(self,
                                    instances: List[ObjectInstance],
                                    style: str,
                                    environment_context: Optional[Dict[str, Any]] = None,
                                    artwork_id: str = None) -> Dict[str, Any]:
        """Design the composition for a set of enhanced instances."""
        return self.create_simple_composition(instances, style, environment_context, artwork_id)

    def create_simple_composition(self,
                                  instances: List[ObjectInstance],
                                  style: str,
                                  environment_context: Optional[Dict[str, Any]] = None,
                                  artwork_id: str = None) -> Dict[str, Any]:
        self.logger.info(f"Creating composition for {artwork_id or 'artwork'}")
        composition = self.design_spatial_relationships(
            instances, style, environment_context, artwork_id
        )
        self.logger.info(
            f"Composition created: focus={composition.get('primary_focus')} with "
            f"{len(composition.get('spatial_relations', []))} spatial + "
            f"{len(composition.get('semantic_relations', []))} semantic relations"
        )
        return composition

    def design_spatial_relationships(self,
                                     instances: List[ObjectInstance],
                                     style: str,
                                     environment_context: Optional[Dict[str, Any]] = None,
                                     artwork_id: str = None) -> Dict[str, Any]:
        """Ask the LLM for the composition and validate it against the vocabulary."""
        if not instances:
            raise SceneCompositionError("No object instances to compose")

        self._require_instances(instances)

        main = primary_instance(instances)
        user_prompt = self._build_user_prompt(instances, style, environment_context)
        valid_ids = {i.instance_id for i in instances}
        interactive_ids = {i.instance_id for i in instances if self._is_interactive(i)}
        last_error = "unknown"

        for attempt in range(1, LLM_MAX_RETRIES + 1):
            try:
                result = self.vllm_wrapper.generate_json_response(
                    self.spatial_relationship_system_prompt, user_prompt
                )
            except Exception as exc:
                last_error = f"LLM error: {exc}"
                self.logger.warning(
                    f"Composition error (attempt {attempt}/{LLM_MAX_RETRIES}): {exc}"
                )
                continue

            if not isinstance(result, dict) or "error" in result:
                last_error = f"LLM returned error: {result}"
                self.logger.warning(
                    f"Composition rejected (attempt {attempt}/{LLM_MAX_RETRIES})"
                )
                continue

            parsed = {k: v for k, v in result.items()
                      if k not in ("raw_output", "parsing_details", "error")}

            spatial = self._filter_relations(
                parsed.get("spatial_relations"), valid_ids, self.spatial_menu, kind="spatial"
            )
            if not spatial:
                last_error = "no valid spatial relations"
                self.logger.warning(
                    f"No usable spatial relations (attempt {attempt}/{LLM_MAX_RETRIES})"
                )
                continue

            # Actions require an interactive subject; poses and states apply to
            # inanimate objects too ("hanging on", "filled with", "resting on").
            semantic = self._filter_relations(
                parsed.get("semantic_relations"), valid_ids,
                self.semantic_actions + self.semantic_states, kind="semantic",
                interactive_ids=interactive_ids,
                action_terms={self._normalise(t) for t in self.semantic_actions},
            )

            focus = parsed.get("primary_focus")
            if focus not in valid_ids:
                focus = main.instance_id

            return {
                "primary_focus": focus,
                "spatial_relations": spatial,
                "semantic_relations": semantic,
            }

        raise SceneCompositionError(
            f"Scene composition failed after {LLM_MAX_RETRIES} attempts ({last_error})"
        )

    @staticmethod
    def _require_instances(instances: Any) -> None:
        if instances and not isinstance(instances[0], ObjectInstance):
            raise TypeError(
                "design_spatial_relationships() takes a list of ObjectInstance; "
                "relations are addressed by instance_id."
            )

    # -- prompt ----------------------------------------------------------------

    def _build_user_prompt(self,
                           instances: List[ObjectInstance],
                           style: str,
                           environment_context: Optional[Dict[str, Any]]) -> str:
        main = primary_instance(instances)

        objects_payload = [
            {
                "instance_id": i.instance_id,
                "name": i.name,
                "category": i.coco_category,
                "role": i.role,
                "size": i.size or "Medium",
                "interactive": self._is_interactive(i),
            }
            for i in instances
        ]

        parts = [
            "Objects in the scene:\n" + json.dumps(objects_payload, ensure_ascii=False, indent=2),
            f"Art Style: {style}",
        ]

        if environment_context:
            env_parts = []
            if environment_context.get("scene_brief"):
                env_parts.append(f"Scene: {environment_context['scene_brief']}")
            if environment_context.get("time_of_day"):
                env_parts.append(f"Time: {environment_context['time_of_day']}")
            if environment_context.get("atmosphere"):
                env_parts.append(f"Atmosphere: {environment_context['atmosphere']}")
            if env_parts:
                parts.append("Environment Context:\n" + "\n".join(env_parts))

        parts.append(
            f"Task: Design a {style} artwork composition with these objects.\n\n"
            f"Remember:\n"
            f"- {main.instance_id} ({main.name}) is the main focus\n"
            "- Create spatial relations to position all objects\n"
            "- Add semantic relations only where they create a visual story, and only "
            "with an interactive object as the subject\n"
            "- Not all objects need to interact with each other"
        )

        return "\n\n".join(parts)

    # -- validation ------------------------------------------------------------

    def _filter_relations(self,
                          relations: Any,
                          valid_ids: set,
                          vocabulary: List[str],
                          kind: str,
                          interactive_ids: Optional[set] = None,
                          action_terms: Optional[set] = None) -> List[Dict[str, str]]:
        """Keep only relations whose endpoints are known instances and whose
        predicate is in the vocabulary. Both spatial and semantic are checked."""
        if not isinstance(relations, list):
            return []

        normalised_vocab = {self._normalise(term): term for term in vocabulary}
        kept: List[Dict[str, str]] = []
        seen = set()

        for relation in relations:
            if not isinstance(relation, dict):
                continue

            subject = relation.get("subject")
            obj = relation.get("object")
            predicate = relation.get("relation")

            if subject not in valid_ids or obj not in valid_ids or subject == obj:
                self.logger.warning(
                    f"Dropped {kind} relation with unknown endpoints: {relation}"
                )
                continue

            if not isinstance(predicate, str):
                continue

            canonical = self._canonicalise_predicate(predicate, normalised_vocab)
            if canonical is None:
                self.logger.warning(f"Dropped {kind} relation outside vocabulary: {predicate}")
                continue

            # Only actions need an interactive subject; poses and states do not.
            if (action_terms is not None
                    and interactive_ids is not None
                    and self._normalise(canonical) in action_terms
                    and subject not in interactive_ids):
                self.logger.warning(
                    f"Dropped {kind} action whose subject is not interactive: {relation}"
                )
                continue

            key = (subject, canonical, obj)
            if key in seen:
                continue
            seen.add(key)
            kept.append({"subject": subject, "relation": canonical, "object": obj})

        return kept

    @staticmethod
    def _normalise(term: str) -> str:
        return term.strip().lower().replace("_", " ")

    def _canonicalise_predicate(self,
                                predicate: str,
                                normalised_vocab: Dict[str, str]) -> Optional[str]:
        """Map an LLM predicate onto the vocabulary, tolerating underscores."""
        key = self._normalise(predicate)

        # "operating" is not in the vocabulary; map it to "using".
        if "operating" in key:
            key = "using"

        return normalised_vocab.get(key)

    # -- object capabilities ---------------------------------------------------

    # COCO ids: 1 = person, 15-24 = bird, cat, dog, horse, sheep, cow, elephant,
    # bear, zebra, giraffe.  Used alongside OBJECT_CAPABILITIES because that table
    # does not list every animal category.
    PERSON_COCO_ID = 1
    ANIMAL_COCO_IDS = range(15, 25)

    def _is_interactive(self, instance: ObjectInstance) -> bool:
        """Whether the object (a person or animal) can carry semantic actions.

        Checked against the COCO category, which is the stable layer.
        """
        if instance.coco_id == self.PERSON_COCO_ID or instance.coco_id in self.ANIMAL_COCO_IDS:
            return True

        category = instance.coco_category.lower()
        for capability in ("can_see", "can_communicate", "can_use_tools", "can_hold"):
            if category in OBJECT_CAPABILITIES.get(capability, []):
                return True
        return False
