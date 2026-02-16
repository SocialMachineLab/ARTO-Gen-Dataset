"""
Scene Composer Module - Professional Spatial Relationship Designer
Focuses on spatial relationships between objects, composition layout, and visual hierarchy design
Responsibilities: Spatial relations + Composition layout + Visual techniques, does not involve environment background
"""

import json
import os
from typing import List, Dict, Any, Optional
from arto_kg.conceptualization.utils import setup_logger, parse_json_response
from arto_kg.config.model_config import SPATIAL_RELATIONS, SEMANTIC_RELATIONS, RELATION_CONFIG

class SceneComposer:
    """Professional Scene Spatial COMPOSER - Includes Spatial and Semantic Relations"""

    def __init__(self, vllm_wrapper, output_manager: Optional = None):
        self.logger = setup_logger("scene_composer")
        self.vllm_wrapper = vllm_wrapper
        self.output_manager = output_manager

        # Load Relation Rules (The "Strict Menu")
        # Load Relation Rules from config
        self.relation_rules = RELATION_CONFIG
        self.logger.info("Loaded relation rules from RELATION_CONFIG config")

        # Extract menus
        spatial_menu = ", ".join(self.relation_rules.get("spatial_relations", []))
        semantic_actions = ", ".join(self.relation_rules.get("semantic_relations", {}).get("actions_and_interactions", []))
        semantic_states = ", ".join(self.relation_rules.get("semantic_relations", {}).get("pose_and_states", []))

        # New relationship system prompts
        self.spatial_relationship_system_prompt = """CRITICAL: Your entire response must be ONLY the JSON object. Start with { and end with }. Do not write any thinking, explanations, or text before or after the JSON.

You are an expert scene composer creating artwork compositions with spatial and semantic relationships.

Your task is to design TWO types of relationships:

1. SPATIAL RELATIONS (physical positioning - 2D bbox verifiable):
   Available relations: {spatial_menu}

2. SEMANTIC RELATIONS (meaningful interactions and actions):
   STRICT MENU - YOU MUST ONLY USE RELATIONS FROM THIS LIST:
   - Actions: {semantic_actions}
   - States/Poses: {semantic_states}

   CRITICAL CONSTRAINT: Do NOT use "parked" for vehicles. Use "standing" or "waiting" instead.

DESIGN PRINCIPLES (NO hard number constraints):

For Spatial Relations:
- Every object needs at least one spatial relation to be positioned
- Use simple, clear positioning
- Focus on establishing where objects are in the scene

For Semantic Relations:
- NOT all objects need semantic relationships
- Choose the most interesting and story-telling relationships
- Main object should be the focal point with more interactions
- Secondary objects may have minimal or no semantic relations
- Prioritize relationships that create a visual narrative
- Avoid listing all possible relationships

MAIN vs SECONDARY Objects:
- Main object: The primary focus, should have richer interactions
- Secondary objects: Supporting roles, mainly positioned spatially
- Interactive objects (person, animal) naturally have more semantic relations
- Passive objects (furniture, containers) mainly use spatial + state relations

OUTPUT FORMAT (valid JSON only):
{
    "primary_focus": "object_name",
    "spatial_relations": [
        {
            "subject": "object1",
            "relation": "left_of",
            "object": "object2"
        }
    ],
    "semantic_relations": [
        {
            "subject": "person",
            "relation": "talking_to",
            "object": "cat",
            "relation_type": "communicative"
        }
    ]
}

CRITICAL RULES:
- Return ONLY the JSON object. Your response must start with { and end with }
- No thinking, no explanations, no text before or after the JSON
- Create a focused, story-driven composition
- Quality over quantity for semantic relations

REMINDER: Output ONLY the JSON. Nothing else."""


    def design_spatial_relationships(self, enhanced_objects: List[Dict[str, Any]],
                                   style: str,
                                   environment_context: Optional[Dict[str, Any]] = None,
                                   artwork_id: str = None) -> Dict[str, Any]:
        """
        Design spatial and semantic relationships

        Args:
            enhanced_objects: Enhanced object descriptions
            style: Art style
            environment_context: Environment context (scene type, atmosphere, etc.)
            artwork_id: Artwork ID (used for saving detailed output)

        Returns:
            Design results containing spatial and semantic relationships
        """


        # Determine main object (the first object is usually the main object)
        main_object = enhanced_objects[0].get('name', 'unknown') if enhanced_objects else 'unknown'

        # Build user prompt
        objects_info = []
        for i, obj in enumerate(enhanced_objects):
            obj_name = obj.get('name', 'unknown')
            obj_size = obj.get('size', 'Medium')
            role = "Main Object" if i == 0 else "Secondary Object"
            obj_desc = f"- {obj_name} ({role}): size={obj_size}"
            objects_info.append(obj_desc)

        # Add environment context
        env_info = ""
        if environment_context:
            scene_brief = environment_context.get('scene_brief', 'abstract scene')
            time_of_day = environment_context.get('time_of_day', '')
            atmosphere = environment_context.get('atmosphere', '')

            env_parts = [f"Scene: {scene_brief}"]
            if time_of_day:
                env_parts.append(f"Time: {time_of_day}")
            if atmosphere:
                env_parts.append(f"Atmosphere: {atmosphere}")

            env_info = f"\n\nEnvironment Context:\n" + "\n".join(env_parts)

        user_prompt = f"""Objects in the scene:
{chr(10).join(objects_info)}

Art Style: {style}{env_info}

Task: Design a {style} artwork composition with these objects.

Remember:
- {main_object} is the main focus
- Create spatial relations to position all objects
- Add semantic relations where they create interesting visual stories
- Not all objects need to interact with each other
- Focus on meaningful relationships that enhance the narrative"""


        
        try:

            
            # Use improved JSON parsing method
            result = self.vllm_wrapper.generate_json_response(
                self.spatial_relationship_system_prompt,
                user_prompt
            )
            

            
            if "error" in result:

                fallback = self._create_fallback_relationships(enhanced_objects, main_object)

                return fallback

            raw_content = result.get("raw_output", "")


            # Extract parsed result - generate_json_response already contains parsed data
            parsed_result = {k: v for k, v in result.items()
                           if k not in ['raw_output', 'parsing_details', 'error']}


            if not parsed_result or "error" in parsed_result:

                fallback = self._create_fallback_relationships(enhanced_objects, main_object)

                return fallback
            
            # Validate required fields (new format)
            required_fields = ["primary_focus", "spatial_relations"]
            missing_fields = [field for field in required_fields if field not in parsed_result]
            if missing_fields:

                fallback = self._create_fallback_relationships(enhanced_objects, main_object)

                return fallback

            # semantic_relations is optional
            if "semantic_relations" not in parsed_result:
                parsed_result["semantic_relations"] = []

            
            # Application of Strict Menu Filtering & Correction
            parsed_result = self._filter_and_correct_relations(parsed_result)
            
            # parsed_result["llm_raw_output"] = raw_content
            # parsed_result["debug_info"] = {
            #     "vllm_called": True,
            #     "vllm_success": True,
            #     "json_parsed": True,
            #     "all_fields_present": True,
            #     "objects_count": len(enhanced_objects),
            #     "style": style
            # }
            

            return parsed_result
            
        except Exception as e:

            import traceback


            fallback = self._create_fallback_relationships(enhanced_objects, main_object)
            fallback["debug_info"] = {
                "vllm_called": False,
                "vllm_success": False,
                "error": str(e),
                "exception_type": str(type(e)),
                "objects_count": len(enhanced_objects),
                "style": style
            }

            return fallback

    def create_simple_composition(self, enhanced_objects: List[Dict[str, Any]],
                                style: str,
                                environment_context: Optional[Dict[str, Any]] = None,
                                artwork_id: str = None) -> Dict[str, Any]:
        """
        Create composition design (including spatial and semantic relationships)

        Args:
            enhanced_objects: Enhanced object descriptions
            style: Art style
            environment_context: Environment context
            artwork_id: Artwork ID

        Returns:
            Directly return relationship data {primary_focus, spatial_relations, semantic_relations, llm_raw_output, debug_info}
        """
        self.logger.info(f"Creating composition for {artwork_id or 'artwork'}")

        # Design spatial and semantic relationships, return directly
        relationship_data = self.design_spatial_relationships(
            enhanced_objects, style, environment_context, artwork_id
        )

        self.logger.info(f"Composition created successfully: {relationship_data.get('primary_focus', 'unknown')} with {len(relationship_data.get('spatial_relations', []))} spatial + {len(relationship_data.get('semantic_relations', []))} semantic relations")

        # Return relationship_data directly, no longer wrapped
        return relationship_data

    def create_complete_composition(self, enhanced_objects: List[Dict[str, Any]],
                                  style: str,
                                  environment_context: Optional[Dict[str, Any]] = None,
                                  artwork_id: str = None) -> Dict[str, Any]:
        """
        Create complete composition design

        Args:
            enhanced_objects: Enhanced object descriptions
            style: Art style
            environment_context: Environment context
            artwork_id: Artwork ID

        Returns:
            Composition design result
        """
        # Call composition method
        return self.create_simple_composition(enhanced_objects, style, environment_context, artwork_id)

    # =========================== Helper Methods ===========================

    def _create_fallback_relationships(self, enhanced_objects: List[Dict[str, Any]], main_object: str) -> Dict[str, Any]:
        """Create fallback relationship system"""
        objects = [obj.get('name', f'object_{i}') for i, obj in enumerate(enhanced_objects)]

        if not objects:
            return {
                "primary_focus": "unknown",
                "spatial_relations": [],
                "semantic_relations": [],
                "llm_raw_output": ""
            }

        spatial_relations = []
        semantic_relations = []

        # Simple spatial relationships: each object positioned relative to main object
        spatial_positions = ["left_of", "right_of", "above", "below"]
        for i, obj_name in enumerate(objects):
            if i == 0:  # Skip main object itself
                continue

            relation = spatial_positions[i % len(spatial_positions)]
            spatial_relations.append({
                "subject": obj_name,
                "relation": relation,
                "object": main_object
            })

        # If there is an interactive object, add a simple semantic relationship
        interactive_objects = ["person", "animal", "cat", "dog", "bird"]
        if main_object in interactive_objects:
            for obj_name in objects[1:]:  # Check secondary objects
                if obj_name in interactive_objects:
                    # Add a looking_at relationship
                    semantic_relations.append({
                        "subject": main_object,
                        "relation": "looking_at",
                        "object": obj_name,
                        "relation_type": "visual"
                    })
                    break  # Add only one

        return {
            "primary_focus": main_object,
            "spatial_relations": spatial_relations,
            "semantic_relations": semantic_relations,
            "llm_raw_output": ""
        }

    def _create_fallback_spatial_relationships(self, enhanced_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Backward compatible fallback method"""
        main_object = enhanced_objects[0].get('name', 'unknown') if enhanced_objects else 'unknown'
        return self._create_fallback_relationships(enhanced_objects, main_object)












# Keep original method name for backward compatibility
    def design_scene_framework(self, objects: List[str], style: str, artwork_id: str = None) -> Dict[str, Any]:
        """Backward compatible method - maps to first stage spatial relationship design"""
        enhanced_objects = [{"name": obj, "artistic_description": f"a {obj} in {style} style"} for obj in objects]
        return self.design_spatial_relationships(enhanced_objects, style, artwork_id)
    
    def _filter_and_correct_relations(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-processing: Verify relations against strict menu and auto-correct common errors.
        """
        semantic_rels = scene_data.get("semantic_relations", [])
        valid_actions = set(self.relation_rules.get("semantic_relations", {}).get("actions_and_interactions", []))
        valid_states = set(self.relation_rules.get("semantic_relations", {}).get("pose_and_states", []))
        
        filtered_semantics = []
        for rel in semantic_rels:
            r_val = rel.get("relation", "").lower().replace("_", " ")
            
            # Correction Rules
            if "parked" in r_val:
                rel["relation"] = "standing" # or waiting
                filtered_semantics.append(rel)
                continue
            
            if "operating" in r_val:
                rel["relation"] = "using"
                filtered_semantics.append(rel)
                continue
            
            # Strict Check
            is_valid = (r_val in valid_actions) or (r_val in valid_states) or (rel.get("relation") in valid_actions) or (rel.get("relation") in valid_states)
            
            if is_valid:
                filtered_semantics.append(rel)
            else:
                self.logger.warning(f"Filtered invalid semantic relation: {rel.get('relation')}")
        
        scene_data["semantic_relations"] = filtered_semantics
        return scene_data
    
    