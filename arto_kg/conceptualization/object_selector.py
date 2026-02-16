import random
import json
import os
from typing import List, Dict, Any, Optional
from collections import Counter
from arto_kg.conceptualization.utils import setup_logger, parse_json_response
from arto_kg.config.model_config import (
    COCO_OBJECTS,
    COMPATIBILITY_SYSTEM_PROMPT,
    COMPATIBILITY_WITH_SCENE_INFERENCE_PROMPT,
    GENERATION_CONFIG,
    STYLE_COCO_VARIETY
)

class ObjectSelector:
    
    def __init__(self, vllm_wrapper):
        self.logger = setup_logger("object_selector")
        self.vllm_wrapper = vllm_wrapper
        self.coco_objects = COCO_OBJECTS
        self.max_iterations = 5 
        
        # Load Style-Specific Object Mappings from config
        self.style_objects = STYLE_COCO_VARIETY
        self.logger.info("Loaded style objects from STYLE_COCO_VARIETY config")
        
    def select_objects(self, max_secondary_objects: Optional[int] = None, style: str = "Baroque") -> Dict[str, Any]:
        """
        Select objects using Twin-Layer Strategy (Filter -> Refine) based on Art Style.
        """
        if max_secondary_objects is None:
            max_secondary_objects = GENERATION_CONFIG["max_secondary_objects"]

        self.logger.info(f"Selecting objects for style: {style}")

        # Reverse lookup for COCO IDs (Name -> ID)
        name_to_id = {v: k for k, v in self.coco_objects.items()}
        
        # Helper to get valid choices
        style_data = self.style_objects.get(style, {})
        if not style_data:
            self.logger.warning(f"Style '{style}' not found in configuration. Fallback to random COCO.")
            valid_coco_names = list(name_to_id.keys())
        else:
            valid_coco_names = list(style_data.keys())

        # 1. Select Main Object (Filter -> Refine)
        main_coco_name = random.choice(valid_coco_names)
        main_id = name_to_id.get(main_coco_name)
        
        # Refine: Pick specific variant
        if style_data and main_coco_name in style_data:
            main_specific_name = random.choice(style_data[main_coco_name])
        else:
            main_specific_name = main_coco_name

        self.logger.info(f"Main object selected: {main_coco_name} -> {main_specific_name}")

        # 2. Select Secondary Objects
        secondary_count = random.randint(1, max_secondary_objects) # Ensure at least 1 secondary for interest
        secondary_objects_data = [] # List of tuples (id, coco_name, specific_name)
        
        for _ in range(secondary_count):
            # Try to pick a different COCO category for diversity, but allow duplicates if needed
            # Weighted random? or simple random? Simple random is fine.
            sec_coco = random.choice(valid_coco_names)
            sec_id = name_to_id.get(sec_coco)
            
            # Refine
            if style_data and sec_coco in style_data:
                # Ensure unique variant if COCO category is same as main (e.g. 2 People -> Monk + Peasant)
                variants = style_data[sec_coco]
                # Filter out used variants for this category?
                # Simplify: Just pick random. 
                sec_specific = random.choice(variants)
            else:
                sec_specific = sec_coco
            
            secondary_objects_data.append({"id": sec_id, "name": sec_coco, "specific_name": sec_specific})

        # 3. Conflict Resolution (The Re-Roll Rule) implemented in resolve_conflicts
        # We pass the specific names for validation context if possible (logic update needed in resolve_conflicts)
        # For now, we pass IDs as system expects IDs.
        
        all_ids = [main_id] + [obj["id"] for obj in secondary_objects_data]
        
        # Store metadata for later use
        object_metadata = {
            main_id: {"specific_name": main_specific_name, "coco_name": main_coco_name},
        }
        for obj in secondary_objects_data:
             object_metadata[obj["id"]] = {"specific_name": obj["specific_name"], "coco_name": obj["name"]}

        self.logger.info(f"Initial selection: {[m['specific_name'] for m in object_metadata.values()]}")

        # Check compatibility
        final_ids, scene_inference = self._resolve_conflicts(main_id, all_ids, style)
        
        # Select a specific scene from candidates
        candidates = scene_inference.get("scene_candidates", [])
        selected_scene = {}
        if candidates:
            selected_scene = random.choice(candidates)
            self.logger.info(f"Selected scene from {len(candidates)} candidates: {selected_scene.get('scene_type', 'unknown')}")
        else:
             self.logger.warning("No scene candidates found in inference result.")
        
        # Re-construct result using specific names
        final_main_meta = object_metadata.get(main_id, {"specific_name": self.coco_objects[main_id], "coco_name": self.coco_objects[main_id]}) # Fallback safe
        
        # Build object_names list for backward compatibility
        object_names = [final_main_meta["specific_name"]] + [
            object_metadata[oid]["specific_name"] 
            for oid in final_ids if oid != main_id and oid in object_metadata
        ]
        
        return {
            "object_ids": final_ids,
            "object_names": object_names,  # Added for backward compatibility
            "primary_object": {
                "id": main_id,
                "name": final_main_meta["coco_name"],
                "specific_name": final_main_meta["specific_name"]
            },
            "secondary_objects": [
                {
                    "id": oid, 
                    "name": object_metadata[oid]["coco_name"], 
                    "specific_name": object_metadata[oid]["specific_name"]
                }
                for oid in final_ids if oid != main_id and oid in object_metadata
            ],
            "scene_inference": {
                "all_candidates": candidates,
                "selected_scene": selected_scene
            }
        }
    
    def _resolve_conflicts(self, main_object: int, all_objects: List[int], style: str = "Abstract", 
                          object_metadata: Optional[Dict[int, Dict[str, str]]] = None) -> tuple[List[int], Optional[Dict]]:
        """
        Two-layer conflict detection, now using Specific Variants if metadata is provided.
        """
        current_objects = all_objects.copy()

        # Layer 1: Basic Conflict Detection
        basic_conflict_iteration = 0
        while basic_conflict_iteration < 3:
            compatibility_result = self._check_object_compatibility(
                current_objects,
                infer_scenes=False,
                object_metadata=object_metadata
            )

            if compatibility_result.get("compatible", True):
                self.logger.info("Basic compatibility check passed")
                break

            self.logger.info("Found basic incompatibilities, resolving...")

            # extract conflicts
            main_conflicts = compatibility_result.get("main_object_conflicts", [])
            secondary_conflicts = compatibility_result.get("secondary_object_conflicts", [])

            to_remove = self._decide_removal_strategy(main_object, current_objects, main_conflicts, secondary_conflicts)

            if to_remove:
                for obj in to_remove:
                    if obj in current_objects:
                        current_objects.remove(obj)
                
                # Update log to show specific names if possible
                removed_names = []
                for obj in to_remove:
                    if object_metadata and obj in object_metadata:
                        removed_names.append(object_metadata[obj]["specific_name"])
                    else:
                        removed_names.append(self.coco_objects[obj])
                
                self.logger.info(f"Removed due to basic conflict: {removed_names}")
            else:
                break

            basic_conflict_iteration += 1

        # Layer 2: Scene Inference (Deep Check)
        scene_inference_result = self._infer_scene_possibilities_with_retry(
            main_object, current_objects, style, max_attempts=5
        )

        return current_objects, scene_inference_result

    def _infer_scene_possibilities_with_retry(self, main_object: int, 
                                             secondary_objects: List[int], 
                                             style: str = "Abstract", 
                                             max_attempts: int = 3) -> Dict[str, Any]:
        """
        Infer possible scenes for the given objects and style with retry logic.
        """
        # Resolve names for the prompt
        main_name = self.coco_objects[main_object]
        secondary_names = [self.coco_objects[obj] for obj in secondary_objects if obj != main_object]
        
        system_prompt = """You are an expert art director and scene designer.
Given a set of objects and an art style, suggest 3 plausible scene settings where these objects could naturally coexist.
You must return the result in valid JSON format only.

INPUT:
Style: [Art Style]
Main Object: [Object]
Secondary Objects: [List of Objects]

OUTPUT FORMAT:
{
  "scene_candidates": [
    {
      "scene_type": "Name of the scene (e.g., specific room, outdoor location)",
      "reasoning": "Why this scene is appropriate",
      "plausibility": 0.9,
      "object_interpretations": {
        "object_name": {"role": "primary/secondary", "form": "description of how it appears", "scale": "relative size"}
      }
    }
  ]
}
"""
        user_prompt = f"Style: {style}\nMain Object: {main_name}\nSecondary Objects: {json.dumps(secondary_names)}"

        for attempt in range(max_attempts):
            try:
                response = self.vllm_wrapper.generate_json_response(system_prompt, user_prompt)
                
                if response and "scene_candidates" in response and len(response["scene_candidates"]) > 0:
                     return response
                
                self.logger.warning(f"Attempt {attempt+1}/{max_attempts}: Invalid scene inference response")
                
            except Exception as e:
                self.logger.error(f"Scene inference error (Attempt {attempt+1}): {e}")

        # Fallback if all attempts fail
        self.logger.warning("All scene inference attempts failed. Using default.")
        return {
            "scene_candidates": [self._create_default_scene(main_object, secondary_objects)]
        }

    def _check_object_compatibility(self, objects: List[int], infer_scenes: bool = False,
                                  object_metadata: Optional[Dict[int, Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Check object compatibility, using Specific Names (e.g. "Wagon") if metadata is provided.
        """
        # Resolve names: Use Specific Name if available, else Generic COCO Name
        if object_metadata:
            object_names = []
            for obj in objects:
                if obj in object_metadata:
                    object_names.append(object_metadata[obj]["specific_name"])
                else:
                    object_names.append(self.coco_objects[obj])
        else:
            object_names = [self.coco_objects[obj] for obj in objects]

        if len(object_names) <= 1:
            return {"compatible": True}

        if infer_scenes:
            system_prompt = COMPATIBILITY_WITH_SCENE_INFERENCE_PROMPT
            self.logger.info("Using scene inference prompt for final compatibility check")
        else:
            system_prompt = COMPATIBILITY_SYSTEM_PROMPT

        # Prompt with Specific Names for better context
        user_prompt = f"Evaluate the compatibility of these specific historical objects in a single artwork:\nMain Object: {object_names[0]}\nSecondary Objects: [{', '.join(object_names[1:])}]"

        try:
            result = self.vllm_wrapper.generate_json_response(
                system_prompt,
                user_prompt
            )

            if "error" not in result and "compatible" in result:
                return result
            else:
                self.logger.warning(f"Compatibility check failed: {result}")
                return {"compatible": True}

        except Exception as e:
            self.logger.error(f"Compatibility check error: {e}")
            return {"compatible": True}  
    
    def _decide_removal_strategy(self, main_object: int, current_objects: List[int], 
                               main_conflicts: List[Dict], secondary_conflicts: List[Dict]) -> List[int]:
        
        to_remove = set()
        main_object_name = self.coco_objects[main_object]
        
        # process conflicts with main object
        if main_conflicts:
            for conflict in main_conflicts:
                if conflict["object1"] == main_object_name:
                    conflicting_name = conflict["object2"]
                elif conflict["object2"] == main_object_name:
                    conflicting_name = conflict["object1"]
                else:
                    continue
                
                for obj_id in current_objects:
                    if self.coco_objects[obj_id] == conflicting_name and obj_id != main_object:
                        to_remove.add(obj_id)

        # process conflicts with secondary objects
        if secondary_conflicts and not to_remove:
            conflict_counter = Counter()
            
            for conflict in secondary_conflicts:
                obj1_name = conflict["object1"]
                obj2_name = conflict["object2"]
                
                for obj_id in current_objects:
                    if self.coco_objects[obj_id] == obj1_name and obj_id != main_object:
                        conflict_counter[obj_id] += 1
                    elif self.coco_objects[obj_id] == obj2_name and obj_id != main_object:
                        conflict_counter[obj_id] += 1
            
            if conflict_counter:
                max_conflicts = max(conflict_counter.values())
                most_conflicted = [obj_id for obj_id, count in conflict_counter.items() 
                                 if count == max_conflicts]
                to_remove.add(random.choice(most_conflicted))
        
        return list(to_remove)
    
    def _generate_scene_brief(self, selected_scene: Dict[str, Any], object_names: List[str]) -> str:
        """
        Generate detailed scene description based on selected scene
        Format: Scene type + Main object role + Secondary object existence

        Args:
            selected_scene: Scene info containing scene_type, reasoning, object_interpretations
            object_names: List of object names

        Returns:
            Detailed scene description string, e.g.:
            "A craft workshop where scissors serve as the main cutting tool, with toy bus and snowboard displayed on shelves"
        """
        scene_type = selected_scene.get("scene_type", "scene")
        reasoning = selected_scene.get("reasoning", "")
        object_interpretations = selected_scene.get("object_interpretations", {})

        if not object_names:
            return f"A {scene_type}"

        # Separate main object and secondary objects
        main_obj_name = object_names[0]
        secondary_obj_names = object_names[1:] if len(object_names) > 1 else []

        # Main object description
        main_desc = self._describe_object_in_scene(
            main_obj_name,
            object_interpretations.get(main_obj_name, {}),
            is_primary=True
        )

        # Secondary object description
        secondary_descs = []
        for obj_name in secondary_obj_names:
            desc = self._describe_object_in_scene(
                obj_name,
                object_interpretations.get(obj_name, {}),
                is_primary=False
            )
            secondary_descs.append(desc)

        # Combine into natural sentence
        if not secondary_descs:
            # Only one object
            brief = f"A {scene_type} {main_desc}"
        else:
            # Multiple objects: describe main object role + secondary object existence
            if len(secondary_descs) == 1:
                secondary_part = secondary_descs[0]
            else:
                # Combine multiple secondary objects
                secondary_part = ", ".join(secondary_descs[:-1]) + ", and " + secondary_descs[-1]

            brief = f"A {scene_type} {main_desc}, with {secondary_part}"

        return brief

    def _describe_object_in_scene(self, obj_name: str, interpretation: Dict[str, Any], is_primary: bool) -> str:
        """
        Describe how a single object exists in the scene

        Args:
            obj_name: Object name
            interpretation: Object interpretation info (form, scale, role)
            is_primary: Whether it is the primary object

        Returns:
            Object description string
        """
        form = interpretation.get("form", "").lower()
        role = interpretation.get("role", "").lower()
        scale = interpretation.get("scale", "").lower()

        # Determine object form
        if "toy" in form or "miniature" in scale:
            obj_type = f"toy {obj_name}"
        elif "model" in form:
            obj_type = f"{obj_name} model"
        elif "decoration" in form or "decorative" in role:
            obj_type = f"decorative {obj_name}"
        elif "real" in form or "actual" in form:
            obj_type = obj_name
        else:
            obj_type = obj_name

        # Primary object: describe its role/function
        if is_primary:
            if "primary" in role or "main" in role:
                if "functional" in role or "appliance" in role:
                    return f"featuring {obj_type} as the main functional element"
                elif "tool" in role:
                    return f"where {obj_type} serves as the primary tool"
                else:
                    return f"centered around {obj_type}"
            else:
                return f"featuring {obj_type}"

        # Secondary object: describe its existence
        else:
            if "decoration" in form or "decorative" in role:
                return f"{obj_type} as decoration"
            elif "shelf" in role or "display" in role:
                return f"{obj_type} displayed on shelves"
            elif "background" in role:
                return f"{obj_type} in the background"
            elif "toy" in form:
                return f"{obj_type} placed nearby"
            else:
                return obj_type

    def _create_default_scene(self, main_object: int, all_objects: List[int]) -> Dict[str, Any]:
        """
        Create default scene (when LLM fails to return scene candidates)
        Infer reasonable scene based on main object category
        """
        main_name = self.coco_objects[main_object]

        # Simple object-scene mapping rules
        object_scene_map = {
            # Kitchen appliances
            "oven": "modern kitchen",
            "microwave": "kitchen counter",
            "refrigerator": "home kitchen",
            "toaster": "breakfast nook",
            "sink": "kitchen or bathroom",

            # Vehicles
            "car": "city street",
            "train": "train station",
            "airplane": "airport",
            "bicycle": "outdoor path",
            "bus": "bus stop",
            "truck": "highway",
            "boat": "harbor",
            "motorcycle": "road",

            # Furniture
            "bed": "bedroom",
            "chair": "living room",
            "couch": "living room",
            "dining table": "dining room",

            # Animals
            "cat": "home interior",
            "dog": "backyard",
            "horse": "rural landscape",
            "bird": "outdoor environment",
            "cow": "farm",
            "sheep": "pasture",

            # Electronics
            "tv": "living room",
            "laptop": "home office",
            "cell phone": "everyday setting",

            # Food
            "apple": "kitchen or outdoor",
            "banana": "kitchen",
            "pizza": "dining table",
            "cake": "celebration setting",
        }

        scene_type = object_scene_map.get(main_name, "art studio still life")

        return {
            "scene_type": scene_type,
            "plausibility": 0.7,
            "reasoning": f"Default scene based on primary object '{main_name}'",
            "fallback": True
        }

    def select_specific_objects(self, main_object_id: int,
                               secondary_object_ids: List[int],
                               style: str = "Abstract") -> Dict[str, Any]:
        """
        Select specified objects and infer scene

        Args:
            main_object_id: Main object ID
            secondary_object_ids: List of secondary object IDs
            style: Art style

        Returns:
            Same data structure as select_objects()
        """
        if main_object_id not in self.coco_objects:
            raise ValueError(f"Invalid main object ID: {main_object_id}")

        invalid_secondary = [obj_id for obj_id in secondary_object_ids
                           if obj_id not in self.coco_objects]
        if invalid_secondary:
            raise ValueError(f"Invalid secondary object IDs: {invalid_secondary}")

        all_objects = [main_object_id] + secondary_object_ids
        self.logger.info(f"Checking compatibility for specified objects: {[self.coco_objects[obj] for obj in all_objects]}")

        # Perform compatibility check and conflict resolution and scene inference (pass style)
        final_objects, scene_inference = self._resolve_conflicts(main_object_id, all_objects, style)

        # Randomly select one from scene candidates
        scene_candidates = scene_inference.get("scene_candidates", [])


        if scene_candidates:
            # Copy selected scene to avoid modifying original scene_candidates
            selected_scene = random.choice(scene_candidates).copy()

        else:
            selected_scene = self._create_default_scene(main_object_id, final_objects)


        # Generate more detailed scene description
        final_names = [self.coco_objects[obj] for obj in final_objects]

        selected_scene["scene_brief"] = self._generate_scene_brief(selected_scene, final_names)


        # Build return structure
        return {
            "object_ids": final_objects,
            "object_names": final_names,
            "primary_object": {
                "id": main_object_id,
                "name": self.coco_objects[main_object_id]
            },
            "secondary_objects": [
                {"id": obj_id, "name": self.coco_objects[obj_id]}
                for obj_id in final_objects[1:]
            ],
            "scene_inference": {
                "all_candidates": scene_candidates,
                "selected_scene": selected_scene
            }
        }
    
    def get_object_names(self, object_ids: List[int]) -> List[str]:
        
        return [self.coco_objects[obj_id] for obj_id in object_ids if obj_id in self.coco_objects]
    
    def get_object_categories(self, object_ids: List[int]) -> Dict[str, List[str]]:
        categories = {
            "people": [1],  # person
            "vehicles": [2, 3, 4, 5, 6, 7, 8, 9],  # bicycle, car, motorcycle, airplane, bus, train, truck, boat
            "animals": [15, 16, 17, 18, 19, 20, 21, 22, 23, 24],  # bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
            "furniture": [57, 58, 60, 61],  # chair, couch, bed, dining table
            "electronics": [63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73],  # tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator
            "kitchenware": [40, 41, 42, 43, 44, 45, 46],  # bottle, wine glass, cup, fork, knife, spoon, bowl
            "food": [47, 48, 49, 50, 51, 52, 53, 54, 55, 56],  # banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
            "accessories": [25, 26, 27, 28, 29],  # backpack, umbrella, handbag, tie, suitcase
            "sports": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],  # frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket
            "household": [59, 74, 75, 76, 77, 78, 79, 80],  # potted plant, book, clock, vase, scissors, teddy bear, hair drier, toothbrush
            "infrastructure": [10, 11, 12, 13, 14, 62]  # traffic light, fire hydrant, stop sign, parking meter, bench, toilet
        }
        
        result = {}
        object_names = self.get_object_names(object_ids)
        
        for category, category_ids in categories.items():
            category_objects = []
            for obj_id in object_ids:
                if obj_id in category_ids:
                    category_objects.append(self.coco_objects[obj_id])
            if category_objects:
                result[category] = category_objects
        
        return result
    
    def validate_object_ids(self, object_ids: List[int]) -> bool:
        
        return all(obj_id in self.coco_objects for obj_id in object_ids)
    
    def get_random_objects(self, count: int, exclude: Optional[List[int]] = None) -> List[int]:
        
        candidates = list(self.coco_objects.keys())
        
        if exclude:
            candidates = [obj_id for obj_id in candidates if obj_id not in exclude]
            
        if count >= len(candidates):
            return candidates
        
        return random.sample(candidates, count)
    
    

