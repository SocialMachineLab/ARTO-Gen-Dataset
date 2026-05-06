"""
Object Enhancer Module - Pure Object Visual Describer
Focuses on the visual characteristics, material, color, and artistic description of individual objects
Responsibilities: Visual description + Style adaptation + Symbolic meaning, does not involve spatial relationships
"""

import json
from typing import List, Dict, Any, Optional
from arto_kg.conceptualization.utils import setup_logger, parse_json_response
from arto_kg.config.model_config import COCO_OBJECTS

class ObjectEnhancer:
    """Professional Object Visual Enhancer"""
    
    def __init__(self, vllm_wrapper, output_manager: Optional = None):
        self.logger = setup_logger("object_enhancer")
        self.vllm_wrapper = vllm_wrapper
        self.coco_objects = COCO_OBJECTS
        self.output_manager = output_manager
        
        # Object visual enhancement system prompts - Environment-aware version
        self.object_enhancement_system_prompt = """You are an expert object enhancer that creates environment-aware visual descriptions. Return ONLY valid JSON, no other text.

Your task is to describe each object's visual appearance considering the environment's lighting, atmosphere, and conditions.

Format:
{
    "enhanced_objects": [
        {
            "name": "object_name",
            "artistic_description": "clear artistic description of the object's appearance, considering environmental lighting and atmosphere",
            "size": "one of: Dominant, Large, Medium, Small, Marginal",
            "state": "current condition or state of the object (influenced by environment)",
            "primary_colors": ["main color as it appears in this lighting", "secondary color"],
            "material": "physical material like wood, metal, plastic, fabric, glass, ceramic, etc. or empty if not applicable"
        }
    ]
}

IMPORTANT CONSIDERATIONS:
- Lighting affects color appearance: warm lighting (sunset, morning) adds golden/orange tones; cool lighting (overcast, night) adds blue/gray tones
- Weather affects object state: rainy weather may make surfaces wet/reflective; sunny weather creates strong shadows
- Time of day affects visibility and mood: morning light is fresh, evening light is warm, night emphasizes artificial lights
- Indoor vs outdoor affects how light interacts with materials
- Period/era influences the condition and aging of objects

Size options: Dominant, Large, Medium, Small, Marginal.

CRITICAL: Respond with ONLY the JSON object, no explanations or additional text."""

    def enhance_objects(self, object_ids: List[int], style: str, 
                       scene_context: Optional[Dict[str, Any]] = None,
                       return_details: bool = False) -> List[Dict[str, Any]]:
        """
        Enhance object visual descriptions

        Args:
            object_ids: List of object IDs
            style: Art style
            scene_context: Scene context (for better style adaptation)
            return_details: Whether to return detailed processing info

        Returns:
            List of enhanced object descriptions or detailed result dictionary
        """
        if not object_ids:
            self.logger.warning("No objects to enhance")
            return []
            
        object_names = [self.coco_objects[obj_id] for obj_id in object_ids]
        self.logger.info(f"Enhancing visual descriptions for {len(object_names)} objects: {object_names}")
        
        # Build user prompt
        user_prompt = self._build_enhancement_prompt(object_names, style, scene_context)
        
        try:

            
            # Use improved JSON parsing method
            result = self.vllm_wrapper.generate_json_response(
                self.object_enhancement_system_prompt,
                user_prompt
            )
            

            
            if "error" in result:

                return self._create_fallback_enhancement(object_ids, style)
            
            raw_content = result.get("raw_output", "")

            
            # Extract parsed result - generate_json_response already contains parsed data
            parsed_result = {k: v for k, v in result.items() 
                           if k not in ['raw_output', 'parsing_details', 'error']}

            
            if not parsed_result or "error" in parsed_result:

                return self._create_fallback_enhancement(object_ids, style)
            
            # Validate and process results
            enhanced_objects = parsed_result.get("enhanced_objects", [])
            if not enhanced_objects:
                self.logger.warning("No enhanced objects returned, using fallback")
                return self._create_fallback_enhancement(object_ids, style)
            
            # Ensure object count and matching
            processed_objects = self._process_enhancement_result(enhanced_objects, object_ids, object_names)
            
            self.logger.info(f"Successfully enhanced {len(processed_objects)} objects")
            
            # Return detailed info or simplified result as needed
            if return_details:
                return {
                    "enhanced_objects": processed_objects,
                    "raw_output": result.get("content", ""),
                    "parsing_details": {"status": "success", "objects_parsed": len(processed_objects)},
                    "user_prompt": user_prompt,
                    "system_prompt": self.object_enhancement_system_prompt,
                    "enhancement_notes": parsed_result.get("enhancement_notes", {})
                }
            else:
                return processed_objects
                
        except Exception as e:
            self.logger.error(f"Object enhancement error: {e}")
            return self._create_fallback_enhancement(object_ids, style)

    def _build_enhancement_prompt(self, object_names: List[str], style: str,
                                scene_context: Optional[Dict[str, Any]]) -> str:
        """Build environment-aware object enhancement prompt"""
        prompt_parts = []

        # Basic object info
        prompt_parts.append(f"Objects to enhance: {', '.join(object_names)}")
        prompt_parts.append(f"Art Style: {style}")

        # Environment context info (from full environment details of environment_designer)
        if scene_context:
            env_details = []

            # Scene description
            if "scene_brief" in scene_context:
                env_details.append(f"Scene: {scene_context['scene_brief']}")

            # Time and lighting
            if "time_of_day" in scene_context:
                env_details.append(f"Time of Day: {scene_context['time_of_day']}")

            # Weather (if any)
            if "weather" in scene_context and scene_context.get("weather"):
                env_details.append(f"Weather: {scene_context['weather']}")

            # Historical period
            if "period" in scene_context:
                env_details.append(f"Historical Period: {scene_context['period']}")

            # Indoor/Outdoor
            if "is_indoor" in scene_context:
                location = "Indoor" if scene_context["is_indoor"] else "Outdoor"
                env_details.append(f"Location: {location}")

            # Lighting info
            if "lighting" in scene_context:
                lighting = scene_context["lighting"]
                lighting_desc = f"Lighting: {lighting.get('type', 'natural')} light"
                if lighting.get('intensity'):
                    lighting_desc += f", {lighting['intensity']} intensity"
                if lighting.get('color_temperature'):
                    lighting_desc += f", {lighting['color_temperature']} temperature"
                env_details.append(lighting_desc)

            # Atmosphere
            if "atmosphere" in scene_context:
                env_details.append(f"Atmosphere: {scene_context['atmosphere']}")

            # Object form hints (from scene inference object_contexts)
            if "object_contexts" in scene_context:
                obj_contexts = scene_context["object_contexts"]
                context_hints = []
                for obj_name in object_names:
                    if obj_name in obj_contexts:
                        ctx = obj_contexts[obj_name]
                        hint = f"{obj_name}: {ctx.get('context_interpretation', 'normal form')}"
                        context_hints.append(hint)

                if context_hints:
                    env_details.append(f"Object Forms: {'; '.join(context_hints)}")

            if env_details:
                prompt_parts.append("ENVIRONMENT DETAILS:")
                prompt_parts.append("\n".join(f"- {detail}" for detail in env_details))

        # Task description
        task_description = f"""
Create rich, environment-aware visual descriptions for each object in {style} style.

Focus on:
- How the object appears under the specific lighting conditions
- Color variations caused by the environmental light (warm/cool tones)
- Surface appearance affected by weather (wet, dry, reflective, matte)
- State and condition appropriate to the historical period
- Material interactions with the environment (how light reflects, absorbs)
- Atmospheric effects on visibility and color saturation
- Style-specific artistic treatments that enhance the environmental mood
- Fine details that capture the time and place

CRITICAL RULES:
- Describe ONLY visual and material qualities
- Do NOT mention spatial positioning or object relationships
- Consider how environment affects perceived colors and textures
- Adapt object state to match environmental conditions"""

        prompt_parts.append(task_description)

        return "\n\n".join(prompt_parts)

    def _process_enhancement_result(self, enhanced_objects: List[Dict], 
                                  original_object_ids: List[int],
                                  original_names: List[str]) -> List[Dict[str, Any]]:
        """
        Process enhancement results, ensure matching with original objects
        """
        processed = []
        
        # Match enhancement info for each original object
        for i, (obj_id, obj_name) in enumerate(zip(original_object_ids, original_names)):
            # Try to find matching object in returned result
            enhanced_obj = None
            
            # Match by name
            for obj in enhanced_objects:
                if obj.get("name", "").lower() == obj_name.lower():
                    enhanced_obj = obj
                    break
            
            # If no match found, match by index (if possible)
            if not enhanced_obj and i < len(enhanced_objects):
                enhanced_obj = enhanced_objects[i]
                # Ensure correct name
                enhanced_obj["name"] = obj_name
            
            # If still no match, create fallback
            if not enhanced_obj:
                enhanced_obj = self._create_single_object_fallback(obj_id, obj_name, "Abstract")
            
            # Validate and standardize object structure
            standardized_obj = self._standardize_enhanced_object(enhanced_obj, obj_id, obj_name)
            processed.append(standardized_obj)
        
        return processed

    def _standardize_enhanced_object(self, enhanced_obj: Dict[str, Any], 
                                   obj_id: int, obj_name: str) -> Dict[str, Any]:
        """Standardize enhanced object data structure"""
        # Simplified data structure
        standardized = {
            "object_id": obj_id,
            "name": obj_name,
            "artistic_description": enhanced_obj.get("artistic_description", f"A beautifully rendered {obj_name}"),
            "size": enhanced_obj.get("size", "Medium"),  # Default to Medium size
            "state": enhanced_obj.get("state", "well-maintained"),
            "primary_colors": enhanced_obj.get("primary_colors", ["natural tones"]),
            "material": enhanced_obj.get("material", "")
        }
        
        return standardized

    def _create_fallback_enhancement(self, object_ids: List[int], style: str) -> List[Dict[str, Any]]:
        """Create fallback enhancement result"""
        self.logger.info("Creating fallback enhancement")
        
        fallback_objects = []
        for obj_id in object_ids:
            obj_name = self.coco_objects.get(obj_id, f"object_{obj_id}")
            fallback_obj = self._create_single_object_fallback(obj_id, obj_name, style)
            fallback_objects.append(fallback_obj)
        
        return fallback_objects

    def _create_single_object_fallback(self, obj_id: int, obj_name: str, style: str) -> Dict[str, Any]:
        """Create fallback enhancement for single object"""
        
        # Create description and default size based on object type
        object_defaults = {
            "person": {"desc": "A figure rendered with artistic sensitivity", "size": "Large", "colors": ["skin tone", "clothing color"]},
            "cat": {"desc": "An elegant feline with graceful curves", "size": "Medium", "colors": ["fur color", "eye color"]}, 
            "dog": {"desc": "A loyal companion with warm, intelligent eyes", "size": "Medium", "colors": ["fur color", "eye color"]},
            "car": {"desc": "A sleek vehicle with clean lines", "size": "Large", "colors": ["body color", "metallic accents"]},
            "book": {"desc": "A knowledge vessel with aged pages", "size": "Small", "colors": ["cover color", "page tone"]},
            "chair": {"desc": "A functional seat with balanced proportions", "size": "Medium", "colors": ["wood tone", "fabric color"]},
            "table": {"desc": "A stable surface with sturdy construction", "size": "Medium", "colors": ["wood tone", "surface color"]},
            "flower": {"desc": "A delicate bloom with vibrant petals", "size": "Small", "colors": ["petal color", "stem green"]},
            "tree": {"desc": "A majestic plant with reaching branches", "size": "Large", "colors": ["bark brown", "leaf green"]},
            "house": {"desc": "An architectural shelter with welcoming character", "size": "Dominant", "colors": ["wall color", "roof color"]},
            "microwave": {"desc": "A kitchen appliance with modern design", "size": "Medium", "colors": ["metallic silver", "control panel black"]}
        }
        
        # Real physical material based on object type
        object_materials = {
            "person": "",  # Person usually doesn't need material
            "cat": "",     # Animal usually doesn't need material  
            "dog": "",     # Animal usually doesn't need material
            "car": "metal",
            "book": "paper",
            "chair": "wood",
            "table": "wood", 
            "flower": "",   # Plant usually doesn't need material
            "tree": "",     # Plant usually doesn't need material
            "house": "brick",
            "microwave": "metal",
            "bottle": "glass",
            "cup": "ceramic",
            "bowl": "ceramic",
            "spoon": "metal",
            "knife": "metal",
            "fork": "metal",
            "plate": "ceramic",
            "laptop": "metal",
            "phone": "metal",
            "tv": "plastic",
            "couch": "fabric",
            "bed": "fabric",
            "pillow": "fabric",
            "blanket": "fabric",
            "backpack": "fabric",
            "handbag": "leather",
            "suitcase": "fabric",
            "bicycle": "metal",
            "motorcycle": "metal",
            "truck": "metal",
            "bus": "metal",
            "boat": "wood",
            "airplane": "metal"
        }
        
        defaults = object_defaults.get(obj_name, {
            "desc": f"A beautifully rendered {obj_name}",
            "size": "Medium", 
            "colors": ["natural tone", "shadow color"]
        })
        
        material = object_materials.get(obj_name, "")
        
        return {
            "object_id": obj_id,
            "name": obj_name,
            "artistic_description": f"{defaults['desc']}, interpreted in {style} style",
            "size": defaults["size"],
            "state": "well-maintained",
            "primary_colors": defaults["colors"],
            "material": material
        }

    def get_object_visual_summary(self, enhanced_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get object visual summary, for other modules' reference
        
        Args:
            enhanced_objects: List of enhanced objects
            
        Returns:
            Visual summary info
        """
        if not enhanced_objects:
            return {}
        
        # Extract key visual info (simplified)
        all_colors = []
        all_materials = []
        size_distribution = {"Dominant": 0, "Large": 0, "Medium": 0, "Small": 0, "Marginal": 0}
        
        for obj in enhanced_objects:
            # Collect color info
            colors = obj.get("primary_colors", [])
            all_colors.extend(colors)
            
            # Collect material info
            material = obj.get("material")
            if material:
                all_materials.append(material)
            
            # Count size distribution
            size = obj.get("size", "Medium")
            if size in size_distribution:
                size_distribution[size] += 1
        
        return {
            "objects_count": len(enhanced_objects),
            "dominant_colors": list(set(all_colors)),
            "material_variety": list(set(all_materials)),
            "size_distribution": size_distribution,
            "visual_complexity": "high" if len(enhanced_objects) > 5 else "medium" if len(enhanced_objects) > 2 else "low"
        }

    def get_color_palette_summary(self, enhanced_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get overall color palette summary (simplified)
        """
        if not enhanced_objects:
            return {}
        
        all_colors = []
        
        for obj in enhanced_objects:
            colors = obj.get("primary_colors", [])
            all_colors.extend(colors)
        
        return {
            "overall_palette": list(set(all_colors)),
            "color_diversity": len(set(all_colors))
        }

    def validate_enhancement_quality(self, enhanced_objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Simplified enhancement quality validation
        """
        if not enhanced_objects:
            return {"status": "failed", "reason": "no_objects"}
        
        # Simple validation: check if required fields exist
        required_fields = ["artistic_description", "size", "state", "primary_colors"]
        valid_objects = 0
        
        for obj in enhanced_objects:
            if all(field in obj and obj[field] for field in required_fields):
                valid_objects += 1
        
        success_rate = valid_objects / len(enhanced_objects)
        
        return {
            "status": "passed" if success_rate > 0.8 else "partial" if success_rate > 0.5 else "failed",
            "valid_objects": valid_objects,
            "total_objects": len(enhanced_objects),
            "success_rate": success_rate
        }