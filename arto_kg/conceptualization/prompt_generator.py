"""
Prompt Generator Module
Responsible for creating final artistic expression and image generation prompts
Reuses and extends the final stage logic of the original code
"""

import json
from typing import List, Dict, Any, Optional
from arto_kg.conceptualization.utils import setup_logger


class PromptGenerator:
    """Prompt Generator"""
    
    def __init__(self, vllm_wrapper):
        self.logger = setup_logger("prompt_generator")
        self.vllm_wrapper = vllm_wrapper
        
        # Artistic expression system prompt (reuse and extend original code)
        self.artistic_expression_system_prompt = """
You are an art expression specialist. Based on all previous stages, determine the artistic expression elements.

Your task is to determine:
1. Color palette and tones
2. Emotional theme and mood
3. Symbolic relationships and meanings
4. Final artistic prompt for image generation

Respond ONLY with a valid JSON object in this format:
{
    "color_palette": {
        "primary_colors": ["color1", "color2", "color3"],
        "secondary_colors": ["color1", "color2"],
        "overall_tone": "warm/cool/neutral",
        "saturation": "high/medium/low",
        "brightness": "bright/medium/dark"
    },
    "emotional_theme": "main emotional expression",
    "symbolic_meanings": {
        "object_name": "symbolic meaning or role"
    },
    "artistic_techniques": "specific techniques relevant to the style",
    "final_prompt": "comprehensive prompt for image generation combining all elements"
}"""
        
        # Simplified prompt generation system prompt
        self.simple_prompt_system_prompt = """You are a prompt generator. Return ONLY valid JSON, no other text.

Format:
{
    "main_prompt": "comprehensive prompt for image generation",
    "negative_prompt": "things to avoid in the generation"
}

CRITICAL: Respond with ONLY the JSON object, no explanations or additional text."""
    
    def generate_artistic_expression(self, objects: List[str], style: str,
                                   framework_data: Dict[str, Any],
                                   layout_data: Dict[str, Any],
                                   environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate artistic expression
        Reuse stage 4 of original code: Artistic Expression
        
        Args:
            objects: List of object names
            style: Art style
            framework_data: Scene framework data
            layout_data: Spatial layout data
            environment_data: Environment data
            
        Returns:
            Artistic expression info
        """
        self.logger.info("Generating artistic expression")
        
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\nAll previous stages: {json.dumps({'stage1': framework_data, 'stage2': layout_data, 'stage3': environment_data})}\n\nDetermine the artistic expression elements and create a final image generation prompt."
        
        try:
            result = self.vllm_wrapper.generate_json_response(
                self.artistic_expression_system_prompt,
                user_prompt
            )
            
            if "error" in result:
                self.logger.warning(f"Artistic expression failed: {result}")
                return self._create_fallback_artistic_expression(objects, style, environment_data)
            
            # Validate required fields
            if "final_prompt" not in result:
                self.logger.warning("No final prompt generated, using fallback")
                return self._create_fallback_artistic_expression(objects, style, environment_data)
            
            self.logger.info(f"Generated artistic expression with final prompt")
            return result
            
        except Exception as e:
            self.logger.error(f"Artistic expression error: {e}")
            return self._create_fallback_artistic_expression(objects, style, environment_data)
    
    def generate_simple_prompt(self, enhanced_objects: List[Dict[str, Any]], 
                             style: str, composition_data: Dict[str, Any] = None,
                             environment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate simplified prompt
        
        Args:
            enhanced_objects: Enhanced object info
            style: Art style
            composition_data: Composition data
            environment_data: Environment data
            
        Returns:
            Simplified prompt info
        """
        self.logger.info("Generating simple prompt")
        
        # Build input info
        objects_info = []
        for obj in enhanced_objects:
            name = obj.get('name', '')
            size = obj.get('size', 'Medium')
            colors = ', '.join(obj.get('primary_colors', []))
            state = obj.get('state', '')
            
            obj_desc = f"{name} ({size} size"
            if colors:
                obj_desc += f", {colors}"
            if state:
                obj_desc += f", {state}"
            obj_desc += ")"
            objects_info.append(obj_desc)
        
        context_info = f"Objects: {'; '.join(objects_info)}\nStyle: {style}"

        # Add environment info (if any)
        if environment_data:
            env_details = environment_data.get('environment_details', {})
            scene_brief = env_details.get('scene_brief', '')
            if scene_brief:
                context_info += f"\nSetting: {scene_brief}"
        
        user_prompt = f"{context_info}\n\nCreate a clear image generation prompt."
        
        try:
            result = self.vllm_wrapper.generate_json_response(
                self.simple_prompt_system_prompt,
                user_prompt
            )
            
            if "error" in result:
                self.logger.warning(f"Simple prompt generation failed: {result}")
                return self._create_fallback_simple_prompt(enhanced_objects, style)
            
            # Result already contains parsed data
            if "main_prompt" not in result:
                self.logger.warning("Failed to parse simple prompt")
                return self._create_fallback_simple_prompt(enhanced_objects, style)
            
            return {
                "main_prompt": result.get("main_prompt", ""),
                "negative_prompt": result.get("negative_prompt", "blurry, low quality"),
                "objects_count": len(enhanced_objects),
                "style": style
            }
            
        except Exception as e:
            self.logger.error(f"Simple prompt generation error: {e}")
            return self._create_fallback_simple_prompt(enhanced_objects, style)

    def optimize_final_prompt(self, artistic_data: Dict[str, Any],
                            enhanced_objects: Optional[List[Dict[str, Any]]] = None,
                            composition_data: Optional[Dict[str, Any]] = None,
                            environment_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize final prompt
        New feature: Create more professional image generation prompts
        
        Args:
            artistic_data: Artistic expression data
            enhanced_objects: Enhanced object info
            composition_data: Composition data
            environment_data: Environment data
            
        Returns:
            Optimized prompt info
        """
        self.logger.info("Optimizing final prompt")
        
        # Build complete info context
        context = {
            "artistic_expression": artistic_data,
            "enhanced_objects": enhanced_objects or [],
            "composition": composition_data or {},
            "environment": environment_data or {}
        }
        
        user_prompt = f"Complete Artwork Design Information:\n{json.dumps(context, indent=2)}\n\nCreate an optimized, comprehensive prompt for high-quality image generation."
        
        try:
            result = self.vllm_wrapper.generate_json_response(
                self.prompt_optimization_system_prompt,
                user_prompt
            )
            
            if "error" in result:
                self.logger.warning(f"Prompt optimization failed: {result}")
                return self._create_fallback_optimized_prompt(artistic_data, enhanced_objects)
            
            if "optimized_prompt" not in result:
                self.logger.warning("No optimized prompt generated, using fallback")
                return self._create_fallback_optimized_prompt(artistic_data, enhanced_objects)
            
            self.logger.info("Successfully optimized prompt")
            return result
            
        except Exception as e:
            self.logger.error(f"Prompt optimization error: {e}")
            return self._create_fallback_optimized_prompt(artistic_data, enhanced_objects)
    
    def create_final_prompt_package(self, enhanced_objects: List[Dict[str, Any]],
                                  style: str, composition_data: Dict[str, Any] = None,
                                  environment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create complete final prompt package, including multiple formats
        
        Args:
            enhanced_objects: Enhanced object info
            style: Art style
            composition_data: Composition data
            environment_data: Environment data
            
        Returns:
            Complete prompt package with multiple formats
        """
        self.logger.info("Creating final prompt package with multiple formats")
        
        # Generate simplified prompt as a base
        simple_data = self.generate_simple_prompt(enhanced_objects, style, composition_data, environment_data)
        
        # Create complex format prompt
        complex_prompt = self._create_complex_prompt(enhanced_objects, style, composition_data, environment_data)
        
        # Create comma-separated prompt format
        comma_separated_prompt = self._create_comma_separated_prompt(enhanced_objects, style, environment_data)
        
        # Create tag format prompt
        tags_prompt = self._create_tags_prompt(enhanced_objects, style, environment_data)
        
        return {
            # Main prompt (simple format)
            "main_prompt": simple_data.get("main_prompt", ""),
            "negative_prompt": simple_data.get("negative_prompt", "blurry, low quality"),
            
            # Multiple format prompts
            "prompt_formats": {
                "simple": simple_data.get("main_prompt", ""),
                "complex": complex_prompt,
                "comma_separated": comma_separated_prompt,
                "tags": tags_prompt
            },
            
            # Metadata
            "objects_count": len(enhanced_objects),
            "style": style
        }
    
    def _create_fallback_simple_prompt(self, enhanced_objects: List[Dict[str, Any]], style: str) -> Dict[str, Any]:
        """Create fallback for simplified prompt"""
        if not enhanced_objects:
            return {
                "main_prompt": f"artwork in {style} style, high quality",
                "negative_prompt": "blurry, low quality",
                "objects_count": 0,
                "style": style
            }
        
        # Extract object info
        object_names = [obj.get('name', '') for obj in enhanced_objects]
        colors = []
        for obj in enhanced_objects:
            colors.extend(obj.get('primary_colors', []))
        
        # Create simple prompt
        objects_desc = ", ".join(object_names)
        colors_desc = ", ".join(list(set(colors))[:3])  # Max 3 colors
        
        main_prompt = f"{objects_desc} in {style} style"
        if colors_desc:
            main_prompt += f", {colors_desc} colors"
        main_prompt += ", high quality artwork"
        
        return {
            "main_prompt": main_prompt,
            "negative_prompt": "blurry, low quality, distorted",
            "objects_count": len(enhanced_objects),
            "style": style
        }
    
    def _create_simple_prompt(self, enhanced_objects: List[Dict[str, Any]], 
                            style: str, environment_data: Dict[str, Any]) -> str:
        """Create simple version prompt"""
        if not enhanced_objects:
            return f"artwork in {style} style"
        
        # Extract main objects
        main_objects = []
        for obj in enhanced_objects:
            name = obj.get("name", "")
            quantity = obj.get("quantity", 1)
            if quantity == 1:
                main_objects.append(name)
            else:
                main_objects.append(f"{quantity} {name}s")
        
        objects_desc = ", ".join(main_objects)

        # Basic environment
        env_details = environment_data.get('environment_details', {})
        environment = env_details.get("scene_brief", "artistic setting")
        lighting_info = env_details.get("lighting", {})
        lighting = lighting_info.get("type", "natural")
        
        return f"{objects_desc} in {environment}, {style} style, {lighting} lighting, high quality artwork"
    
    def _create_complex_prompt(self, enhanced_objects: List[Dict[str, Any]], 
                             style: str, composition_data: Dict[str, Any] = None,
                             environment_data: Dict[str, Any] = None) -> str:
        """Create complex detailed prompt"""
        if not enhanced_objects:
            return f"detailed artwork in {style} style, high quality, professional"
        
        prompt_parts = []
        
        # Style and quality intro
        prompt_parts.append(f"A highly detailed {style} style artwork featuring")
        
        # Detailed object descriptions
        detailed_objects = []
        for obj in enhanced_objects:
            name = obj.get('name', 'object')
            size = obj.get('size', 'medium')
            colors = obj.get('primary_colors', [])
            state = obj.get('state', '')
            materials = obj.get('materials', [])
            
            # Build object description
            obj_desc = f"a {size.lower()}-sized {name}"
            
            if colors:
                color_desc = ' and '.join(colors[:2])  # Limit to first 2 colors
                obj_desc += f" rendered in {color_desc}"
            
            if materials:
                material_desc = ', '.join(materials[:2])  # Limit to first 2 materials
                obj_desc += f" with {material_desc} texture"
            
            if state:
                obj_desc += f" in {state} condition"
            
            detailed_objects.append(obj_desc)
        
        prompt_parts.append(', '.join(detailed_objects))
        
        # Composition and relationship info
        if composition_data:
            # DIRECT ACCESS REFACOR: Access flat structure
            spatial_data = composition_data 
            # spatial_data = composition_data.get('spatial_relationships', {})

            # Add primary focus info
            primary_focus = spatial_data.get('primary_focus')
            if primary_focus:
                prompt_parts.append(f"with {primary_focus} as the focal point")

            # Add semantic relations (core interactions)
            semantic_relations = spatial_data.get('semantic_relations', [])
            if semantic_relations:
                interactions = []
                for rel in semantic_relations[:3]:  # Take only top 3 most important
                    subject = rel.get('subject')
                    relation = rel.get('relation', '').replace('_', ' ')
                    obj = rel.get('object')
                    if subject and relation and obj:
                        interactions.append(f"{subject} {relation} {obj}")

                if interactions:
                    prompt_parts.append(f"showing {', '.join(interactions)}")
        
        # Environment and setting
        if environment_data:
            env_details = environment_data.get('environment_details', {})

            # Scene and location
            scene_brief = env_details.get('scene_brief', '')
            if scene_brief:
                prompt_parts.append(f"set in {scene_brief}")

            # Time and weather
            time_of_day = env_details.get('time_of_day', '')
            weather = env_details.get('weather', '')
            if time_of_day:
                time_weather = time_of_day
                if weather:
                    time_weather += f", {weather} weather"
                prompt_parts.append(f"during {time_weather}")

            # Lighting
            lighting = env_details.get('lighting', {})
            if lighting:
                light_type = lighting.get('type', 'natural')
                light_intensity = lighting.get('intensity', '')
                light_temp = lighting.get('color_temperature', '')

                light_desc = f"{light_type} lighting"
                if light_intensity:
                    light_desc = f"{light_intensity} {light_desc}"
                if light_temp:
                    light_desc += f" with {light_temp} tones"

                prompt_parts.append(light_desc)
        
        # Technical specs
        prompt_parts.append("rendered with exceptional detail, professional quality, masterpiece level craftsmanship")
        
        return ', '.join(prompt_parts)
    
    def _create_comma_separated_prompt(self, enhanced_objects: List[Dict[str, Any]], 
                                     style: str, environment_data: Dict[str, Any] = None) -> str:
        """Create comma-separated tag-style prompt"""
        tags = []
        
        # Style tags
        tags.append(f"{style.lower()} style")
        
        # Object tags
        for obj in enhanced_objects:
            name = obj.get('name', 'object')
            tags.append(name.lower())
            
            # Add color tags
            colors = obj.get('primary_colors', [])
            for color in colors[:2]:  # Max 2 colors
                tags.append(f"{color.lower()}")
            
            # Add size tags
            size = obj.get('size', '')
            if size:
                tags.append(f"{size.lower()} size")
        
        # Environment tags
        if environment_data:
            env_details = environment_data.get('environment_details', {})

            # Scene tags
            scene_brief = env_details.get('scene_brief', '')
            if scene_brief:
                tags.append(scene_brief.lower().replace(' ', '_'))

            # Time tags
            time_of_day = env_details.get('time_of_day', '')
            if time_of_day:
                tags.append(time_of_day.lower())

            # Weather tags
            weather = env_details.get('weather', '')
            if weather:
                tags.append(weather.lower())

            # Lighting tags
            lighting = env_details.get('lighting', {})
            light_type = lighting.get('type', '')
            if light_type:
                tags.append(f"{light_type.lower()}_lighting")
        
        # Quality tags
        tags.extend(['high quality', 'detailed', 'professional artwork'])
        
        return ', '.join(tags)
    
    def _create_tags_prompt(self, enhanced_objects: List[Dict[str, Any]], 
                          style: str, environment_data: Dict[str, Any] = None) -> str:
        """Create tag format prompt (separated by #)"""
        tags = []
        
        # Main tags
        tags.append(f"#{style.replace(' ', '')}")
        tags.append("#artwork")
        
        # Object tags
        for obj in enhanced_objects:
            name = obj.get('name', 'object')
            tags.append(f"#{name.replace(' ', '')}")
            
            # Color tags
            colors = obj.get('primary_colors', [])
            for color in colors[:2]:
                tags.append(f"#{color.replace(' ', '')}")
        
        # Environment tags
        if environment_data:
            env_details = environment_data.get('environment_details', {})
            location = env_details.get('scene_brief', '')
            if location:
                clean_location = location.replace(' ', '').replace(',', '')
                tags.append(f"#{clean_location}")
        
        # Quality tags
        tags.extend(['#highquality', '#detailed', '#professional'])
        
        return ' '.join(tags)
    
    def _create_fallback_artistic_expression(self, objects: List[str], style: str, 
                                           environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback artistic expression"""
        # Default palette based on style
        style_palettes = {
            "Renaissance": {
                "primary_colors": ["gold", "deep blue", "rich red"],
                "overall_tone": "warm",
                "saturation": "medium"
            },
            "Impressionism": {
                "primary_colors": ["light blue", "yellow", "pink"],
                "overall_tone": "warm",
                "saturation": "high"
            },
            "Abstract": {
                "primary_colors": ["bold red", "bright blue", "yellow"],
                "overall_tone": "neutral",
                "saturation": "high"
            },
            "Chinese Ink Painting": {
                "primary_colors": ["black", "gray", "subtle blue"],
                "overall_tone": "cool",
                "saturation": "low"
            }
        }
        
        palette = style_palettes.get(style, {
            "primary_colors": ["blue", "brown", "white"],
            "overall_tone": "neutral",
            "saturation": "medium"
        })
        
        # Create simple final prompt
        objects_str = ", ".join(objects)
        env_details = environment_data.get('environment_details', {})
        environment = env_details.get("scene_brief", "artistic setting")
        
        final_prompt = f"{objects_str} in {environment}, {style} style, {palette['overall_tone']} color tone, high quality artwork"
        
        return {
            "color_palette": {
                **palette,
                "secondary_colors": ["white", "gray"],
                "brightness": "medium"
            },
            "emotional_theme": "balanced and harmonious",
            "symbolic_meanings": {obj: f"represents {obj} in artistic context" for obj in objects},
            "artistic_techniques": f"traditional {style} techniques",
            "final_prompt": final_prompt
        }
    
    def _create_fallback_optimized_prompt(self, artistic_data: Dict[str, Any],
                                        enhanced_objects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create fallback optimized prompt"""
        # Use final prompt from artistic data as base
        base_prompt = artistic_data.get("final_prompt", "artistic artwork")
        
        # If enhanced object info exists, improve prompt
        if enhanced_objects:
            object_descs = []
            for obj in enhanced_objects:
                name = obj.get("name", "")
                colors = obj.get("colors", [])
                state = obj.get("state", "")
                
                desc = name
                if colors:
                    desc += f" in {' and '.join(colors)}"
                if state:
                    desc += f" ({state})"
                
                object_descs.append(desc)
            
            base_prompt = f"{', '.join(object_descs)}, " + base_prompt
        
        return {
            "optimized_prompt": base_prompt + ", high quality, detailed, professional",
            "negative_prompt": "blurry, low quality, distorted, incomplete",
            "style_modifiers": ["artistic", "detailed", "high quality"],
            "quality_tags": ["high quality", "detailed", "professional"],
            "technical_parameters": {
                "aspect_ratio": "16:9",
                "composition_focus": "balanced composition",
                "detail_level": "high"
            },
            "prompt_breakdown": {
                "subject": "artwork subjects",
                "style": "artistic style",
                "composition": "balanced arrangement",
                "environment": "artistic setting",
                "colors": "harmonious colors",
                "quality": "high quality rendering"
            }
        }
    
    
    
    
    
    
    





