"""Extracts and builds the image-generation prompt from a specification."""

import re

from arto_kg.conceptualization.object_instance import colours_of, materials_of
from arto_kg.config.grounding_vocabulary import place_label
from typing import Dict, List, Any, Optional


class PromptProcessor:
    """Selects the specification's prompt, or assembles a fallback one."""
    
    def __init__(self):
        # Prompt priority weights
        self.prompt_priorities = {
            'main_prompt': 1.0,           # Highest priority
            'format_complex': 0.95,       # complex in prompt_formats
            'complex': 0.9,
            'format_simple': 0.85,        # simple in prompt_formats
            'detailed': 0.8,
            'simple': 0.7,
            'format_comma_separated': 0.65, # comma_separated in prompt_formats
            'comma_separated': 0.6,
            'format_tags': 0.55,          # tags in prompt_formats
            'tags': 0.5,
            'objects_based': 0.4,
            'composition_based': 0.3,
            'environment_based': 0.2
        }
        
        # Style keyword enhancement
        self.style_enhancers = {
            'photorealistic': ['highly detailed', 'sharp focus', 'professional photography'],
            'painting': ['brushstrokes', 'artistic', 'canvas texture'],
            'digital_art': ['digital painting', 'concept art', 'high resolution'],
            'sketch': ['pencil drawing', 'hand drawn', 'artistic sketch'],
            'watercolor': ['watercolor painting', 'fluid brushstrokes', 'soft edges'],
            'oil_painting': ['oil on canvas', 'rich colors', 'textured brushwork'],
            'renaissance': ['classical composition', 'chiaroscuro lighting', 'masterpiece'],
            'impressionist': ['loose brushstrokes', 'light and shadow', 'plein air'],
            'abstract': ['abstract art', 'non-representational', 'experimental'],
            'surreal': ['surrealistic', 'dreamlike', 'fantastical']
        }
    
    def extract_prompts(self, artwork_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract all available prompts from artwork data"""
        prompts = {}
        
        # Extract from final_prompts
        final_prompts = artwork_data.get('final_prompts', {})
        if isinstance(final_prompts, dict):
            for prompt_type, prompt_text in final_prompts.items():
                if prompt_text and isinstance(prompt_text, str):
                    prompts[prompt_type] = prompt_text.strip()
                elif prompt_type == 'prompt_formats' and isinstance(prompt_text, dict):
                    # Handle prompt_formats dictionary
                    for format_type, format_text in prompt_text.items():
                        if format_text and isinstance(format_text, str):
                            prompts[f'format_{format_type}'] = format_text.strip()
        
        # Build prompts from enhanced_objects
        enhanced_objects = artwork_data.get('objects', {}).get('enhanced_objects', [])
        if enhanced_objects:
            object_prompt = self._build_object_prompt(enhanced_objects)
            if object_prompt:
                prompts['objects_based'] = object_prompt
        
        # Extract from composition
        composition = artwork_data.get('composition', {})
        if composition:
            comp_prompt = self._build_composition_prompt(composition,
                                                         enhanced_objects)
            if comp_prompt:
                prompts['composition_based'] = comp_prompt
        
        # Extract from environment
        environment = artwork_data.get('environment', {})
        if environment:
            env_prompt = self._build_environment_prompt(environment)
            if env_prompt:
                prompts['environment_based'] = env_prompt
        
        return prompts
    
    def _build_object_prompt(self, enhanced_objects: List[Dict[str, Any]]) -> str:
        """Describe each enhanced object, preferring its artistic description."""
        if not enhanced_objects:
            return ""
        
        object_descriptions = []
        
        for obj in enhanced_objects:
            if not isinstance(obj, dict):
                continue
            
            obj_name = obj.get('name', '')
            artistic_desc = obj.get('artistic_description', '')
            size = obj.get('size', '')
            state = obj.get('state', '')
            colors = colours_of(obj)
            materials = materials_of(obj)
            
            # Build single object description
            obj_desc_parts = []
            
            # Size description
            if size and size.lower() != 'unknown':
                obj_desc_parts.append(f"{size.lower()}")
            
            # Color description
            if colors and isinstance(colors, list):
                # Process color data, can be string or dictionary
                color_strs = []
                for color in colors[:3]:  # Up to 3 colors
                    if isinstance(color, dict):
                        # If it's a dictionary, try to extract color name
                        color_name = color.get('name', color.get('color', str(color)))
                        color_strs.append(str(color_name))
                    else:
                        color_strs.append(str(color))
                if color_strs:
                    color_desc = " and ".join(color_strs)
                    obj_desc_parts.append(f"{color_desc}")
            
            # Material description
            if materials:
                obj_desc_parts.append(', '.join(materials))
            
            # Object name
            if obj_name:
                obj_desc_parts.append(f"{obj_name}")
            
            # State description
            if state:
                obj_desc_parts.append(f"{state}")
            
            # Artistic description (preferred)
            if artistic_desc:
                object_descriptions.append(artistic_desc)
            elif obj_desc_parts:
                object_descriptions.append(" ".join(obj_desc_parts))
        
        if object_descriptions:
            return ", ".join(object_descriptions)
        
        return ""
    
    def _build_composition_prompt(self, composition: Dict[str, Any],
                                  enhanced_objects: List[Dict[str, Any]] = None) -> str:
        """Render the composition's relations as text, by object name.

        Relations are `{subject, relation, object}` over instance ids; ids are
        replaced by names, and relations with an unnamed endpoint are skipped.
        """
        comp_parts = []

        names = {}
        for obj in enhanced_objects or []:
            instance_id = obj.get('instance_id')
            if instance_id and obj.get('name'):
                names[instance_id] = obj['name']

        def in_words(relations):
            worded = []
            for rel in relations or []:
                if not isinstance(rel, dict):
                    continue
                subject = names.get(rel.get('subject'))
                obj_name = names.get(rel.get('object'))
                relation = str(rel.get('relation', '')).replace('_', ' ')
                if subject and obj_name and relation:
                    worded.append(f"{subject} {relation} {obj_name}")
            return worded

        # Interactions first, since image models drop them most often.
        comp_parts += in_words(composition.get('semantic_relations'))
        comp_parts += in_words(composition.get('spatial_relations'))

        return ", ".join(comp_parts) if comp_parts else ""
    
    def _build_environment_prompt(self, environment: Dict[str, Any]) -> str:
        """Describe place, time of day, weather, lighting and atmosphere.

        Reads `environment.environment_details`, the block Stage 1 writes.
        """
        details = environment.get('environment_details') or {}
        if not isinstance(details, dict):
            return ""
        env_parts = []

        place = place_label(details.get('place365_name'))
        if place:
            article = "an" if place[0] in "aeiou" else "a"
            env_parts.append(f"set in {article} {place}")

        time_of_day = details.get('time_of_day')
        if time_of_day:
            env_parts.append(f"in the {time_of_day}")

        weather = details.get('weather')
        if weather:
            env_parts.append(f"{weather} weather")

        lighting = details.get('lighting')
        if isinstance(lighting, dict):
            words = [lighting.get(k) for k in ('quality', 'type')]
            words = [str(w) for w in words if w]
            if words:
                env_parts.append(f"{' '.join(words)} lighting")

        atmosphere = details.get('atmosphere')
        if atmosphere:
            env_parts.append(f"{atmosphere} atmosphere")

        return ", ".join(env_parts)
    
    def build_final_prompt(self, prompts: Dict[str, str], 
                          style_info: Dict[str, Any],
                          artwork_data: Dict[str, Any]) -> str:
        """Return main_prompt as-is, or assemble a fallback prompt."""

        main_prompt = prompts.get('main_prompt')
        if main_prompt:
            print(f"[INFO] Using main_prompt ({len(main_prompt)} chars)")
            return main_prompt

        # Fallback: pick the highest-priority prompt and enhance it.
        print("[WARNING] No main_prompt found, using fallback logic")
        best_prompt = self._select_best_prompt(prompts)
        
        if not best_prompt:
            return "A beautiful artwork"
        
        # Clean and optimize prompt
        cleaned_prompt = self._clean_prompt(best_prompt)
        
        # Add style enhancement
        enhanced_prompt = self._enhance_with_style(cleaned_prompt, style_info)
        
        # Add quality enhancement words
        final_prompt = self._add_quality_enhancers(enhanced_prompt, style_info)
        
        # Remove duplicate comma-separated phrases
        final_prompt = self._dedupe_phrases(final_prompt)

        return final_prompt

    def _dedupe_phrases(self, prompt: str) -> str:
        """Remove duplicate comma-separated phrases."""
        # Split on ", " and keep the first occurrence of each phrase
        words = prompt.split(', ')
        unique_words = []
        seen = set()
        
        for word in words:
            word_lower = word.lower().strip()
            if word_lower not in seen and word_lower:
                unique_words.append(word.strip())
                seen.add(word_lower)
        
        # Recombine
        final_prompt = ', '.join(unique_words)
        
        return final_prompt.strip()
    
    def _select_best_prompt(self, prompts: Dict[str, str]) -> str:
        """Return the highest-priority prompt."""
        if not prompts:
            return ""

        # Rank by priority weight.
        sorted_prompts = []
        for prompt_type, prompt_text in prompts.items():
            # Prompt priority weight
            priority = self.prompt_priorities.get(prompt_type, 0.1)
            sorted_prompts.append((priority, prompt_type, prompt_text))

        # Select the highest priority
        sorted_prompts.sort(key=lambda x: x[0], reverse=True)
        return sorted_prompts[0][2] if sorted_prompts else ""
    
    def _clean_prompt(self, prompt: str) -> str:
        """Strip hashtags and quotes and normalise spacing."""
        if not prompt:
            return ""

        # Remove hashtag tags
        prompt = re.sub(r'#\w+', '', prompt)

        # Remove quotes
        prompt = prompt.replace('"', '').replace("'", "")

        # Normalize commas and spaces: multiple spaces become single space, ensure space after comma
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = re.sub(r',\s*', ', ', prompt)
        prompt = re.sub(r'\s*,', ',', prompt)

        # Remove leading/trailing commas and spaces
        prompt = prompt.strip(', ')

        # Capitalize first letter
        prompt = prompt.capitalize() if prompt else ""

        return prompt
    
    def _enhance_with_style(self, prompt: str, style_info: Dict[str, Any]) -> str:
        """Append style keywords and the style name."""
        if not style_info:
            return prompt
        
        style_name = style_info.get('style_name', '').lower()
        style_enhancers = []
        
        # Match style enhancement words
        for style_key, enhancers in self.style_enhancers.items():
            if style_key in style_name:
                style_enhancers.extend(enhancers)
                break
        
        # Add style description
        if style_enhancers:
            enhanced_prompt = f"{prompt}, {', '.join(style_enhancers[:3])}"
        else:
            enhanced_prompt = prompt
        
        # Add specific style name
        if style_name and 'style' not in prompt.lower():
            enhanced_prompt += f", {style_name} style"
        
        return enhanced_prompt
    
    def _add_quality_enhancers(self, prompt: str, style_info: Dict[str, Any]) -> str:
        """Append style-appropriate quality words."""
        # Select appropriate quality words based on style
        style_name = style_info.get('style_name', '').lower()
        
        selected_enhancers = []
        
        if any(x in style_name for x in ['photorealistic', 'realistic', 'photo']):
            selected_enhancers = ['highly detailed', 'sharp focus', 'professional photography']
        elif any(x in style_name for x in ['painting', 'art', 'artistic']):
            selected_enhancers = ['masterpiece', 'beautiful composition', 'fine art']
        elif 'digital' in style_name:
            selected_enhancers = ['high resolution', 'digital art', 'concept art']
        else:
            selected_enhancers = ['best quality', 'detailed', 'beautiful']
        
        if selected_enhancers:
            return f"{prompt}, {', '.join(selected_enhancers[:2])}"
        
        return prompt
