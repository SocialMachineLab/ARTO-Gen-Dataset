"""
Prompt Generator Module
Creates the final image generation prompts for an artwork.
"""

from typing import List, Dict, Any
from arto_kg.conceptualization.utils import setup_logger
from arto_kg.conceptualization.object_instance import colours_of, materials_of

PROMPT_MAX_RETRIES = 3


class PromptGenerationError(RuntimeError):
    """Raised when no usable image prompt could be written for an artwork."""

# The prompt format sent to the image model: the detailed description.
SELECTED_PROMPT_FORMAT = "complex"


class PromptGenerator:
    """Prompt Generator"""
    
    def __init__(self, vllm_wrapper):
        self.logger = setup_logger("prompt_generator")
        self.vllm_wrapper = vllm_wrapper

        # All prompt formats come from one LLM call; the detailed format is a
        # scene description.
        self.prompt_formats_system_prompt = """You are a prompt generator for an image generation model. Return ONLY valid JSON, no other text.

You are given an art style, the objects in the artwork with their visual attributes, their spatial and semantic relationships, and the environment. Write THREE prompt formats describing the SAME artwork.

Format:
{
    "simple": "one flowing sentence naming the main objects and the setting",
    "complex": "a detailed scene description of several sentences",
    "comma_separated": "keyword list separated by commas",
    "negative_prompt": "things to avoid in the generation"
}

REQUIREMENTS FOR "complex" - this is the prompt actually sent to the image model:
- Several sentences, substantially longer and richer than "simple"
- Describe the scene spatially: place objects in the foreground, middle ground and background, and state how they sit relative to each other
- Weave in each object's size, colours, material and state from the attributes given
- Include the lighting, time of day, weather and atmosphere
- End with a sentence on the style's visual treatment (brushwork, palette, mood)
- Write flowing prose. Do NOT produce a list of attributes.

RULES FOR ALL FORMATS:
- Refer to objects by their NAME. Never write an instance id such as obj_0.
- Describe only what is visible in the picture.

CRITICAL: Respond with ONLY the JSON object, no explanations or additional text."""

    def generate_simple_prompt(self, enhanced_objects: List[Dict[str, Any]], 
                             style: str, composition_data: Dict[str, Any] = None,
                             environment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate all prompt formats with one LLM call.

        Raises PromptGenerationError if no usable detailed prompt is returned.
        """
        self.logger.info("Generating prompt formats")

        user_prompt = self._build_prompt_request(
            enhanced_objects, style, composition_data, environment_data
        )

        # Fail closed: an artwork without a usable prompt is rejected.
        problem = None
        for attempt in range(1, PROMPT_MAX_RETRIES + 1):
            try:
                result = self.vllm_wrapper.generate_json_response(
                    self.prompt_formats_system_prompt,
                    user_prompt
                )
            except Exception as e:
                problem = f"error: {e}"
                self.logger.warning(f"Prompt generation (attempt {attempt}/{PROMPT_MAX_RETRIES}) {problem}")
                continue

            if "error" in result:
                problem = f"failed: {result.get('error')}"
                self.logger.warning(f"Prompt generation (attempt {attempt}/{PROMPT_MAX_RETRIES}) {problem}")
                continue

            formats = {
                key: result.get(key, "").strip()
                for key in ("simple", "complex", "comma_separated")
                if isinstance(result.get(key), str)
            }

            if not formats.get("complex"):
                problem = "returned no detailed prompt"
                self.logger.warning(f"Prompt generation (attempt {attempt}/{PROMPT_MAX_RETRIES}) {problem}")
                continue

            return {
                "formats": formats,
                "negative_prompt": result.get("negative_prompt", "blurry, low quality"),
                "objects_count": len(enhanced_objects),
                "style": style
            }

        raise PromptGenerationError(
            f"No usable prompt after {PROMPT_MAX_RETRIES} attempts; last: {problem}")

    def _build_prompt_request(self, enhanced_objects: List[Dict[str, Any]],
                              style: str,
                              composition_data: Dict[str, Any] = None,
                              environment_data: Dict[str, Any] = None) -> str:
        """Describe the artwork to the prompt-writing LLM.

        Relations are rendered by object name, never by instance id.
        """
        lines = [f"Art style: {style}", "", "Objects:"]

        names = {}
        for obj in enhanced_objects:
            name = obj.get('name', '')
            instance_id = obj.get('instance_id')
            if instance_id:
                names[instance_id] = name

            parts = [f"- {name}"]
            attrs = []
            if obj.get('size'):
                attrs.append(f"{obj['size']} size")
            if colours_of(obj):
                attrs.append(', '.join(colours_of(obj)))
            if materials_of(obj):
                attrs.append(', '.join(materials_of(obj)))
            if obj.get('state'):
                attrs.append(obj['state'])
            if attrs:
                parts.append(f" ({'; '.join(attrs)})")
            if obj.get('artistic_description'):
                parts.append(f"\n  appearance: {obj['artistic_description']}")
            lines.append(''.join(parts))

        def in_words(relations):
            worded = []
            for rel in relations or []:
                if not isinstance(rel, dict):
                    continue
                subject = names.get(rel.get('subject'), rel.get('subject'))
                obj_name = names.get(rel.get('object'), rel.get('object'))
                relation = str(rel.get('relation', '')).replace('_', ' ')
                if subject and obj_name and relation:
                    worded.append(f"- {subject} {relation} {obj_name}")
            return worded

        if composition_data:
            spatial = in_words(composition_data.get('spatial_relations'))
            semantic = in_words(composition_data.get('semantic_relations'))
            focus = names.get(composition_data.get('primary_focus'))

            if focus:
                lines += ["", f"Main focus: {focus}"]
            if spatial:
                lines += ["", "Spatial layout:"] + spatial
            if semantic:
                # Interactions are stated as a requirement so they are not
                # reduced to proximity. The rule sits in this header, not the
                # system prompt, so artworks without interactions never see it.
                lines += ["", "Interactions - each line below MUST appear in "
                          "\"complex\" as the subject performing that action on "
                          "the object; placing the two near each other does not "
                          "count:"] + semantic

        if environment_data:
            env = environment_data.get('environment_details', environment_data)
            env_lines = []
            for label, key in (("Setting", "scene_brief"), ("Place", "place365_name"),
                               ("Time of day", "time_of_day"), ("Weather", "weather"),
                               ("Period", "period"), ("Atmosphere", "atmosphere")):
                if env.get(key):
                    env_lines.append(f"- {label}: {env[key]}")
            lighting = env.get('lighting')
            if isinstance(lighting, dict):
                described = ' '.join(str(lighting.get(k, '')) for k in
                                     ('quality', 'type', 'direction') if lighting.get(k))
                if described:
                    env_lines.append(f"- Lighting: {described}")
            if env_lines:
                lines += ["", "Environment:"] + env_lines

        lines += ["", "Write the three prompt formats for this artwork."]
        return "\n".join(lines)

    def create_final_prompt_package(self, enhanced_objects: List[Dict[str, Any]],
                                  style: str, composition_data: Dict[str, Any] = None,
                                  environment_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create the final prompt package: main prompt, negative prompt and all formats."""
        self.logger.info("Creating final prompt package with multiple formats")

        prompt_data = self.generate_simple_prompt(
            enhanced_objects, style, composition_data, environment_data
        )

        # generate_simple_prompt either returns formats or raises
        formats = prompt_data["formats"]

        simple_data = {"negative_prompt": prompt_data.get("negative_prompt",
                                                          "blurry, low quality")}

        # main_prompt is the text sent to the image model (the detailed format).
        main_prompt = formats.get(SELECTED_PROMPT_FORMAT) or formats["simple"]

        return {
            "main_prompt": main_prompt,
            "negative_prompt": simple_data.get("negative_prompt", "blurry, low quality"),

            "prompt_formats": formats,

            # Metadata
            "objects_count": len(enhanced_objects),
            "style": style
        }
    
    
    
    
    
    
    





