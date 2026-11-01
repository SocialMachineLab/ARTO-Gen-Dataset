


"""
Overall Alignment Evaluation Module
Evaluates the overall alignment between text descriptions and images
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from PIL import Image
import json

from arto_kg.conceptualization.object_instance import colours_of, materials_of

# The three alignment dimensions. Style is excluded because it is judged by
# the style gate.
ALIGNMENT_DIMENSIONS = ("content_coverage", "semantic_similarity", "overall_alignment")

# JSON schema the decoder is constrained to, so every reply parses.
ALIGNMENT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "content_coverage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "semantic_similarity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "overall_alignment": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "analysis": {"type": "string"},
    },
    "required": list(ALIGNMENT_DIMENSIONS) + ["analysis"],
    "additionalProperties": False,
}

# Room for three scores plus a prose analysis; a truncated reply would not parse.
ALIGNMENT_MAX_TOKENS = 1536

logger = logging.getLogger(__name__)

# OpenCLIP and SentenceTransformer are comparison methods only; the alignment
# score comes from the VLM. A missing package is listed in `degraded_components`.
try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    logger.warning("OpenCLIP not installed: the CLIP comparison will be reported "
                   "as degraded. It does not affect the alignment score.")
    OPENCLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed: the text-similarity "
                   "comparison will be reported as degraded. It does not affect "
                   "the alignment score.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def _objects_of(json_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Object records of a specification.

    Reads `objects.enhanced_objects`, or `objects` itself when it is a list.
    """
    objects = json_data.get('objects')
    if isinstance(objects, list):
        return [obj for obj in objects if isinstance(obj, dict)]
    if isinstance(objects, dict):
        enhanced = objects.get('enhanced_objects')
        if isinstance(enhanced, list):
            return [obj for obj in enhanced if isinstance(obj, dict)]
    return []


class AlignmentValidator:
    """Detailed text-image alignment validator"""
    
    def __init__(self, device: Optional[str] = None, vlm_wrapper=None):
        """Initialize alignment validator
        
        Args:
            device: Computation device ('cuda', 'cpu', or None for auto selection)
            vlm_wrapper: VLM wrapper for deep semantic alignment analysis
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = None
        self.clip_preprocess = None
        self.text_model = None
        self.vlm_wrapper = vlm_wrapper
        
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize models"""
        # Initialize OpenCLIP model
        if OPENCLIP_AVAILABLE:
            try:
                self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
                )
                self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
                logger.info(f"OpenCLIP model loaded on {self.device}")
            except Exception as e:
                logger.error(f"Failed to load OpenCLIP model: {e}")
                self.clip_model = None
        
        # Initialize text model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("SentenceTransformer model loaded")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer: {e}")
                self.text_model = None
    
    def extract_text_descriptions(self, json_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract various text descriptions from JSON data"""
        descriptions = {
            'overall_description': [],
            'object_descriptions': [],
            'style_descriptions': [],
            'color_descriptions': [],
            'spatial_descriptions': [],
            'state_descriptions': [],
            'environment_descriptions': []
        }
        
        # Overall description
        if 'description' in json_data:
            descriptions['overall_description'].append(json_data['description'])
        if 'prompt' in json_data:
            descriptions['overall_description'].append(json_data['prompt'])
        
        if 'final_prompts' in json_data:
            final_prompts = json_data['final_prompts']
            if isinstance(final_prompts, dict):
                # Main prompt
                if 'main_prompt' in final_prompts:
                    descriptions['overall_description'].append(final_prompts['main_prompt'])
                
                # Prompts in different formats
                if 'prompt_formats' in final_prompts:
                    prompt_formats = final_prompts['prompt_formats']
                    if isinstance(prompt_formats, dict):
                        for format_name, format_prompt in prompt_formats.items():
                            if format_prompt and format_name in ['simple', 'complex']:
                                descriptions['overall_description'].append(format_prompt)
        
        # Style description
        if 'style' in json_data:
            style = json_data['style']
            if isinstance(style, list):
                descriptions['style_descriptions'].extend([str(s) for s in style])
            else:
                descriptions['style_descriptions'].append(str(style))
        
        # `objects` is a dict (enhanced_objects, object_names) or a plain list,
        # read further below.
        objects_data = json_data.get('objects') or {}
        if not isinstance(objects_data, dict):
            objects_data = {}

        # Object descriptions from enhanced_objects
        enhanced_objects = objects_data.get('enhanced_objects', [])
        for obj in enhanced_objects:
            if isinstance(obj, dict):
                # Object name
                if 'name' in obj:
                    descriptions['object_descriptions'].append(obj['name'])
                
                if 'artistic_description' in obj:
                    descriptions['object_descriptions'].append(obj['artistic_description'])
                
                # General description field
                if 'description' in obj:
                    descriptions['object_descriptions'].append(obj['description'])
                
                # Declared colours (`main_colours` or `primary_colors`).
                descriptions['color_descriptions'].extend(colours_of(obj))

                # Optional plain `color` field
                if 'color' in obj:
                    color = obj['color']
                    if isinstance(color, list):
                        descriptions['color_descriptions'].extend([str(c) for c in color])
                    else:
                        descriptions['color_descriptions'].append(str(color))
                
                # State description
                if 'state' in obj:
                    descriptions['state_descriptions'].append(str(obj['state']))
                
                # Material description
                for material in materials_of(obj):
                    descriptions['object_descriptions'].append(f"material: {material}")
                
                # Size description
                if 'size' in obj:
                    descriptions['object_descriptions'].append(f"size: {obj['size']}")
        
        object_names = objects_data.get('object_names', [])
        if isinstance(object_names, list):
            descriptions['object_descriptions'].extend([str(name) for name in object_names])

        # `objects` given as a plain list
        objects = json_data.get('objects', [])
        if isinstance(objects, list):
            for obj in objects:
                if isinstance(obj, dict):
                    # Object name and description
                    if 'name' in obj:
                        descriptions['object_descriptions'].append(obj['name'])
                    if 'description' in obj:
                        descriptions['object_descriptions'].append(obj['description'])
                    
                    # Color description
                    if 'color' in obj:
                        color = obj['color']
                        if isinstance(color, list):
                            descriptions['color_descriptions'].extend([str(c) for c in color])
                        else:
                            descriptions['color_descriptions'].append(str(color))
                    
                    # State description
                    if 'state' in obj:
                        descriptions['state_descriptions'].append(str(obj['state']))
                    
                    # Spatial relation description
                    if 'spatial_relation' in obj:
                        descriptions['spatial_descriptions'].append(str(obj['spatial_relation']))
        
        # Environment description
        if 'environment' in json_data:
            env = json_data['environment']
            if isinstance(env, dict):
                # Environment details
                if 'environment_details' in env:
                    env_details = env['environment_details']
                    if isinstance(env_details, dict):
                        if 'specific_location' in env_details:
                            descriptions['environment_descriptions'].append(str(env_details['specific_location']))
                        if 'time_period' in env_details:
                            descriptions['environment_descriptions'].append(str(env_details['time_period']))
                        if 'lighting' in env_details:
                            lighting = env_details['lighting']
                            if isinstance(lighting, dict):
                                for key, value in lighting.items():
                                    descriptions['environment_descriptions'].append(f"lighting {key}: {value}")
                            else:
                                descriptions['environment_descriptions'].append(str(lighting))
                
                # Colour scheme, under environment_details or environment, in
                # either spelling.
                color_scheme = None
                for container in (env.get('environment_details') or {}, env):
                    if not isinstance(container, dict):
                        continue
                    for key in ('colour_scheme', 'color_scheme'):
                        candidate = container.get(key)
                        if isinstance(candidate, dict):
                            color_scheme = candidate
                            break
                    if color_scheme:
                        break

                if color_scheme:
                    if isinstance(color_scheme, dict):
                        # Flat shape with vocabulary names.
                        for colour in color_scheme.get('main_colours') or []:
                            if colour:
                                descriptions['color_descriptions'].append(str(colour))
                        if color_scheme.get('background_colour'):
                            descriptions['color_descriptions'].append(
                                str(color_scheme['background_colour']))
                        shift = color_scheme.get('lighting_shift')
                        if isinstance(shift, dict):
                            for key in ('light', 'shadow'):
                                if shift.get(key):
                                    descriptions['color_descriptions'].append(str(shift[key]))

                        # Nested main_palette shape.
                        if 'main_palette' in color_scheme:
                            main_palette = color_scheme['main_palette']
                            if isinstance(main_palette, dict):
                                for color_type in ['primary_colors', 'secondary_colors', 'accent_colors']:
                                    if color_type in main_palette:
                                        colors = main_palette[color_type]
                                        if isinstance(colors, list):
                                            descriptions['color_descriptions'].extend([str(c) for c in colors])

                        # Background color scheme
                        if 'background_scheme' in color_scheme:
                            bg_scheme = color_scheme['background_scheme']
                            if isinstance(bg_scheme, dict):
                                if 'base_color' in bg_scheme:
                                    descriptions['color_descriptions'].append(str(bg_scheme['base_color']))
                                if 'gradient_colors' in bg_scheme:
                                    gradient_colors = bg_scheme['gradient_colors']
                                    if isinstance(gradient_colors, list):
                                        descriptions['color_descriptions'].extend([str(c) for c in gradient_colors])
                        
                        # Lighting colors
                        if 'lighting_colors' in color_scheme:
                            lighting_colors = color_scheme['lighting_colors']
                            if isinstance(lighting_colors, dict):
                                for color_name, color_value in lighting_colors.items():
                                    if color_value:
                                        descriptions['color_descriptions'].append(str(color_value))
                
                # Flat environment fields
                if 'background' in env:
                    descriptions['environment_descriptions'].append(str(env['background']))
                if 'setting' in env:
                    descriptions['environment_descriptions'].append(str(env['setting']))
                if 'lighting' in env and 'environment_details' not in env:
                    descriptions['environment_descriptions'].append(str(env['lighting']))
                if 'color' in env:
                    color = env['color']
                    if isinstance(color, list):
                        descriptions['color_descriptions'].extend([str(c) for c in color])
                    else:
                        descriptions['color_descriptions'].append(str(color))
            else:
                descriptions['environment_descriptions'].append(str(env))
        
        return descriptions
    
    def calculate_clip_alignment(self, image: np.ndarray, text_descriptions: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate image-text alignment using CLIP"""
        if not self.clip_model:
            return {'error': 'CLIP model not available'}
        
        try:
            # Convert image format
            if image.shape[-1] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            alignment_scores = {}
            detailed_scores = {}
            
            # Calculate alignment for each description type
            for desc_type, descriptions in text_descriptions.items():
                if not descriptions:
                    continue
                
                type_scores = []
                type_details = []
                
                for desc in descriptions:
                    if desc.strip():
                        # Prepare text
                        text_input = self.clip_tokenizer([desc]).to(self.device)
                        
                        # Calculate features and similarity
                        with torch.no_grad():
                            image_features = self.clip_model.encode_image(image_input)
                            text_features = self.clip_model.encode_text(text_input)
                            
                            # Calculate cosine similarity
                            similarity = torch.cosine_similarity(image_features, text_features, dim=-1)
                            score = similarity.item()
                        
                        type_scores.append(score)
                        type_details.append({
                            'text': desc,
                            'alignment_score': score
                        })
                
                if type_scores:
                    alignment_scores[desc_type] = {
                        'average_score': np.mean(type_scores),
                        'max_score': np.max(type_scores),
                        'min_score': np.min(type_scores),
                        'count': len(type_scores)
                    }
                    detailed_scores[desc_type] = type_details
            
            # Calculate overall alignment score
            all_scores = []
            for type_scores in alignment_scores.values():
                all_scores.append(type_scores['average_score'])
            
            overall_alignment = np.mean(all_scores) if all_scores else 0.0
            
            return {
                'overall_score': overall_alignment,
                'category_scores': alignment_scores,
                'detailed_scores': detailed_scores,
                'method': 'clip_cosine_similarity'
            }
            
        except Exception as e:
            logger.error(f"CLIP alignment calculation failed: {e}")
            return {'error': str(e)}
    
    def calculate_semantic_coherence(self, text_descriptions: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate semantic coherence between text descriptions"""
        if not self.text_model:
            return {'error': 'Text model not available'}
        
        try:
            # Collect all text descriptions
            all_texts = []
            text_categories = []
            
            for category, descriptions in text_descriptions.items():
                for desc in descriptions:
                    if desc.strip():
                        all_texts.append(desc)
                        text_categories.append(category)
            
            if len(all_texts) < 2:
                return {'semantic_coherence_score': 1.0, 'note': 'Insufficient text for coherence analysis'}
            
            # Calculate text embeddings
            embeddings = self.text_model.encode(all_texts)
            
            # Calculate pairwise similarity
            similarities = []
            category_coherence = {}
            
            for i in range(len(embeddings)):
                for j in range(i + 1, len(embeddings)):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                    )
                    similarities.append(sim)
                    
                    # Calculate by category group
                    cat_pair = f"{text_categories[i]}_{text_categories[j]}"
                    if cat_pair not in category_coherence:
                        category_coherence[cat_pair] = []
                    category_coherence[cat_pair].append(sim)
            
            # Calculate overall semantic coherence
            overall_coherence = np.mean(similarities) if similarities else 0.0
            
            # Calculate coherence for each category pair
            category_coherence_scores = {}
            for cat_pair, sims in category_coherence.items():
                category_coherence_scores[cat_pair] = {
                    'average_similarity': np.mean(sims),
                    'count': len(sims)
                }
            
            return {
                'semantic_coherence_score': overall_coherence,
                'category_coherence': category_coherence_scores,
                'total_comparisons': len(similarities),
                'method': 'sentence_transformer_similarity'
            }
            
        except Exception as e:
            logger.error(f"Semantic coherence calculation failed: {e}")
            return {'error': str(e)}
    
    def calculate_vlm_semantic_alignment(self, json_data: Dict[str, Any], 
                                       image_path: str, 
                                       text_descriptions: Dict[str, List[str]]) -> Dict[str, Any]:
        """Calculate deep semantic alignment using VLM"""
        if self.vlm_wrapper is None:
            return {
                'error': 'VLM not available',
                'note': 'Deep semantic alignment requires VLM analysis'
            }
        
        try:
            # Collect main text descriptions
            main_descriptions = []
            
            # Prioritize overall description
            overall_desc = text_descriptions.get('overall_description', [])
            if overall_desc:
                main_descriptions.extend(overall_desc[:2])  # First two overall descriptions

            # Object names, then the first style; object details are already
            # in the overall description.
            objects = json_data.get('objects') or {}
            if isinstance(objects, dict):
                objects = objects.get('enhanced_objects') or []
            names = [str(obj['name']) for obj in objects
                     if isinstance(obj, dict) and obj.get('name')]
            if names:
                main_descriptions.append("objects: " + ", ".join(names))
            for style in text_descriptions.get('style_descriptions', [])[:1]:
                main_descriptions.append(f"style: {style}")
            
            if not main_descriptions:
                # Nothing to compare the image against, so no score.
                return {
                    'score': None,
                    'note': 'No text descriptions available for VLM analysis',
                    'method': 'no_text'
                }
            
            # Construct comprehensive analysis prompt
            combined_description = '; '.join(main_descriptions)
            
            prompt = f"""
Analyze the consistency between this image and its description.

Description: {combined_description}

Evaluate three things:
1. Content Coverage: is everything the description mentions actually present?
2. Semantic Similarity: does the image mean what the description says, beyond
   the objects being present?
3. Overall Alignment: taken as a whole, is this image what the description asks for?

Do not judge the art style here; it is assessed separately.

IMPORTANT: Provide your response in VALID JSON format only:

{{
  "content_coverage": 0.0,
  "semantic_similarity": 0.0,
  "overall_alignment": 0.0,
  "analysis": "detailed explanation of your evaluation"
}}

Replace the 0.0 values with your actual scores (0.0-1.0). Ensure the JSON is valid and parseable.
"""
            
            # Decoding is constrained to ALIGNMENT_JSON_SCHEMA.
            vlm_result = self.vlm_wrapper.generate_response(
                prompt, image_path,
                max_tokens=ALIGNMENT_MAX_TOKENS,
                json_schema=ALIGNMENT_JSON_SCHEMA,
            )

            if vlm_result.get('success', False):
                response = vlm_result.get('response', '')
                parsed_result = self._parse_vlm_alignment_response(response)

                # Mean of the dimensions that were returned; missing ones are skipped.
                scores = [parsed_result[key] for key in ALIGNMENT_DIMENSIONS
                          if parsed_result.get(key) is not None]
                overall_alignment = float(np.mean(scores)) if scores else None
                
                return {
                    'score': overall_alignment,
                    'content_coverage': parsed_result.get('content_coverage'),
                    'semantic_similarity': parsed_result.get('semantic_similarity'),
                    'overall_alignment': parsed_result.get('overall_alignment'),
                    'dimensions': list(ALIGNMENT_DIMENSIONS),
                    'analysis_explanation': parsed_result.get('explanation', ''),
                    'method': 'vlm_semantic_analysis',
                    'text_used': combined_description
                }
            else:
                # A failed call records no score.
                logger.warning(f"VLM alignment analysis failed: {vlm_result.get('error', 'Unknown error')}")
                return {
                    'score': None,
                    'note': 'VLM analysis failed; alignment not measured',
                    'error': vlm_result.get('error', 'Unknown VLM error'),
                    'method': 'vlm_failed'
                }

        except Exception as e:
            logger.error(f"VLM semantic alignment failed: {e}")
            return {
                'score': None,
                'error': str(e),
                'method': 'vlm_error'
            }
    
    def _parse_vlm_alignment_response(self, response: str) -> Dict[str, Any]:
        """Read the three dimension scores from a VLM reply; unreadable ones are None.

        Parses JSON first, then falls back to matching each field name, for
        unconstrained or truncated replies.
        """
        import json
        import re

        empty = {key: None for key in ALIGNMENT_DIMENSIONS}

        try:
            text = (response or "").strip()

            # Try a markdown-fenced block, then the outermost braces, then the raw text.
            fenced = re.search(r'```(?:json)?\s*(.*?)```', text, re.DOTALL)
            candidates = [fenced.group(1)] if fenced else []
            brace = re.search(r'\{.*\}', text, re.DOTALL)
            if brace:
                candidates.append(brace.group(0))
            candidates.append(text)

            # Method 1: real JSON.
            for candidate in candidates:
                try:
                    parsed_json = json.loads(candidate)
                except (json.JSONDecodeError, ValueError):
                    continue
                if not isinstance(parsed_json, dict):
                    continue

                result = dict(empty)
                for key in ALIGNMENT_DIMENSIONS:
                    value = parsed_json.get(key)
                    if value is None:
                        continue
                    try:
                        result[key] = float(value)
                    except (TypeError, ValueError):
                        result[key] = None
                if parsed_json.get('analysis'):
                    result['explanation'] = parsed_json['analysis']
                logger.info("Successfully parsed JSON response from VLM")
                return result

            # Method 2: read each number by its field name, for replies that are
            # not valid JSON (e.g. raw newlines inside `analysis`).
            result = dict(empty)
            found_any = False
            for key in ALIGNMENT_DIMENSIONS:
                match = re.search(
                    rf'["\']?{key}["\']?\s*[:=]\s*([0-9]*\.?[0-9]+)',
                    text, re.IGNORECASE)
                if match:
                    try:
                        result[key] = float(match.group(1))
                        found_any = True
                    except ValueError:
                        pass

            if found_any:
                logger.warning(
                    "Reply was not valid JSON; recovered "
                    f"{sum(v is not None for v in result.values())} of "
                    f"{len(ALIGNMENT_DIMENSIONS)} scores by field name.")
            else:
                logger.warning(
                    "Reply carried none of the three dimensions; reporting "
                    "them as not measured.")
            result['explanation'] = ''
            return result

        except Exception as e:
            logger.error(f"Failed to parse VLM alignment response: {e}")
            result = dict(empty)
            result['explanation'] = f'Parse failed: {str(e)}'
            result['error'] = str(e)
            return result
    
    def evaluate_completeness(self, json_data: Dict[str, Any], 
                            detected_objects: List[str]) -> Dict[str, Any]:
        """Evaluate description completeness"""
        try:
            objects = _objects_of(json_data)

            # Coverage is counted by instance_id; detected names are mapped
            # back to ids.
            name_by_id = {}
            for position, obj in enumerate(objects):
                instance_id = obj.get('instance_id') or f'obj_{position}'
                name_by_id[instance_id] = str(obj.get('name') or 'object')
            id_by_name = {}
            for instance_id, name in name_by_id.items():
                id_by_name.setdefault(name.lower(), instance_id)

            expected_objects = set(name_by_id)
            detected_set = set()
            for entry in detected_objects:
                key = str(entry)
                if key in name_by_id:
                    detected_set.add(key)
                elif key.lower() in id_by_name:
                    detected_set.add(id_by_name[key.lower()])

            # A specification with no objects counts as fully covered.
            covered_objects = expected_objects.intersection(detected_set)
            if expected_objects:
                object_coverage = len(covered_objects) / len(expected_objects)
            else:
                object_coverage = 1.0

            def label(instance_id):
                """Display name for an instance id."""
                return name_by_id.get(instance_id, instance_id)
            
            # Check key attribute coverage
            attribute_coverage = {}
            
            # Style coverage
            has_style = 'style' in json_data and json_data['style']
            attribute_coverage['style'] = 1.0 if has_style else 0.5
            
            # Color coverage: any object declares `main_colours` or `primary_colors`.
            has_colors = any(
                obj.get('main_colours') or obj.get('primary_colors')
                for obj in objects if isinstance(obj, dict)
            )
            attribute_coverage['color'] = 1.0 if has_colors else 0.5

            # Spatial relation coverage, from `composition`.
            composition = json_data.get('composition') or {}
            has_spatial = bool(
                composition.get('spatial_relations')
                or composition.get('spatial_relationships')
            )
            attribute_coverage['spatial'] = 1.0 if has_spatial else 0.5
            
            # Environment coverage
            has_environment = 'environment' in json_data and json_data['environment']
            attribute_coverage['environment'] = 1.0 if has_environment else 0.5
            
            # Overall completeness score
            overall_completeness = (
                object_coverage * 0.4 +
                np.mean(list(attribute_coverage.values())) * 0.6
            )
            
            return {
                'completeness_score': overall_completeness,
                'object_coverage': object_coverage,
                'attribute_coverage': attribute_coverage,
                'covered_objects': [label(i) for i in sorted(covered_objects)],
                'missing_objects': [label(i) for i in
                                    sorted(expected_objects - detected_set)],
                'extra_objects': [label(i) for i in
                                  sorted(detected_set - expected_objects)]
            }
            
        except Exception as e:
            logger.error(f"Completeness evaluation failed: {e}")
            return {'error': str(e)}
    
    def create_alignment_visualization(self, alignment_result: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> Optional[str]:
        """Plot the alignment result; save to `save_path` or show it."""
        try:
            score = alignment_result.get('overall_score')
            dimensions = alignment_result.get('dimensions') or {}
            completeness = alignment_result.get('completeness') or {}

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle('Text-Image Alignment (score: ' + ('-' if score is None else f'{score:.1f}/100') + ')',
                         fontsize=16, fontweight='bold')

            # 1. The three VLM dimensions (0-1)
            names = list(ALIGNMENT_DIMENSIONS)
            values = [dimensions.get(name) for name in names]
            if any(v is not None for v in values):
                axes[0].barh([n.replace('_', ' ') for n in names],
                             [v if v is not None else 0 for v in values], alpha=0.7)
                axes[0].invert_yaxis()
                axes[0].set_xlim(0, 1)
            else:
                axes[0].text(0.5, 0.5, 'No VLM answer', ha='center', va='center',
                             transform=axes[0].transAxes)
            axes[0].set_title('VLM dimensions')

            # 2. Object and attribute coverage of the specification
            coverage = {}
            if 'object_coverage' in completeness:
                coverage['objects'] = completeness['object_coverage']
            coverage.update(completeness.get('attribute_coverage') or {})
            if coverage:
                axes[1].barh(list(coverage), list(coverage.values()), alpha=0.7)
                axes[1].invert_yaxis()
                axes[1].set_xlim(0, 1)
            else:
                axes[1].text(0.5, 0.5, 'No completeness data', ha='center', va='center',
                             transform=axes[1].transAxes)
            axes[1].set_title('Completeness (not scored)')

            # 3. Summary
            lines = [f"Method: {alignment_result.get('method', '-')}"]
            for name in names:
                value = dimensions.get(name)
                lines.append(f"{name}: {'-' if value is None else f'{value:.2f}'}")
            if completeness.get('missing_objects'):
                lines += ["", "Missing objects: " + ", ".join(completeness['missing_objects'][:10])]
            if alignment_result.get('degraded_components'):
                lines += ["", "Not installed: " + ", ".join(alignment_result['degraded_components'])]
            axes[2].text(0.02, 0.98, "\n".join(lines), transform=axes[2].transAxes,
                         fontsize=11, verticalalignment='top', family='monospace')
            axes[2].set_title('Summary')
            axes[2].axis('off')

            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return save_path
            else:
                plt.show()
                return None
                
        except Exception as e:
            logger.error(f"Alignment visualization failed: {e}")
            if save_path:
                # Create simple error image
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.text(0.5, 0.5, f"Visualization failed:\n{str(e)}", 
                       ha='center', va='center', transform=ax.transAxes)
                plt.savefig(save_path, dpi=150)
                plt.close()
                return save_path
            return None
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str,
                       bboxes_dict: Optional[Dict[str, List[int]]] = None,
                       visualize: bool = False) -> Dict[str, Any]:
        """Main text-image alignment evaluation method"""
        detected_objects = list(bboxes_dict.keys()) if bboxes_dict else []
        return self.comprehensive_alignment_evaluation(json_data, image_path, detected_objects, visualize=visualize)
    
    def _assemble(self, image_path, overall_score, vlm_semantic_alignment,
                  clip_alignment, semantic_coherence, completeness,
                  text_descriptions, visualization_path):
        """Build the alignment result.

        `dimensions` holds the three VLM scores. OpenCLIP and SentenceTransformer
        results sit under `comparison_methods` and do not affect the score.
        """
        return {
            'image_path': image_path,
            'method': 'vlm_only',
            # None when the VLM did not answer.
            'overall_score': overall_score,
            'overall_alignment_score': (overall_score / 100.0
                                        if overall_score is not None else None),
            'dimensions': {
                key: vlm_semantic_alignment.get(key) for key in ALIGNMENT_DIMENSIONS
            },
            'vlm_semantic_alignment': vlm_semantic_alignment,
            'comparison_methods': {
                'openclip': clip_alignment,
                'sentence_transformer': semantic_coherence,
            },
            # Optional packages that are not installed.
            'degraded_components': [
                name for name, available in (
                    ('openclip', OPENCLIP_AVAILABLE),
                    ('sentence_transformers', SENTENCE_TRANSFORMERS_AVAILABLE),
                ) if not available
            ],
            'completeness': completeness,
            'text_descriptions': text_descriptions,
            'visualization_path': visualization_path,
        }

    def comprehensive_alignment_evaluation(self, json_data: Dict[str, Any], 
                                         image_path: str,
                                         detected_objects: Optional[List[str]] = None,
                                         visualize: bool = False) -> Dict[str, Any]:
        """Comprehensive alignment evaluation"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to load image: {image_path}")
            
            logger.info(f"Starting comprehensive alignment evaluation for: {image_path}")
            
            # Extract text descriptions
            text_descriptions = self.extract_text_descriptions(json_data)
            
            # Comparison methods: CLIP alignment and text coherence
            clip_alignment = self.calculate_clip_alignment(image, text_descriptions)

            semantic_coherence = self.calculate_semantic_coherence(text_descriptions)
            
            # Calculate deep semantic alignment using VLM
            vlm_semantic_alignment = self.calculate_vlm_semantic_alignment(
                json_data, image_path, text_descriptions
            )
            
            # Evaluate completeness
            completeness = self.evaluate_completeness(
                json_data, detected_objects or []
            ) if detected_objects is not None else {}
            
            # The score comes from the VLM alone; a failed VLM call gives None.
            if 'score' in vlm_semantic_alignment and 'error' not in vlm_semantic_alignment:
                vlm_score = vlm_semantic_alignment['score']
                overall_score = vlm_score * 100 if vlm_score is not None else None
            else:
                logger.warning("VLM alignment analysis failed; no score recorded")
                overall_score = None
            
            result = self._assemble(image_path, overall_score,
                                    vlm_semantic_alignment, clip_alignment,
                                    semantic_coherence, completeness,
                                    text_descriptions, None)

            if visualize:
                output_dir = Path(image_path).parent / "alignment_analysis"
                output_dir.mkdir(exist_ok=True)
                viz_path = output_dir / f"alignment_analysis_{Path(image_path).stem}.png"
                result['visualization_path'] = self.create_alignment_visualization(result, str(viz_path))

            if overall_score is None:
                logger.info("Alignment evaluation completed with no score (VLM unavailable)")
            else:
                logger.info(f"Alignment evaluation completed. Overall score: {overall_score:.3f}")

            return result
            
        except Exception as e:
            logger.error(f"Comprehensive alignment evaluation failed: {e}")
            # No evaluation took place; the score is None.
            return {
                'error': str(e),
                'image_path': image_path,
                'overall_score': None,
                'overall_alignment_score': None,
            }