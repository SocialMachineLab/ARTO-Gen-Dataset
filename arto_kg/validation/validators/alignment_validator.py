


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

logger = logging.getLogger(__name__)

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    logger.warning("OpenCLIP not available, alignment evaluation will be limited")
    OPENCLIP_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers not available, text similarity will be limited")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


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
        
        # Extract from final_prompts (new structure)
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
        
        # Object description - Supports new JSON structure
        objects_data = json_data.get('objects', {})
        
        # Extract detailed descriptions from enhanced_objects
        enhanced_objects = objects_data.get('enhanced_objects', [])
        for obj in enhanced_objects:
            if isinstance(obj, dict):
                # Object name
                if 'name' in obj:
                    descriptions['object_descriptions'].append(obj['name'])
                
                # Artistic description (new field)
                if 'artistic_description' in obj:
                    descriptions['object_descriptions'].append(obj['artistic_description'])
                
                # General description field
                if 'description' in obj:
                    descriptions['object_descriptions'].append(obj['description'])
                
                # Color description - Supports primary_colors array
                if 'primary_colors' in obj:
                    primary_colors = obj['primary_colors']
                    if isinstance(primary_colors, list):
                        descriptions['color_descriptions'].extend([str(c) for c in primary_colors])
                    else:
                        descriptions['color_descriptions'].append(str(primary_colors))
                
                # Compatible with old color field
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
                if 'material' in obj:
                    descriptions['object_descriptions'].append(f"material: {obj['material']}")
                
                # Size description
                if 'size' in obj:
                    descriptions['object_descriptions'].append(f"size: {obj['size']}")
        
        # Extract from basic object_names (compatible)
        object_names = objects_data.get('object_names', [])
        if isinstance(object_names, list):
            descriptions['object_descriptions'].extend([str(name) for name in object_names])
        
        # Compatible with old direct objects array structure
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
        
        # Environment description - Supports new nested structure
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
                
                # Color scheme
                if 'color_scheme' in env:
                    color_scheme = env['color_scheme']
                    if isinstance(color_scheme, dict):
                        # Main palette
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
                
                # Compatible with old format
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
                main_descriptions.extend(overall_desc[:2])  # Take top 2 most important descriptions
            
            # Supplement with object and style descriptions
            object_desc = text_descriptions.get('object_descriptions', [])[:3]  # Take top 3 objects
            style_desc = text_descriptions.get('style_descriptions', [])[:2]   # Take top 2 styles
            
            main_descriptions.extend(object_desc)
            main_descriptions.extend(style_desc)
            
            if not main_descriptions:
                return {
                    'score': 0.7,
                    'note': 'No text descriptions available for VLM analysis',
                    'method': 'no_text'
                }
            
            # Construct comprehensive analysis prompt
            combined_description = '; '.join(main_descriptions[:5])  # Limit length
            
            prompt = f"""
Analyze the consistency between this image and its description.

Description: {combined_description}

Please evaluate the semantic consistency from these aspects:
1. Content Consistency: How well does the image content match the description?
2. Visual-Textual Alignment: Are the visual elements aligned with textual descriptions?  
3. Style Consistency: Does the image style match the described style?
4. Overall Coherence: Does the image as a whole represent what is described?

IMPORTANT: Provide your response in VALID JSON format only:

{{
  "content_consistency": 0.0,
  "visual_textual_alignment": 0.0,
  "style_consistency": 0.0,
  "overall_coherence": 0.0,
  "analysis": "detailed explanation of your evaluation"
}}

Replace the 0.0 values with your actual scores (0.0-1.0). Ensure the JSON is valid and parseable.
"""
            
            # Call VLM for analysis
            vlm_result = self.vlm_wrapper.generate_response(prompt, image_path)
            
            # Add debug logs
            logger.info(f"VLM Alignment Analysis Debug:")
            logger.info(f"   Success: {vlm_result.get('success', False)}")
            logger.info(f"   Response length: {len(vlm_result.get('response', ''))}")
            logger.info(f"   Raw response: {vlm_result.get('response', '')[:500]}...")
            
            if vlm_result.get('success', False):
                response = vlm_result.get('response', '')
                # Parse VLM response
                parsed_result = self._parse_vlm_alignment_response(response)
                
                # Debug parsed results
                logger.info(f"   Parsed scores: {parsed_result}")
                logger.info(f"   Explanation: {parsed_result.get('explanation', 'No explanation')[:200]}...")
                
                # Calculate overall score
                scores = []
                for key in ['semantic_similarity', 'content_coverage', 'style_consistency', 'overall_harmony']:
                    if key in parsed_result:
                        scores.append(parsed_result[key])
                
                overall_alignment = np.mean(scores) if scores else 0.5
                
                return {
                    'score': overall_alignment,
                    'semantic_similarity': parsed_result.get('semantic_similarity', 0.5),
                    'content_coverage': parsed_result.get('content_coverage', 0.5),
                    'style_consistency': parsed_result.get('style_consistency', 0.5),
                    'overall_harmony': parsed_result.get('overall_harmony', 0.5),
                    'analysis_explanation': parsed_result.get('explanation', ''),
                    'method': 'vlm_semantic_analysis',
                    'text_used': combined_description
                }
            else:
                # Fallback response when VLM call fails
                logger.warning(f"VLM alignment analysis failed: {vlm_result.get('error', 'Unknown error')}")
                return {
                    'score': 0.5,
                    'note': 'VLM analysis failed, using fallback score',
                    'error': vlm_result.get('error', 'Unknown VLM error'),
                    'method': 'vlm_failed'
                }
                
        except Exception as e:
            logger.error(f"VLM semantic alignment failed: {e}")
            return {
                'score': 0.4,
                'error': str(e),
                'method': 'vlm_error'
            }
    
    def _parse_vlm_alignment_response(self, response: str) -> Dict[str, Any]:
        """Parse VLM alignment analysis response"""
        try:
            import json
            import re
            result = {}
            
            # Method 1: Try parsing JSON directly
            try:
                # Find JSON block
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                    parsed_json = json.loads(json_text)
                    
                    # Map JSON fields to internal field names
                    field_mapping = {
                        'content_consistency': 'semantic_similarity',
                        'visual_textual_alignment': 'content_coverage', 
                        'style_consistency': 'style_consistency',
                        'overall_coherence': 'overall_harmony'
                    }
                    
                    for json_key, internal_key in field_mapping.items():
                        if json_key in parsed_json:
                            result[internal_key] = float(parsed_json[json_key])
                        else:
                            result[internal_key] = 0.5
                    
                    # Extract analysis explanation
                    if 'analysis' in parsed_json:
                        result['explanation'] = parsed_json['analysis']
                    
                    logger.info("Successfully parsed JSON response from VLM")
                    return result
                    
            except (json.JSONDecodeError, ValueError, KeyError) as json_error:
                logger.warning(f"JSON parsing failed: {json_error}, falling back to regex")
            
            # Method 2: Fallback regex parsing (maintain original logic)
            patterns = {
                'semantic_similarity': [
                    r'Content Consistency:\s*([0-9\.]+)',
                    r'\*?\*?Content Consistency\*?\*?:\s*([0-9\.]+)',
                    r'- \*?\*?Content Consistency\*?\*?:\s*([0-9\.]+)',
                    r'"content_consistency":\s*([0-9\.]+)'
                ],
                'content_coverage': [
                    r'Visual-Textual Alignment:\s*([0-9\.]+)',
                    r'\*?\*?Visual-Textual Alignment\*?\*?:\s*([0-9\.]+)', 
                    r'- \*?\*?Visual-Textual Alignment\*?\*?:\s*([0-9\.]+)',
                    r'"visual_textual_alignment":\s*([0-9\.]+)'
                ],
                'style_consistency': [
                    r'Style Consistency:\s*([0-9\.]+)',
                    r'\*?\*?Style Consistency\*?\*?:\s*([0-9\.]+)',
                    r'- \*?\*?Style Consistency\*?\*?:\s*([0-9\.]+)',
                    r'"style_consistency":\s*([0-9\.]+)'
                ],
                'overall_harmony': [
                    r'Overall Coherence:\s*([0-9\.]+)',
                    r'\*?\*?Overall Coherence\*?\*?:\s*([0-9\.]+)',
                    r'- \*?\*?Overall Coherence\*?\*?:\s*([0-9\.]+)',
                    r'"overall_coherence":\s*([0-9\.]+)'
                ]
            }
            
            for key, pattern_list in patterns.items():
                found = False
                for pattern in pattern_list:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        try:
                            result[key] = float(match.group(1))
                            found = True
                            break
                        except:
                            continue
                if not found:
                    result[key] = 0.5
            
            # Extract analysis explanation
            analysis_match = re.search(r'Analysis:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
            if analysis_match:
                result['explanation'] = analysis_match.group(1).strip()
            else:
                result['explanation'] = 'VLM analysis completed'
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse VLM alignment response: {e}")
            return {
                'semantic_similarity': 0.4,
                'content_coverage': 0.4,
                'style_consistency': 0.4,
                'overall_harmony': 0.4,
                'explanation': f'Parse failed: {str(e)}',
                'error': str(e)
            }
    
    def evaluate_completeness(self, json_data: Dict[str, Any], 
                            detected_objects: List[str]) -> Dict[str, Any]:
        """Evaluate description completeness"""
        try:
            # Expected objects
            expected_objects = set()
            objects = json_data.get('objects', [])
            for obj in objects:
                if isinstance(obj, dict) and 'name' in obj:
                    expected_objects.add(obj['name'].lower())
                elif isinstance(obj, str):
                    expected_objects.add(obj.lower())
            
            # Detected objects
            detected_set = set([obj.lower() for obj in detected_objects])
            
            # Calculate coverage
            if expected_objects:
                covered_objects = expected_objects.intersection(detected_set)
                object_coverage = len(covered_objects) / len(expected_objects)
            else:
                object_coverage = 1.0
            
            # Check key attribute coverage
            attribute_coverage = {}
            
            # Style coverage
            has_style = 'style' in json_data and json_data['style']
            attribute_coverage['style'] = 1.0 if has_style else 0.5
            
            # Color coverage
            has_colors = any('color' in str(obj).lower() for obj in objects) or \
                        ('environment' in json_data and 'color' in json_data.get('environment', {}))
            attribute_coverage['color'] = 1.0 if has_colors else 0.5
            
            # Spatial relation coverage
            has_spatial = any('spatial' in str(obj).lower() or 'position' in str(obj).lower() 
                            for obj in objects)
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
                'covered_objects': list(expected_objects.intersection(detected_set)),
                'missing_objects': list(expected_objects - detected_set),
                'extra_objects': list(detected_set - expected_objects)
            }
            
        except Exception as e:
            logger.error(f"Completeness evaluation failed: {e}")
            return {'error': str(e)}
    
    def create_alignment_visualization(self, alignment_result: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> Optional[str]:
        """Create alignment analysis visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Text-Image Alignment Analysis (Overall: {alignment_result.get("overall_score", 0):.3f})', 
                        fontsize=16, fontweight='bold')
            
            # 1. Category alignment scores
            if 'category_scores' in alignment_result:
                categories = []
                scores = []
                
                for category, score_data in alignment_result['category_scores'].items():
                    if isinstance(score_data, dict) and 'average_score' in score_data:
                        categories.append(category.replace('_', '\n'))
                        scores.append(score_data['average_score'])
                
                if categories and scores:
                    colors = ['green' if s > 0.3 else 'orange' if s > 0.2 else 'red' for s in scores]
                    bars = axes[0, 0].bar(categories, scores, color=colors, alpha=0.7)
                    
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        axes[0, 0].text(bar.get_x() + bar.get_width()/2, height + 0.01,
                                       f'{score:.3f}', ha='center', va='bottom', fontsize=10)
                    
                    axes[0, 0].set_ylim(0, 1)
                    axes[0, 0].set_ylabel('Alignment Score')
                    axes[0, 0].set_title('Category Alignment Scores')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No category data', ha='center', va='center', 
                                   transform=axes[0, 0].transAxes)
                    axes[0, 0].set_title('Category Alignment Scores')
            
            # 2. Semantic coherence
            if 'semantic_coherence' in alignment_result:
                coherence_data = alignment_result['semantic_coherence']
                coherence_score = coherence_data.get('semantic_coherence_score', 0)
                
                # Pie chart showing coherence levels
                labels = ['Coherent', 'Incoherent']
                sizes = [coherence_score, 1 - coherence_score]
                colors = ['lightgreen', 'lightcoral']
                
                axes[0, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title(f'Semantic Coherence: {coherence_score:.3f}')
            else:
                axes[0, 1].text(0.5, 0.5, 'No coherence data', ha='center', va='center', 
                               transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Semantic Coherence')
            
            # 3. Completeness analysis
            if 'completeness' in alignment_result:
                completeness_data = alignment_result['completeness']
                
                # Show object and attribute coverage
                coverage_types = ['Objects', 'Style', 'Color', 'Spatial', 'Environment']
                coverage_scores = [
                    completeness_data.get('object_coverage', 0),
                    completeness_data.get('attribute_coverage', {}).get('style', 0),
                    completeness_data.get('attribute_coverage', {}).get('color', 0),
                    completeness_data.get('attribute_coverage', {}).get('spatial', 0),
                    completeness_data.get('attribute_coverage', {}).get('environment', 0)
                ]
                
                colors = ['green' if s > 0.8 else 'orange' if s > 0.6 else 'red' for s in coverage_scores]
                bars = axes[1, 0].barh(coverage_types, coverage_scores, color=colors, alpha=0.7)
                
                for bar, score in zip(bars, coverage_scores):
                    width = bar.get_width()
                    axes[1, 0].text(width + 0.02, bar.get_y() + bar.get_height()/2,
                                   f'{score:.3f}', va='center', fontsize=10)
                
                axes[1, 0].set_xlim(0, 1.1)
                axes[1, 0].set_xlabel('Coverage Score')
                axes[1, 0].set_title('Description Completeness')
            else:
                axes[1, 0].text(0.5, 0.5, 'No completeness data', ha='center', va='center', 
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Description Completeness')
            
            # 4. Summary information
            summary_text = f"""Overall Alignment: {alignment_result.get('overall_score', 0):.3f}
            
Method: {alignment_result.get('method', 'N/A')}

Top Aligned Categories:"""
            
            if 'category_scores' in alignment_result:
                sorted_categories = sorted(
                    alignment_result['category_scores'].items(),
                    key=lambda x: x[1].get('average_score', 0) if isinstance(x[1], dict) else 0,
                    reverse=True
                )[:3]
                
                for i, (cat, score_data) in enumerate(sorted_categories, 1):
                    if isinstance(score_data, dict):
                        score = score_data.get('average_score', 0)
                        summary_text += f"\n{i}. {cat}: {score:.3f}"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[1, 1].set_title('Alignment Summary')
            axes[1, 1].axis('off')
            
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
            
            # Calculate image-text alignment using CLIP
            clip_alignment = self.calculate_clip_alignment(image, text_descriptions)
            
            # Calculate semantic coherence
            semantic_coherence = self.calculate_semantic_coherence(text_descriptions)
            
            # Calculate deep semantic alignment using VLM
            vlm_semantic_alignment = self.calculate_vlm_semantic_alignment(
                json_data, image_path, text_descriptions
            )
            
            # Evaluate completeness
            completeness = self.evaluate_completeness(
                json_data, detected_objects or []
            ) if detected_objects is not None else {}
            
            # Calculate overall score (Prioritize VLM, CLIP as supplement/fallback)
            if 'score' in vlm_semantic_alignment and 'error' not in vlm_semantic_alignment:
                # VLM analysis successful
                vlm_score = vlm_semantic_alignment['score']
                clip_score = clip_alignment.get('overall_score', 0.0)
                
                # Overall score: 70% VLM + 30% CLIP (if CLIP available)
                if clip_score > 0:
                    overall_score = (vlm_score * 0.7 + clip_score * 0.3) * 100
                else:
                    overall_score = vlm_score * 100
            elif 'overall_score' in clip_alignment:
                # VLM failed, fallback to CLIP alignment score
                logger.warning("VLM analysis failed, falling back to CLIP alignment score")
                overall_score = clip_alignment['overall_score'] * 100
            else:
                overall_score = 0.0
            
            # Generate visualization
            viz_path = None
            if visualize:
                output_dir = Path(image_path).parent / "alignment_analysis"
                output_dir.mkdir(exist_ok=True)
                
                viz_path = output_dir / f"alignment_analysis_{Path(image_path).stem}.png"
                
                result = {
                    'image_path': image_path,
                    'overall_score': overall_score,
                    'vlm_semantic_alignment': vlm_semantic_alignment,
                    'clip_alignment': clip_alignment,
                    'semantic_coherence': semantic_coherence,
                    'completeness': completeness,
                    'overall_alignment_score': overall_score / 100.0, # normalized 0-1
                    'text_descriptions': text_descriptions
                }
                self.create_alignment_visualization(result, str(viz_path))
            
            # Construct final result (without circular dependency if possible, reused code structure)
            result = {
                'image_path': image_path,
                'overall_score': overall_score,
                'vlm_semantic_alignment': vlm_semantic_alignment,
                'clip_alignment': clip_alignment,
                'semantic_coherence': semantic_coherence,
                'completeness': completeness,
                'overall_alignment_score': overall_score / 100.0, # normalized 0-1
                'text_descriptions': text_descriptions,
                'visualization_path': str(viz_path) if viz_path else None
            }
            
            logger.info(f"Alignment evaluation completed. Overall score: {overall_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive alignment evaluation failed: {e}")
            return {
                'error': str(e),
                'image_path': image_path,
                'overall_score': 0.0
            }