
"""
Art Style Evaluation Module
Uses CLIP model for zero-shot style classification
"""

import logging
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import open_clip
    OPENCLIP_AVAILABLE = True
except ImportError:
    logger.warning("OpenCLIP not available, style evaluation will be limited")
    OPENCLIP_AVAILABLE = False


class StyleValidator:
    """Artistic style validator"""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize style validator
        
        Args:
            device: Computation device ('cuda', 'cpu', or None for auto selection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip_model = None
        self.clip_preprocess = None
        
        # Fixed 5 art style categories (updated for experiment styles)
        self.art_styles = [
            "Baroque",
            "Impressionism",
            "Chinese Ink Painting",
            # "Realism",
            "Post-Impressionism",
            "Neoclassicism"
        ]

        # Concise, high-discriminability CLIP prompts (representative descriptions for each style)
        self.style_prompts = {
            "Baroque": "a Baroque painting with dramatic lighting, rich ornamentation, and dynamic composition",
            "Impressionism": "an Impressionist painting with visible brushstrokes and light effects",
            "Chinese Ink Painting": "a traditional Chinese ink wash painting with brush calligraphy",
            # "Realism": "a Realist painting depicting everyday life with accurate and detailed representation",
            "Post-Impressionism": "a Post-Impressionist painting with bold colors, expressive brushwork, and geometric forms",

            "Neoclassicism": "a Neoclassical painting with classical Greek and Roman themes, harmonious composition"
        }

        # Simplified style category mapping (based on 5 fixed styles)
        self.style_categories = {
            'classical': ['Baroque', 'Neoclassicism'],
            'modern': ['Impressionism',"Post-Impressionism"],
            'traditional': ['Chinese Ink Painting'],

            # 'realistic': ['Realism']
        }

        # Style synonym mapping (handling different expressions)
        self.style_synonyms = {
            'Baroque': ['baroque', 'baroque style', 'baroque painting', 'baroque period', 'baroque art'],
            'Impressionism': ['impressionism', 'impressionist', 'impressionist painting', 'impressionist style'],
            'Chinese Ink Painting': ['chinese ink painting', 'ink painting', 'chinese painting', 'traditional chinese painting', 'sumi-e', 'ink wash', 'chinese'],
            # 'Realism': ['realism', 'realist', 'realistic', 'realist painting', 'realist style', 'naturalism'],
            'Post-Impressionism': ['post-impressionism', 'post impressionism', 'postimpressionism', 'post-impressionist', 'post impressionist', 'postimpressionist'],  
            'Neoclassicism': ['neoclassicism', 'neoclassical', 'neoclassical style', 'neoclassical painting', 'neo-classical']
        }
        
        self.initialize_clip_model()
    
    def initialize_clip_model(self):
        """Initialize OpenCLIP model"""
        if not OPENCLIP_AVAILABLE:
            logger.warning("OpenCLIP not available, using fallback style analysis")
            return False
        
        try:
            # Load OpenCLIP model
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device
            )
            self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
            logger.info(f"OpenCLIP model loaded successfully on {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}")
            self.clip_model = None
            return False
    
    def classify_style_with_clip(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify style using CLIP - Competitive classification optimized version"""
        if not self.clip_model:
            return self._fallback_style_analysis(image)
        
        try:
            # Convert image format
            if image.shape[-1] == 3:  # BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            
            # Preprocess image
            image_input = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                
                # Collect prompts for all styles for competitive classification
                all_prompts = []
                style_order = []
                
                for style in self.art_styles:
                    all_prompts.append(self.style_prompts[style])
                    style_order.append(style)
                
                # Compute similarity for all style prompts at once
                text_inputs = self.clip_tokenizer(all_prompts).to(self.device)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # Calculate similarity - Key: Global softmax for 5-style competition
                logits_per_image = (image_features @ text_features.T) * 100
                probs = logits_per_image.softmax(dim=-1)  # Global competitive softmax
                
                # Build style score dict
                style_scores = {}
                for i, style in enumerate(style_order):
                    style_scores[style] = probs[0, i].item()
            
            # Sort by score
            sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Build prediction results
            style_predictions = []
            for i, (style, score) in enumerate(sorted_styles):
                style_predictions.append({
                    'style': style,
                    'confidence': score,
                    'rank': i + 1
                })
            
            # Aggregate by category
            category_scores = self._aggregate_style_categories(style_predictions)
            
            # Primary style
            primary_style = style_predictions[0]['style']
            primary_confidence = style_predictions[0]['confidence']
            
            logger.info(f"CLIP Competitive Style Analysis: {primary_style} ({primary_confidence:.3f})")
            for pred in style_predictions:  # Show scores for all 5 styles
                logger.info(f"   {pred['rank']}. {pred['style']}: {pred['confidence']:.3f}")
            
            # Calculate score distribution clarity (gap between top1 and others)
            score_gap = style_predictions[0]['confidence'] - style_predictions[1]['confidence']
            logger.info(f"   Classification clarity (top1-top2 gap): {score_gap:.3f}")
            
            return {
                'primary_style': primary_style,
                'primary_confidence': primary_confidence,
                'top_styles': style_predictions,
                'category_scores': category_scores,
                'method': 'clip_competitive_classification'
            }
            
        except Exception as e:
            logger.error(f"CLIP style classification failed: {e}")
            return self._fallback_style_analysis(image)
    
    def _aggregate_style_categories(self, style_predictions: List[Dict]) -> Dict[str, float]:
        """Aggregate style predictions into broad categories"""
        category_scores = {cat: 0.0 for cat in self.style_categories.keys()}
        
        for pred in style_predictions:
            style = pred['style']
            confidence = pred['confidence']
            
            # Find which category this style belongs to
            for category, styles in self.style_categories.items():
                if style in styles:
                    category_scores[category] += confidence
                    break
        
        return category_scores
    
    def _fallback_style_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback style analysis (based on traditional CV features)"""
        try:
            # Simple style analysis based on color and texture
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Edge detection intensity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Color saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])
            
            # Brightness variance
            brightness_var = np.var(hsv[:, :, 2])
            
            # Simple heuristic classification
            if edge_density > 0.15:
                primary_style = "sketch" if saturation_mean < 50 else "detailed painting"
            elif saturation_mean > 150:
                primary_style = "vibrant art"
            elif brightness_var < 1000:
                primary_style = "minimalist art"
            else:
                primary_style = "realistic painting"
            
            return {
                'primary_style': primary_style,
                'primary_confidence': 0.7,  # Lower confidence
                'top_styles': [{'style': primary_style, 'confidence': 0.7, 'rank': 1}],
                'category_scores': {'realistic': 0.7},
                'method': 'traditional_cv',
                'features': {
                    'edge_density': edge_density,
                    'saturation_mean': saturation_mean / 255.0,
                    'brightness_variance': brightness_var / 10000.0
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback style analysis failed: {e}")
            return {
                'primary_style': 'unknown',
                'primary_confidence': 0.0,
                'top_styles': [],
                'category_scores': {},
                'method': 'failed',
                'error': str(e)
            }
    
    def evaluate_style_match(self, image: np.ndarray, 
                           expected_styles: List[str]) -> Dict[str, Any]:
        """Evaluate style match"""
        try:
            # Get style predictions
            style_analysis = self.classify_style_with_clip(image)
            
            if not style_analysis.get('top_styles'):
                return {
                    'style_match_score': 0.0,
                    'style_analysis': style_analysis,
                    'expected_styles': expected_styles,
                    'matches': [],
                    'evaluation_details': 'Style analysis failed'
                }
            
            # Calculate match score
            matches = []
            total_match_score = 0.0
            
            for expected_style in expected_styles:
                best_match = None
                best_score = 0.0
                
                # Find best match in predictions
                for pred in style_analysis['top_styles']:
                    predicted_style = pred['style']
                    confidence = pred['confidence']
                    
                    # Calculate match score (can use more complex semantic similarity)
                    match_score = self._calculate_style_similarity(expected_style, predicted_style)
                    final_score = match_score * confidence
                    
                    if final_score > best_score:
                        best_score = final_score
                        best_match = {
                            'expected': expected_style,
                            'predicted': predicted_style,
                            'match_score': match_score,
                            'confidence': confidence,
                            'final_score': final_score
                        }
                
                if best_match:
                    matches.append(best_match)
                    total_match_score += best_match['final_score']
            
            # Normalize score
            final_score = min(total_match_score / len(expected_styles), 1.0) if expected_styles else 0.0
            
            return {
                'style_match_score': final_score,
                'style_analysis': style_analysis,
                'expected_styles': expected_styles,
                'matches': matches,
                'evaluation_details': f"Matched {len(matches)}/{len(expected_styles)} expected styles"
            }
            
        except Exception as e:
            logger.error(f"Style evaluation failed: {e}")
            return {
                'style_match_score': 0.0,
                'error': str(e),
                'expected_styles': expected_styles
            }
    
    def _calculate_style_similarity(self, style1: str, style2: str) -> float:
        """Calculate similarity between two styles - Optimized version based on 5 fixed categories"""
        style1_lower = style1.lower().strip()
        style2_lower = style2.lower().strip()
        
        # 1. Exact match
        if style1_lower == style2_lower:
            return 1.0
        
        # 2. Check synonym mapping
        for main_style, synonyms in self.style_synonyms.items():
            main_style_lower = main_style.lower()
            synonyms_lower = [s.lower() for s in synonyms]
            
            # Check if one is main style and other is synonym
            if ((style1_lower == main_style_lower and style2_lower in synonyms_lower) or
                (style2_lower == main_style_lower and style1_lower in synonyms_lower)):
                return 1.0  # Main style and synonym match perfectly
            
            # Check if both are synonyms of the same main style
            if style1_lower in synonyms_lower and style2_lower in synonyms_lower:
                return 1.0
        
        # 3. Inclusion relation (stricter check)
        if style1_lower in style2_lower or style2_lower in style1_lower:
            return 0.8
        
        # 4. Keyword match (for compound words)
        words1 = set(style1_lower.split())
        words2 = set(style2_lower.split())
        common_words = words1.intersection(words2)
        
        if common_words:
            similarity = len(common_words) / max(len(words1), len(words2))
            if similarity >= 0.5:  # At least 50% words match
                return 0.6
        
        # 5. Handle common confusion cases specially
        special_matches = {
            ('baroque', 'baroque style'): 1.0,
            ('baroque', 'baroque art'): 1.0,
            ('impressionism', 'impressionist'): 1.0,
            ('realism', 'realistic'): 1.0,
            ('realism', 'realist'): 1.0,
            ('neoclassicism', 'neoclassical'): 1.0,
            ('neoclassicism', 'classical'): 0.8,
            ('chinese ink painting', 'ink painting'): 1.0,
            ('chinese ink painting', 'traditional chinese painting'): 1.0,
            ('chinese ink painting', 'chinese'): 1.0
        }
        
        for (s1, s2), score in special_matches.items():
            if ((style1_lower == s1 and style2_lower == s2) or 
                (style1_lower == s2 and style2_lower == s1)):
                return score
        
        # 6. No match
        return 0.0
    
    def main_evaluation(self, json_data: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """Main evaluation function, conforms to validation system interface"""
        try:
            print("Starting comprehensive style evaluation for:", Path(image_path).name)
            
            # Load image
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Extract expected style information from JSON
            expected_styles = self._extract_expected_styles(json_data)
            print(f"   Found {len(expected_styles)} expected styles")
            
            # Execute style analysis
            style_analysis = self.classify_style_with_clip(image)
            print(f"   Detected primary style: {style_analysis.get('primary_style', 'unknown')}")
            
            # Calculate style match score
            if expected_styles:
                evaluation_result = self.evaluate_style_match(image, expected_styles)
                style_match_score = evaluation_result.get('style_match_score', 0.0)
            else:
                # If no expected styles, score based on analysis quality
                style_match_score = style_analysis.get('primary_confidence', 0.0)
                evaluation_result = {
                    'style_match_score': style_match_score,
                    'style_analysis': style_analysis,
                    'expected_styles': [],
                    'note': 'No expected styles found, scored based on analysis confidence'
                }
            
            # Calculate overall score (0-100)
            overall_score = evaluation_result['style_match_score'] * 100
            
            print(f"   Style evaluation completed! Overall score: {overall_score:.1f}/100")
            
            # Create visualization path
            visualization_path = None
            if image_path:
                import os
                output_dir = os.path.join(os.path.dirname(image_path), 'style_analysis')
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                visualization_path = os.path.join(output_dir, f'style_analysis_{base_name}.png')
            
            # Return result in validation system format
            return {
                'image_path': image_path,
                'style_analysis': style_analysis,
                'style_match_result': evaluation_result,
                'visualization_path': visualization_path,
                'evaluation_timestamp': None,
                'score': overall_score
            }
            
        except Exception as e:
            logger.error(f"Style evaluation failed: {e}")
            return {
                'image_path': image_path,
                'style_analysis': {'error': str(e)},
                'style_match_result': {'error': str(e)},
                'visualization_path': None,
                'evaluation_timestamp': None,
                'score': 0.0
            }
    
    def _extract_expected_styles(self, json_data: Dict[str, Any]) -> List[str]:
        """Extract expected style information from JSON - Optimized version for 5 fixed categories"""
        expected_styles = []
        
        # 1. Check top-level direct style fields
        style_fields = ['style', 'art_style', 'artistic_style', 'painting_style', 'visual_style']
        
        for field in style_fields:
            if field in json_data:
                style_value = json_data[field]
                if isinstance(style_value, str):
                    normalized_style = self._normalize_style_name(style_value)
                    if normalized_style:
                        expected_styles.append(normalized_style)
                elif isinstance(style_value, list):
                    for s in style_value:
                        if isinstance(s, str):
                            normalized_style = self._normalize_style_name(s)
                            if normalized_style:
                                expected_styles.append(normalized_style)
        
        # 2. Check artistic_description in enhanced_objects within objects
        objects_data = json_data.get('objects', {})
        enhanced_objects = objects_data.get('enhanced_objects', [])
        for obj in enhanced_objects:
            if isinstance(obj, dict):
                artistic_desc = obj.get('artistic_description', '')
                if artistic_desc:
                    found_style = self._extract_style_from_description(artistic_desc)
                    if found_style:
                        expected_styles.append(found_style)
        
        # 3. Check time_period style info in environment
        environment = json_data.get('environment', {})
        time_period = environment.get('time_period', '')
        if time_period:
            found_style = self._normalize_style_name(time_period)
            if found_style:
                expected_styles.append(found_style)
        
        # 4. Check style info in metadata
        metadata = json_data.get('metadata', {})
        for field in style_fields:
            if field in metadata:
                style_value = metadata[field]
                if isinstance(style_value, str):
                    normalized_style = self._normalize_style_name(style_value)
                    if normalized_style:
                        expected_styles.append(normalized_style)
        
        # 5. Infer style from various description texts
        description_fields = ['description', 'caption', 'prompt', 'text', 'overall_description']
        for field in description_fields:
            if field in json_data:
                desc = json_data[field]
                if isinstance(desc, str):
                    found_style = self._extract_style_from_description(desc)
                    if found_style:
                        expected_styles.append(found_style)
        
        # Deduplicate and return
        unique_styles = list(set(expected_styles))
        logger.info(f"Extracted expected styles: {unique_styles}")
        return unique_styles
    
    def _normalize_style_name(self, style_input: str) -> Optional[str]:
        """Normalize input style name to one of 5 fixed categories"""
        if not style_input:
            return None
            
        style_lower = style_input.lower().strip()
        
        # Directly match main style name
        for main_style in self.art_styles:
            if style_lower == main_style.lower():
                return main_style
        
        # Match synonyms
        for main_style, synonyms in self.style_synonyms.items():
            if style_lower in [s.lower() for s in synonyms]:
                return main_style
        
        return None
    
    def _extract_style_from_description(self, description: str) -> Optional[str]:
        """Extract style information from description text"""
        if not description:
            return None
            
        desc_lower = description.lower()
        
        # Search keywords for 5 main styles
        style_keywords = {
            'Baroque': ['baroque', 'baroque style', 'baroque period', 'baroque art', 'ornate'],
            'Impressionism': ['impressionist', 'impressionism', 'plein air', 'impressionist style'],
            'Chinese Ink Painting': ['ink', 'chinese', 'traditional chinese', 'sumi-e', 'ink wash', 'brush painting'],
            # 'Realism': ['realism', 'realist', 'realistic', 'naturalism', 'naturalistic'],
            'Post-Impressionism': ['post-impressionism', 'post impressionism', 'postimpressionism', 'van gogh', 'cezanne', 'gauguin'],  
            'Neoclassicism': ['neoclassicism', 'neoclassical', 'classical', 'greco-roman', 'neo-classical']
        }
        
        for style, keywords in style_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return style
        
        return None
    
    def create_style_visualization(self, image: np.ndarray, 
                                 style_analysis: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> Optional[str]:
        """Create style analysis visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Show original image
            if image.shape[-1] == 3:
                display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                display_image = image
                
            axes[0, 0].imshow(display_image)
            axes[0, 0].set_title('Original Image', fontsize=14)
            axes[0, 0].axis('off')
            
            # Top style prediction bar chart
            if style_analysis.get('top_styles'):
                styles = [s['style'] for s in style_analysis['top_styles']]
                confidences = [s['confidence'] for s in style_analysis['top_styles']]
                
                axes[0, 1].barh(styles, confidences)
                axes[0, 1].set_title('Top Style Predictions', fontsize=14)
                axes[0, 1].set_xlabel('Confidence')
                
                # Add value labels
                for i, (style, conf) in enumerate(zip(styles, confidences)):
                    axes[0, 1].text(conf + 0.01, i, f'{conf:.3f}', 
                                   va='center', fontsize=10)
            
            # Category scores
            if style_analysis.get('category_scores'):
                categories = list(style_analysis['category_scores'].keys())
                scores = list(style_analysis['category_scores'].values())
                
                axes[1, 0].pie(scores, labels=categories, autopct='%1.1f%%')
                axes[1, 0].set_title('Style Category Distribution', fontsize=14)
            
            # Style information text
            info_text = f"Primary Style: {style_analysis.get('primary_style', 'N/A')}\n"
            info_text += f"Confidence: {style_analysis.get('primary_confidence', 0):.3f}\n"
            info_text += f"Method: {style_analysis.get('method', 'N/A')}\n"
            
            if 'features' in style_analysis:
                info_text += "\nTraditional CV Features:\n"
                for feature, value in style_analysis['features'].items():
                    info_text += f"  {feature}: {value:.3f}\n"
            
            axes[1, 1].text(0.1, 0.9, info_text, transform=axes[1, 1].transAxes, 
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
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
            logger.error(f"Style visualization failed: {e}")
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
                       bboxes_dict: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """Main style evaluation method"""
        # Get expected styles
        expected_styles = []
        if 'style' in json_data:
            if isinstance(json_data['style'], list):
                expected_styles = json_data['style']
            else:
                expected_styles = [json_data['style']]
        
        return self.comprehensive_style_evaluation(image_path, expected_styles)
    
    def comprehensive_style_evaluation(self, image_path: str,
                                     expected_styles: Optional[List[str]] = None) -> Dict[str, Any]:
        """Comprehensive style evaluation"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to load image: {image_path}")
            
            logger.info(f"Starting comprehensive style evaluation for: {image_path}")
            
            # Style classification
            style_analysis = self.classify_style_with_clip(image)
            
            # If expected styles exist, calculate match score
            style_match_result = None
            if expected_styles:
                style_match_result = self.evaluate_style_match(image, expected_styles)
            
            # Generate visualization
            output_dir = Path(image_path).parent / "style_analysis"
            output_dir.mkdir(exist_ok=True)
            
            viz_path = output_dir / f"style_analysis_{Path(image_path).stem}.png"
            self.create_style_visualization(image, style_analysis, str(viz_path))
            
            # Summarize results
            result = {
                'image_path': image_path,
                'style_analysis': style_analysis,
                'style_match_result': style_match_result,
                'visualization_path': str(viz_path),
                'evaluation_timestamp': None,
                'score': (style_match_result['style_match_score'] * 100) if style_match_result else (style_analysis['primary_confidence'] * 100)
            }
            
            logger.info(f"Style evaluation completed. Primary style: {style_analysis.get('primary_style')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Comprehensive style evaluation failed: {e}")
            return {
                'error': str(e),
                'image_path': image_path,
                'score': 0.0
            }
            
            
