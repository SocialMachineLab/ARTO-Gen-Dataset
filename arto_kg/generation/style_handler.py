"""
Style Processing Module
Extract and process art style information from JSON data
"""

from typing import Dict, List, Any, Optional, Tuple


class StyleHandler:
    """Style Processor"""
    
    def __init__(self):
        # Style mapping and normalization
        self.style_mapping = {
            # Painting styles
            'oil_painting': ['oil painting', 'oil on canvas', 'traditional painting'],
            'watercolor': ['watercolor', 'watercolor painting', 'aquarelle'],
            'acrylic': ['acrylic painting', 'acrylic on canvas'],
            'digital_painting': ['digital art', 'digital painting', 'concept art'],
            
            # Art movements
            'renaissance': ['renaissance', 'classical', 'old master'],
            'baroque': ['baroque', 'dramatic lighting', 'chiaroscuro'],
            'impressionist': ['impressionist', 'impressionism', 'plein air'],
            'expressionist': ['expressionist', 'expressionism', 'emotional'],
            'surrealist': ['surrealist', 'surrealism', 'dreamlike'],
            'cubist': ['cubist', 'cubism', 'geometric'],
            'abstract': ['abstract', 'non-representational'],
            'pop_art': ['pop art', 'andy warhol style', 'bright colors'],
            
            # Modern styles
            'photorealistic': ['photorealistic', 'hyperrealistic', 'photo-like'],
            'cinematic': ['cinematic', 'movie still', 'film photography'],
            'anime': ['anime', 'manga', 'japanese animation'],
            'cartoon': ['cartoon', 'animated', 'stylized'],
            'sketch': ['sketch', 'pencil drawing', 'hand drawn'],
            
            # Special effects
            'vintage': ['vintage', 'retro', 'old-fashioned'],
            'minimalist': ['minimalist', 'simple', 'clean'],
            'gothic': ['gothic', 'dark', 'medieval'],
            'steampunk': ['steampunk', 'victorian', 'mechanical']
        }
        
        # Style parameter mapping
        self.style_parameters = {
            'photorealistic': {
                'enhance_detail': True,
                'cfg_scale_boost': 0.2,
                'steps_boost': 20,
                'keywords': ['highly detailed', 'sharp focus', 'professional photography']
            },
            'oil_painting': {
                'enhance_texture': True,
                'cfg_scale_boost': 0.1,
                'keywords': ['oil on canvas', 'brushstrokes', 'painterly']
            },
            'watercolor': {
                'soften_edges': True,
                'keywords': ['watercolor painting', 'soft edges', 'fluid']
            },
            'digital_painting': {
                'enhance_color': True,
                'keywords': ['digital art', 'concept art', 'high resolution']
            },
            'renaissance': {
                'enhance_composition': True,
                'keywords': ['classical composition', 'chiaroscuro', 'masterpiece']
            },
            'cinematic': {
                'aspect_ratio': (16, 9),
                'enhance_drama': True,
                'keywords': ['cinematic lighting', 'dramatic', 'film still']
            },
            'anime': {
                'enhance_color': True,
                'keywords': ['anime style', 'cel shading', 'vibrant colors']
            }
        }
        
        # Style conflict detection
        self.style_conflicts = {
            'photorealistic': ['anime', 'cartoon', 'abstract'],
            'anime': ['photorealistic', 'renaissance', 'baroque'],
            'abstract': ['photorealistic', 'cinematic'],
            'sketch': ['photorealistic', 'digital_painting']
        }
    
    def extract_style_info(self, artwork_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract style info from artwork data"""
        style_info = {
            'style_name': '',
            'style_keywords': [],
            'style_parameters': {},
            'confidence': 0.0,
            'source': 'unknown'
        }
        
        # Extract style info from multiple sources
        style_sources = [
            ('style', artwork_data.get('style', '')),
            ('artistic_style', artwork_data.get('artistic_style', '')),
            ('environment.style', artwork_data.get('environment', {}).get('style', '')),
            ('metadata.style', artwork_data.get('metadata', {}).get('style', ''))
        ]
        
        best_style = None
        best_confidence = 0.0
        
        for source_name, style_value in style_sources:
            if not style_value:
                continue
            
            # Process string or list
            if isinstance(style_value, list):
                style_text = ', '.join(str(s) for s in style_value)
            else:
                style_text = str(style_value)
            
            # Recognize style
            recognized_style, confidence = self._recognize_style(style_text)
            
            if confidence > best_confidence:
                best_style = recognized_style
                best_confidence = confidence
                style_info['source'] = source_name
        
        if best_style:
            style_info['style_name'] = best_style
            style_info['confidence'] = best_confidence
            style_info['style_keywords'] = self.style_mapping.get(best_style, [])
            style_info['style_parameters'] = self.style_parameters.get(best_style, {})
        
        return style_info
    
    def _recognize_style(self, style_text: str) -> Tuple[str, float]:
        """Recognize style type in style text"""
        if not style_text:
            return '', 0.0
        
        style_text_lower = style_text.lower()
        best_match = ''
        best_score = 0.0
        
        for style_name, keywords in self.style_mapping.items():
            score = 0.0
            matches = 0
            
            for keyword in keywords:
                if keyword in style_text_lower:
                    # Higher score for exact match
                    if keyword == style_text_lower.strip():
                        score += 1.0
                    else:
                        score += 0.5
                    matches += 1
            
            # Calculate confidence
            if matches > 0:
                confidence = score / len(keywords)
                if confidence > best_score:
                    best_match = style_name
                    best_score = confidence
        
        return best_match, min(best_score, 1.0)
    
    def enhance_style_prompt(self, base_prompt: str, style_info: Dict[str, Any]) -> str:
        """Enhance prompt based on style info"""
        if not style_info or not style_info.get('style_name'):
            return base_prompt
        
        enhanced_prompt = base_prompt
        style_name = style_info['style_name']
        style_keywords = style_info.get('style_keywords', [])
        
        # Add style keywords
        if style_keywords:
            # Select most relevant keywords
            selected_keywords = style_keywords[:2]  # Max 2 keywords
            enhanced_prompt += f", {', '.join(selected_keywords)}"
        
        # Add specific style enhancement
        style_params = style_info.get('style_parameters', {})
        param_keywords = style_params.get('keywords', [])
        if param_keywords:
            selected_param_keywords = param_keywords[:2]
            enhanced_prompt += f", {', '.join(selected_param_keywords)}"
        
        return enhanced_prompt
    
    def get_generation_parameters(self, style_info: Dict[str, Any], 
                                base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust generation parameters based on style info"""
        if not style_info or not style_info.get('style_name'):
            return base_params
        
        adjusted_params = base_params.copy()
        style_name = style_info['style_name']
        style_params = style_info.get('style_parameters', {})
        
        # Adjust CFG scale
        if style_params.get('cfg_scale_boost'):
            current_cfg = adjusted_params.get('true_cfg_scale', 1.0)
            adjusted_params['true_cfg_scale'] = min(2.0, current_cfg + style_params['cfg_scale_boost'])
        
        # Adjust generation steps
        if style_params.get('steps_boost'):
            current_steps = adjusted_params.get('num_inference_steps', 80)
            adjusted_params['num_inference_steps'] = min(150, current_steps + style_params['steps_boost'])
        
        # Adjust aspect ratio
        if style_params.get('aspect_ratio'):
            aspect_w, aspect_h = style_params['aspect_ratio']
            base_size = max(adjusted_params.get('width', 1024), adjusted_params.get('height', 1024))
            
            if aspect_w > aspect_h:  # Landscape
                adjusted_params['width'] = base_size
                adjusted_params['height'] = int(base_size * aspect_h / aspect_w)
            else:  # Portrait
                adjusted_params['height'] = base_size
                adjusted_params['width'] = int(base_size * aspect_w / aspect_h)
        
        # Style specific adjustments
        if style_name == 'photorealistic':
            adjusted_params['num_inference_steps'] = max(100, adjusted_params.get('num_inference_steps', 80))
        elif style_name in ['sketch', 'cartoon']:
            adjusted_params['num_inference_steps'] = min(60, adjusted_params.get('num_inference_steps', 80))
        elif style_name == 'anime':
            adjusted_params['true_cfg_scale'] = max(1.2, adjusted_params.get('true_cfg_scale', 1.0))
        
        return adjusted_params
    

    
    def merge_style_info(self, primary_style: Dict[str, Any], 
                        secondary_style: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple style infos"""
        if not secondary_style:
            return primary_style
        
        if not primary_style:
            return secondary_style
        
        merged = primary_style.copy()
        
        # Merge keywords
        primary_keywords = set(merged.get('style_keywords', []))
        secondary_keywords = set(secondary_style.get('style_keywords', []))
        merged['style_keywords'] = list(primary_keywords | secondary_keywords)
        
        # Keep primary style name but lower confidence
        if secondary_style.get('confidence', 0) > merged.get('confidence', 0):
            merged['confidence'] = (merged.get('confidence', 0) + secondary_style.get('confidence', 0)) / 2
        
        return merged