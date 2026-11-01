"""Recognises the art style of an artwork specification and its generation hints."""

from typing import Dict, List, Any, Optional, Tuple

from arto_kg.config.model_config import STYLE_COCO_VARIETY


class StyleHandler:
    """Maps style text to a known style and its prompt keywords and parameters."""

    def __init__(self):
        # The styles Stage 1 generates, matched by name before any keyword.
        self.known_styles = {name.lower(): name for name in STYLE_COCO_VARIETY}


        # Style mapping and normalization
        self.style_mapping = {
            # Painting styles
            'oil_painting': ['oil painting', 'oil on canvas', 'traditional painting'],
            'watercolor': ['watercolor', 'watercolor painting', 'aquarelle'],
            'acrylic': ['acrylic painting', 'acrylic on canvas'],
            'digital_painting': ['digital art', 'digital painting', 'concept art'],
            
            # Art movements. An equal score goes to the earlier entry, so these
            # come before the movements whose keywords they contain.
            'neoclassical': ['neoclassical', 'neoclassicism', 'classical antiquity'],
            'post_impressionist': ['post-impressionist', 'post-impressionism', 'bold brushwork'],
            'chinese_ink': ['chinese ink painting', 'ink wash painting', 'xuan paper'],
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
    
    def extract_style_info(self, artwork_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return the best-matching style across the specification's style fields."""
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
            key = self._style_key(best_style)
            style_info['style_name'] = best_style
            style_info['confidence'] = best_confidence
            style_info['style_keywords'] = self.style_mapping.get(key, [])
            style_info['style_parameters'] = self.style_parameters.get(key, {})

        return style_info

    def _style_key(self, style: str) -> str:
        """The style_mapping key of a style: the key itself, else the entry that
        lists the style's name, so "Impressionism" resolves to 'impressionist'."""
        name = style.lower()
        if name in self.style_mapping:
            return name
        return next((key for key, keywords in self.style_mapping.items()
                     if name in keywords), name)

    def _recognize_style(self, style_text: str) -> Tuple[str, float]:
        """Return (style, confidence) for the best keyword match in style_text."""
        if not style_text:
            return '', 0.0
        
        # A known style name wins outright, so "Post-Impressionism" is not
        # read as Impressionism by the keyword match below.
        known = self.known_styles.get(style_text.strip().lower())
        if known:
            return known, 1.0

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
