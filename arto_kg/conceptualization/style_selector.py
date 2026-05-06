import random
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from arto_kg.conceptualization.utils import setup_logger, load_json
from arto_kg.config.model_config import ART_STYLES, STYLE_DATABASE


class StyleSelector:
    
    def __init__(self):
        self.logger = setup_logger("style_selector")
        self.style_database = STYLE_DATABASE
        self.available_styles = list(self.style_database.keys())
        self.logger.info(f"Initialized StyleSelector with {len(self.available_styles)} styles from config")
    
    def select_style(self, preference: Optional[str] = None, 
                    exclude_styles: Optional[List[str]] = None) -> str:
 
        # if preference is specified and valid, use it
        if preference and preference in self.available_styles:
            self.logger.info(f"Using specified style: {preference}")
            return preference

        # build candidate style list
        candidates = self.available_styles.copy()
        
        if exclude_styles:
            candidates = [style for style in candidates if style not in exclude_styles]
            
        if not candidates:
            self.logger.warning("No valid style candidates, using fallback")
            candidates = ["Abstract"]  # fallback

        # randomly select
        selected_style = random.choice(candidates)
        self.logger.info(f"Selected style: {selected_style}")
        
        return selected_style
    
    def get_style_characteristics(self, style_name: str) -> Dict[str, Any]:
        """
        Get style characteristics
        
        Args:
            style_name: Style name
            
        Returns:
            Style characteristics dictionary
        """
        if style_name in self.style_database:
            return self.style_database[style_name]
        else:
            self.logger.warning(f"Style '{style_name}' not found in database, returning default")
            return {
                "characteristics": ["artistic expression"],
                "color_tendencies": ["varied colors"],
                "typical_subjects": ["various subjects"],
                "techniques": ["artistic techniques"]
            }
    
    def get_style_color_tendencies(self, style_name: str) -> List[str]:
        """Get style color tendencies"""
        characteristics = self.get_style_characteristics(style_name)
        return characteristics.get("color_tendencies", ["varied colors"])
    
    def get_style_techniques(self, style_name: str) -> List[str]:
        """Get style techniques"""
        characteristics = self.get_style_characteristics(style_name)
        return characteristics.get("techniques", ["artistic techniques"])
    
    def validate_style_object_compatibility(self, style: str, objects: List[str]) -> Dict[str, Any]:
        """
        Validate style and object compatibility
        Some styles may be better suited for specific object types
        
        Args:
            style: Style name
            objects: List of object names
            
        Returns:
            Compatibility assessment result
        """
        style_info = self.get_style_characteristics(style)
        typical_subjects = style_info.get("typical_subjects", [])
        
        # Simple compatibility check logic
        compatibility_score = 0.5  # Default medium compatibility
        recommendations = []
        
        # Check match between object types and style typical subjects
        if any("portrait" in subject for subject in typical_subjects):
            if "person" in objects:
                compatibility_score += 0.2
                recommendations.append("Portrait subjects match this style well")
                
        if any("landscape" in subject for subject in typical_subjects):
            nature_objects = ["tree", "mountain", "water", "sky", "bird", "flower"]
            if any(obj for obj in objects if any(nature in obj.lower() for nature in nature_objects)):
                compatibility_score += 0.2
                recommendations.append("Nature elements suit this style")
                
        if any("still life" in subject for subject in typical_subjects):
            still_life_objects = ["bottle", "cup", "bowl", "book", "vase", "fruit"]
            if any(obj for obj in objects if any(still in obj.lower() for still in still_life_objects)):
                compatibility_score += 0.2
                recommendations.append("Still life objects match this style")
        
        # Limit score range
        compatibility_score = min(1.0, max(0.1, compatibility_score))
        
        return {
            "style": style,
            "objects": objects,
            "compatibility_score": compatibility_score,
            "recommendations": recommendations,
            "style_characteristics": style_info
        }
    
    def get_style_prompt_template(self, style: str) -> str:
        """
        Get style-specific prompt template
        
        Args:
            style: Style name
            
        Returns:
            Style-specific prompt segment
        """
        style_info = self.get_style_characteristics(style)
        
        characteristics = ", ".join(style_info.get("characteristics", []))
        techniques = ", ".join(style_info.get("techniques", []))
        colors = ", ".join(style_info.get("color_tendencies", []))
        
        template = f"in {style} style, featuring {characteristics}"
        if techniques:
            template += f", using {techniques}"
        if colors:
            template += f", with {colors}"
            
        return template
    
    def get_available_styles(self) -> List[str]:
        """Get all available styles"""
        return self.available_styles.copy()
    
    def get_style_count(self) -> int:
        """Get number of available styles"""
        return len(self.available_styles)
    
    def is_valid_style(self, style: str) -> bool:
        """Check if style is valid"""
        return style in self.available_styles
    
    def get_random_styles(self, count: int, exclude: Optional[List[str]] = None) -> List[str]:

        candidates = self.available_styles.copy()
        
        if exclude:
            candidates = [style for style in candidates if style not in exclude]
            
        if count >= len(candidates):
            return candidates
        
        return random.sample(candidates, count)
    
    
