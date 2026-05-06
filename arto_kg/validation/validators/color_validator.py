
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
from collections import defaultdict
import matplotlib.pyplot as plt


class ColorValidator:
    """Detailed Color Validator - Based on original color analysis logic"""
    
    def __init__(self, color_ontology_path: Optional[str] = None):
        """
        Initialize Color Validator with Two-Tier Color Resolution Strategy
        
        Tier 1: gt_color_mapping.json (project-specific, 19K+ entries)
        Tier 2: color_vocabulary.csv (standard colors, 741 entries)
        Tier 3: Default fallback mapping
        
        Args:
            color_ontology_path: Path to color ontology file (CSV format)
        """
        # Tier 1: Load GT Color Mapping (project-specific)
        self.gt_color_mapping = {}
        self._load_gt_color_mapping()
        
        # Tier 2: Load Color Vocabulary (standard colors)
        self.color_vocabulary = None
        self._load_color_vocabulary()
        
        # Legacy: Load color ontology if provided
        self.color_ontology = None
        if color_ontology_path and os.path.exists(color_ontology_path):
            self.load_color_ontology(color_ontology_path)
        
        # Tier 3: Default color mapping (final fallback)
        self.default_color_mapping = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (139, 69, 19),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            # Metallic colors
            'silver': (192, 192, 192),
            'gold': (255, 215, 0),
            'metallic': (169, 169, 169),
            # Other common colors
            'amber': (255, 191, 0),
            'beige': (245, 245, 220),
            'tan': (210, 180, 140),
            'maroon': (128, 0, 0),
            'navy': (0, 0, 128),
            'teal': (0, 128, 128),
            'olive': (128, 128, 0),
        }
    
    def _load_gt_color_mapping(self):
        """Load GT color mapping JSON file (Tier 1 - Project-specific)"""
        # File is in the same directory as color_validator.py
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "gt_color_mapping.json"),
            os.path.join(os.path.dirname(__file__), "..", "gt_color_mapping.json"),
            os.path.join(os.getcwd(), "gt_color_mapping.json"),
        ]
        
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                try:
                    with open(normalized_path, 'r') as f:
                        self.gt_color_mapping = json.load(f)
                    print(f"[ColorValidator] Loaded GT color mapping: {len(self.gt_color_mapping)} entries from {normalized_path}")
                    return
                except Exception as e:
                    print(f"[ColorValidator] Warning: Could not load GT color mapping from {normalized_path}: {e}")
        
        print("[ColorValidator] Warning: GT color mapping not found, using default mapping only")
    
    def _load_color_vocabulary(self):
        """Load color vocabulary CSV file (Tier 2 - Standard colors)"""
        # File is in the same directory as color_validator.py
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "color_vocabulary.csv"),
            os.path.join(os.path.dirname(__file__), "..", "color_vocabulary.csv"),
            os.path.join(os.getcwd(), "color_vocabulary.csv"),
        ]
        
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                try:
                    import pandas as pd
                    self.color_vocabulary = pd.read_csv(normalized_path)
                    print(f"[ColorValidator] Loaded color vocabulary: {len(self.color_vocabulary)} entries from {normalized_path}")
                    return
                except Exception as e:
                    print(f"[ColorValidator] Warning: Could not load color vocabulary from {normalized_path}: {e}")
        
        print("[ColorValidator] Warning: Color vocabulary not found, will use GT mapping and defaults only")
    
    def load_color_ontology(self, ontology_path: str):
        """Load color ontology"""
        try:
            import pandas as pd
            self.color_ontology = pd.read_csv(ontology_path)
            print(f"[ColorValidator] Loaded color ontology with {len(self.color_ontology)} entries")
        except Exception as e:
            print(f"[ColorValidator] Warning: Could not load color ontology: {e}")
            print("Will use default color mapping")
    
    def extract_dominant_colors(self, image_path: str, k: int = 12) -> List[Tuple[int, int, int]]:
       
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Reshape to pixel array
            pixels = image_rgb.reshape(-1, 3)
            
            # Use K-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get cluster centers
            centers = kmeans.cluster_centers_.astype(int)
            
            # Calculate weight for each cluster
            labels = kmeans.labels_
            weights = np.bincount(labels) / len(labels)
            
            # Sort by weight
            sorted_indices = np.argsort(weights)[::-1]
            dominant_colors = [tuple(centers[i]) for i in sorted_indices]
            
            return dominant_colors
            
        except Exception as e:
            print(f"❌ Error extracting colors from {image_path}: {e}")
            return []
    
    def extract_expected_colors(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract expected color information from JSON data - includes object-level and overall-level colors"""
        color_expectations = {
            'object_colors': {},  # Color expectations for each object
            'overall_colors': {'primary': [], 'secondary': [], 'accent': []},  # Overall color expectations
        }

        try:
            # 1. Extract object-level color information
            enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])

            for obj in enhanced_objects:
                if isinstance(obj, dict):
                    obj_name = obj.get('name', '')
                    if not obj_name:
                        continue

                    obj_colors = {'primary': [], 'secondary': []}

                    # Extract object colors from multiple possible fields
                    has_specific_colors = False
                    
                    # Method 1: Extract from color_palette (new format)
                    color_palette = obj.get('color_palette', {})
                    if isinstance(color_palette, dict):
                        # Process primary_colors
                        primary_colors = color_palette.get('primary_colors', [])
                        if isinstance(primary_colors, list):
                            specific_primary = [self._normalize_color_name(c) for c in primary_colors
                                              if c and c != "natural color accuracy"]
                            if specific_primary:
                                obj_colors['primary'] = specific_primary
                                has_specific_colors = True

                        # Process color_variations (as secondary)
                        color_variations = color_palette.get('color_variations', [])
                        if isinstance(color_variations, list):
                            specific_variations = [self._normalize_color_name(c) for c in color_variations
                                                 if c and c != "light and shadow variations"]
                            if specific_variations:
                                obj_colors['secondary'] = specific_variations
                                has_specific_colors = True
                    
                    # Method 2: Extract from direct primary_colors field (old format)
                    if not has_specific_colors:
                        primary_colors = obj.get('primary_colors', [])
                        if isinstance(primary_colors, list):
                            specific_primary = [self._normalize_color_name(c) for c in primary_colors
                                              if c and c != "natural color accuracy"]
                            if specific_primary:
                                obj_colors['primary'] = specific_primary
                                has_specific_colors = True
                    
                    # Method 3: Extract from colors field (fallback format)
                    if not has_specific_colors:
                        colors = obj.get('colors', [])
                        if isinstance(colors, list):
                            specific_colors = [self._normalize_color_name(c) for c in colors
                                             if c and c != "natural color accuracy"]
                            if specific_colors:
                                obj_colors['primary'] = specific_colors[:3]  # Take top 3 as primary
                                if len(specific_colors) > 3:
                                    obj_colors['secondary'] = specific_colors[3:5]  # Take 4th-5th as secondary
                                has_specific_colors = True

                    # If object has no specific color info, use artistic_expression colors as fallback
                    if not has_specific_colors:
                        print(f"   ⚠️  Object '{obj_name}' has no specific colors, using artistic_expression colors")
                        # Store first, allow filling from artistic_expression later
                        color_expectations['object_colors'][obj_name] = obj_colors
                    else:
                        color_expectations['object_colors'][obj_name] = obj_colors

            # 2. Extract overall artistic expression color information
            # Method 1: Extract from artistic_expression (new format)
            artistic_expression = json_data.get('artistic_expression', {})
            if isinstance(artistic_expression, dict):
                art_color_palette = artistic_expression.get('color_palette', {})
                if isinstance(art_color_palette, dict):
                    # Overall primary colors
                    primary_colors = art_color_palette.get('primary_colors', [])
                    if isinstance(primary_colors, list):
                        color_expectations['overall_colors']['primary'] = [self._normalize_color_name(c) for c in primary_colors if c]

                    # Overall secondary colors
                    secondary_colors = art_color_palette.get('secondary_colors', [])
                    if isinstance(secondary_colors, list):
                        color_expectations['overall_colors']['secondary'] = [self._normalize_color_name(c) for c in secondary_colors if c]

                    # Overall accent colors
                    accent_colors = art_color_palette.get('accent_colors', [])
                    if isinstance(accent_colors, list):
                        color_expectations['overall_colors']['accent'] = [self._normalize_color_name(c) for c in accent_colors if c]
            
            # Method 2: Extract from environment.color_scheme (old format)
            if not any(color_expectations['overall_colors'].values()):  # If no color info yet
                environment = json_data.get('environment', {})
                if isinstance(environment, dict):
                    color_scheme = environment.get('color_scheme', {})
                    if isinstance(color_scheme, dict):
                        main_palette = color_scheme.get('main_palette', {})
                        if isinstance(main_palette, dict):
                            # Overall primary colors
                            primary_colors = main_palette.get('primary_colors', [])
                            if isinstance(primary_colors, list):
                                # Convert hex colors to color names
                                primary_names = []
                                for color in primary_colors:
                                    if isinstance(color, str) and color.startswith('#'):
                                        # For hex colors, keep as is for now, process later
                                        primary_names.append(color)
                                    else:
                                        primary_names.append(self._normalize_color_name(str(color)))
                                color_expectations['overall_colors']['primary'] = primary_names

                            # Overall secondary colors
                            secondary_colors = main_palette.get('secondary_colors', [])
                            if isinstance(secondary_colors, list):
                                secondary_names = []
                                for color in secondary_colors:
                                    if isinstance(color, str) and color.startswith('#'):
                                        secondary_names.append(color)
                                    else:
                                        secondary_names.append(self._normalize_color_name(str(color)))
                                color_expectations['overall_colors']['secondary'] = secondary_names

                            # Overall accent colors
                            accent_colors = main_palette.get('accent_colors', [])
                            if isinstance(accent_colors, list):
                                accent_names = []
                                for color in accent_colors:
                                    if isinstance(color, str) and color.startswith('#'):
                                        accent_names.append(color)
                                    else:
                                        accent_names.append(self._normalize_color_name(str(color)))
                                color_expectations['overall_colors']['accent'] = accent_names

            # 3. Fill overall colors for objects without specific colors
            overall_primary = color_expectations['overall_colors']['primary']
            overall_secondary = color_expectations['overall_colors']['secondary']
            overall_accent = color_expectations['overall_colors']['accent']

            # Ensure list type
            if not isinstance(overall_primary, list):
                overall_primary = []
            if not isinstance(overall_secondary, list):
                overall_secondary = []
            if not isinstance(overall_accent, list):
                overall_accent = []

            if overall_primary or overall_secondary or overall_accent:
                for obj_name, obj_colors in color_expectations['object_colors'].items():
                    if not obj_colors['primary'] and not obj_colors['secondary']:
                        print(f"   🎨 Assigning overall colors to '{obj_name}': {overall_primary + overall_secondary + overall_accent}")
                        # Merge all colors and assign
                        all_overall_colors = overall_primary + overall_secondary + overall_accent
                        obj_colors['primary'] = all_overall_colors[:3] if len(all_overall_colors) >= 3 else all_overall_colors
                        if len(all_overall_colors) > 3:
                            obj_colors['secondary'] = all_overall_colors[3:5] if len(all_overall_colors) >= 5 else all_overall_colors[3:]

        except Exception as e:
            print(f"⚠️  Warning: Error extracting expected colors: {e}")

        return color_expectations

    def _normalize_color_name(self, color_name: str) -> str:
        """Normalize color name"""
        if not color_name or not isinstance(color_name, str):
            return ""

        color_lower = color_name.lower().strip()
        
        # If hex color code, return directly
        if color_lower.startswith('#') and len(color_lower) in [4, 7]:
            return color_lower

        # Map complex color descriptions to basic colors
        color_mapping = {
            'metallic silver': 'silver',
            'charcoal black': 'black',
            'pure white': 'white',
            'soft blue': 'blue',
            'warm amber': 'orange',
            'stainless steel': 'silver',
            'natural color accuracy': '',  # Ignore this generic description
            'light and shadow variations': '',  # Ignore lighting variation description
        }

        normalized = color_mapping.get(color_lower, color_lower)
        return normalized

    def evaluate_overall_color_accuracy(self, extracted_colors: List[Tuple[int, int, int]],
                                       expected_colors: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate overall color accuracy"""
        # Merge all expected colors
        all_expected_colors = []
        for color_list in expected_colors.values():
            all_expected_colors.extend([c for c in color_list if c])  # Filter empty strings

        if not all_expected_colors:
            return {
                'score': 50.0,
                'details': 'No expected colors specified',
                'warning': True
            }

        # Use existing color matching logic
        legacy_expected = {'primary': all_expected_colors, 'secondary': [], 'accent': []}
        return self.evaluate_color_accuracy(extracted_colors, legacy_expected)



    def evaluate_object_colors(self, image_path: str,
                             bboxes_dict: Dict[str, List[int]],
                             object_color_expectations: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """Evaluate color accuracy for each object"""
        if not object_color_expectations:
            return {
                'average_score': 80.0,
                'object_results': {},
                'note': 'No object color expectations specified'
            }

        object_results = {}
        scores = []

        for obj_name, expected_colors in object_color_expectations.items():
            if obj_name not in bboxes_dict:
                continue

            # Extract colors from object region
            bbox = bboxes_dict[obj_name]
            obj_colors = self.extract_colors_from_region(image_path, bbox)

            if obj_colors:
                # Evaluate object color match
                all_expected = expected_colors['primary'] + expected_colors['secondary']
                if all_expected:
                    match_score = self.calculate_color_match_score(obj_colors, all_expected)
                    scores.append(match_score)

                    object_results[obj_name] = {
                        'score': match_score,
                        'expected_colors': expected_colors,
                        'extracted_colors': obj_colors[:5],  # Keep only top 5 dominant colors
                        'bbox': bbox
                    }

        average_score = sum(scores) / len(scores) if scores else 80.0

        return {
            'average_score': average_score,
            'object_results': object_results,
            'objects_analyzed': len(object_results)
        }

    def extract_colors_from_region(self, image_path: str, bbox: List[int]) -> List[Dict[str, Any]]:
        """Extract dominant colors from specified region of image - enhanced version with proportion and color name"""
        try:
            import cv2
            from sklearn.cluster import KMeans

            image = cv2.imread(image_path)
            if image is None:
                return []

            # Crop object region, ensure bbox coordinates are numeric and convert to int
            x1 = int(float(bbox[0]))
            y1 = int(float(bbox[1]))
            x2 = int(float(bbox[2]))
            y2 = int(float(bbox[3]))
            
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = int(x2)
            y2 = int(y2)

            if x2 <= x1 or y2 <= y1:
                return []

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                return []

            # Extract dominant colors using K-means
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pixels = roi_rgb.reshape(-1, 3)

            # Sample pixels to avoid slow computation
            if len(pixels) > 1000:
                indices = np.random.choice(len(pixels), 1000, replace=False)
                pixels = pixels[indices]

            n_clusters = min(5, len(pixels))
            if n_clusters > 0 and len(pixels) > 0:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(pixels)
                colors = kmeans.cluster_centers_.astype(int)
                labels = kmeans.labels_

                # Calculate proportion and name for each color
                color_info = []
                for i, color in enumerate(colors):
                    # Ensure labels is array type
                    if hasattr(labels, '__len__') and len(labels) > 0:
                        proportion = np.sum(labels == i) / len(labels)
                    else:
                        proportion = 1.0 / len(colors)  # Distribute evenly

                    color_name = self.rgb_to_color_name_from_vocabulary(tuple(color))
                    color_info.append({
                        'rgb': tuple(int(c) for c in color),  # Ensure integer
                        'proportion': round(proportion * 100, 1),
                        'color_name': color_name
                    })

                # Sort by proportion
                color_info.sort(key=lambda x: x['proportion'], reverse=True)
                return color_info

        except Exception as e:
            print(f"❌ Error extracting colors from region: {e}")
            return []

    def calculate_color_match_score(self, extracted_colors: List[Dict[str, Any]],
                                   expected_color_names: List[str]) -> float:
        """Calculate color match score - supports new color data structure"""
        if not expected_color_names:
            return 80.0

        total_similarity = 0.0
        valid_matches = 0

        for expected_name in expected_color_names:
            if not expected_name:  # Skip empty strings
                continue

            # Get RGB from color vocabulary for expected color
            expected_color_info = self.find_color_in_vocabulary(expected_name)
            if not expected_color_info or not expected_color_info.get('rgb'):
                # Fallback: Use default mapping
                expected_rgb = self.color_name_to_rgb(expected_name)
                if not expected_rgb:
                    continue
            else:
                expected_rgb = expected_color_info['rgb']

            # Find best match in extracted colors
            best_similarity = 0.0
            for extracted_color in extracted_colors:
                extracted_rgb = extracted_color['rgb']
                # Calculate similarity
                distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(extracted_rgb, expected_rgb)))
                max_distance = np.sqrt(3 * (255 ** 2))
                similarity = max(0, 100 * (1 - distance / max_distance))

                # Consider color proportion in region as weight
                weighted_similarity = similarity * (extracted_color['proportion'] / 100.0)
                best_similarity = max(best_similarity, weighted_similarity)

            total_similarity += best_similarity
            valid_matches += 1

        average_similarity = total_similarity / valid_matches if valid_matches > 0 else 80.0
        return min(100.0, average_similarity)

    def color_name_to_rgb(self, color_name: str) -> Optional[Tuple[int, int, int]]:
        """Convert color name to RGB"""
        return self.default_color_mapping.get(color_name.lower())

    def color_distance(self, color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between two colors"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))

    
    def _infer_colors_from_material(self, material: str) -> List[str]:
        """Infer possible colors from material"""
        material_lower = material.lower()
        colors = []
        
        material_color_mapping = {
            'fur': ['brown', 'black', 'white', 'gray'],
            'hair': ['brown', 'black', 'blond', 'gray'],
            'steel': ['silver', 'gray', 'metallic'],
            'wood': ['brown', 'tan', 'beige'],
            'metal': ['silver', 'gold', 'bronze', 'copper'],
            'glass': ['transparent', 'clear', 'blue'],
            'leather': ['brown', 'black', 'tan'],
            'fabric': ['various'],
            'stone': ['gray', 'white', 'brown'],
            'crystal': ['clear', 'white', 'prismatic']
        }
        
        for mat, mat_colors in material_color_mapping.items():
            if mat in material_lower:
                colors.extend(mat_colors)
        
        return colors
    
    def calculate_color_distance(self, color1: Tuple[int, int, int], 
                                color2: Tuple[int, int, int]) -> float:
      
        try:
            # Simplified Delta E calculation (Euclidean distance in RGB space)
            r1, g1, b1 = color1
            r2, g2, b2 = color2
            
            return np.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)
            
        except Exception as e:
            print(f"Warning: Error calculating color distance: {e}")
            return float('inf')
    
    def find_best_color_match(self, extracted_color: Tuple[int, int, int], 
                             expected_color_name: str) -> Dict[str, Any]:
        """
        Find best expected color match for extracted color
        
        Args:
            extracted_color: Extracted RGB color
            expected_color_name: Expected color name
            
        Returns:
            Match result dictionary
        """
        # Get RGB value for expected color
        expected_rgb = self.get_color_rgb(expected_color_name)
        if expected_rgb is None:
            return {
                'color_name': expected_color_name,
                'match_found': False,
                'error': f'Unknown color name: {expected_color_name}'
            }
        
        # Calculate color distance
        distance = self.calculate_color_distance(extracted_color, expected_rgb)
        
        # Calculate match score (based on distance)
        max_distance = 441.67  # sqrt(255^2 + 255^2 + 255^2)
        score = max(0, 100 * (1 - distance / max_distance))
        
        return {
            'color_name': expected_color_name,
            'expected_rgb': expected_rgb,
            'extracted_rgb': extracted_color,
            'distance': distance,
            'delta_e': distance,  # Simplified version
            'score': score,
            'match_found': True,
            'match_quality': 'excellent' if score >= 80 else 'good' if score >= 60 else 'fair' if score >= 40 else 'poor'
        }
    
    def get_color_rgb(self, color_name: str) -> Optional[Tuple[int, int, int]]:
        """
        Get RGB value by color name using Two-Tier Resolution Strategy
        
        Resolution order:
        1. Hex code (if starts with #)
        2. GT color mapping (project-specific, 19K+ entries)
        3. Color vocabulary (standard colors, 741 entries)
        4. Legacy color ontology (if provided)
        5. Default mapping (exact match)
        6. Fuzzy match in default mapping
        
        Args:
            color_name: Color name or hex code
            
        Returns:
            RGB tuple or None
        """
        original_name = color_name
        color_name = color_name.lower().strip()
        
        # 1. Check if hex color code
        if color_name.startswith('#') and len(color_name) in [4, 7]:
            try:
                rgb = self.hex_to_rgb(color_name)
                if rgb is not None:
                    return rgb
            except:
                pass
        
        # 2.  Check GT color mapping 
        if color_name in self.gt_color_mapping:
            mapping_info = self.gt_color_mapping[color_name]
            if 'hex' in mapping_info:
                return self.hex_to_rgb(mapping_info['hex'])
            elif 'rgb' in mapping_info:
                return tuple(mapping_info['rgb'])
        
        # 3. Check color vocabulary
        if self.color_vocabulary is not None:
            try:
                import pandas as pd
                matches = self.color_vocabulary[
                    self.color_vocabulary['color_name'].str.lower() == color_name
                ]
                if not matches.empty:
                    row = matches.iloc[0]
                    hex_value = row['hex_string']
                    return self.hex_to_rgb(hex_value)
            except Exception as e:
                pass  # Silently continue to next tier
        
        # 4. Check legacy ontology (if provided)
        if self.color_ontology is not None:
            try:
                import pandas as pd
                matches = self.color_ontology[
                    self.color_ontology['color_name'].str.lower() == color_name
                ]
                if not matches.empty:
                    row = matches.iloc[0]
                    return (int(row['red']), int(row['green']), int(row['blue']))
            except Exception as e:
                pass  # Silently continue to next tier
        
        # 5. Use default mapping (exact match)
        if color_name in self.default_color_mapping:
            return self.default_color_mapping[color_name]
        
        # 6. Fuzzy match in default mapping
        for key, rgb in self.default_color_mapping.items():
            if color_name in key or key in color_name:
                return rgb
        
        return None

    def load_color_vocabulary(self):
        """Load color vocabulary CSV"""
        if not hasattr(self, 'color_vocab') or self.color_vocab is None:
            try:
                import pandas as pd
                self.color_vocab = pd.read_csv('color_vocabulary.csv')
                print(f"✅ Loaded color vocabulary: {len(self.color_vocab)} colors")
            except Exception as e:
                print(f"⚠️  Warning: Could not load color vocabulary: {e}")
                self.color_vocab = pd.DataFrame()
        return self.color_vocab

    def find_color_in_vocabulary(self, color_name):
        """Find RGB value for color in vocabulary"""
        # If hex color code, convert directly
        if isinstance(color_name, str) and color_name.startswith('#') and len(color_name) in [4, 7]:
            rgb = self.hex_to_rgb(color_name)
            if rgb is not None:
                return {
                    'rgb': rgb,
                    'hex': color_name,
                    'hue': 0,
                    'saturation': 0,
                    'brightness': 0,
                    'temperature': 'Unknown'
                }
        
        color_vocab = self.load_color_vocabulary()
        if color_vocab.empty:
            return None

        try:
            # Case-insensitive match
            matches = color_vocab[color_vocab['color_name'].str.lower() == color_name.lower()]
            if not matches.empty:
                hex_value = matches.iloc[0]['hex_string']
                rgb = self.hex_to_rgb(hex_value)
                return {
                    'rgb': rgb,
                    'hex': hex_value,
                    'hue': matches.iloc[0].get('hue', 0),
                    'saturation': matches.iloc[0].get('saturation', 0),
                    'brightness': matches.iloc[0].get('brightness', 0),
                    'temperature': matches.iloc[0].get('temperature', 'Unknown')
                }
        except Exception as e:
            print(f"⚠️  Error finding color {color_name}: {e}")

        return None

    def hex_to_rgb(self, hex_string):
        """Convert hex string to RGB"""
        try:
            hex_string = hex_string.lstrip('#')
            if len(hex_string) == 6:
                return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
            else:
                return None
        except:
            return None

    def rgb_to_color_name_from_vocabulary(self, rgb_values):
        """Convert RGB values to closest color name using vocabulary"""
        color_vocab = self.load_color_vocabulary()
        if color_vocab.empty:
            return 'unknown'

        try:
            min_distance = float('inf')
            closest_color = 'unknown'

            for _, row in color_vocab.iterrows():
                vocab_rgb = self.hex_to_rgb(row['hex_string'])
                if vocab_rgb:
                    # Calculate Delta E (simplified version)
                    distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb_values, vocab_rgb)))
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = row['color_name']

            return closest_color
        except Exception as e:
            print(f"⚠️  Error converting RGB to color name: {e}")
            return 'unknown'
    
    def evaluate_color_accuracy(self, extracted_colors: List[Tuple[int, int, int]], 
                               expected_colors: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Evaluate color accuracy
        
        Args:
            extracted_colors: List of extracted dominant colors
            expected_colors: Dictionary of expected colors
            
        Returns:
            Color accuracy evaluation result
        """
        if not extracted_colors:
            return {
                'score': 0.0,
                'details': 'No colors extracted from image',
                'error': True
            }
        
        all_expected = []
        importance_weights = {'primary': 1.0, 'secondary': 0.7, 'accent': 0.5}
        
        # Collect all expected colors and their importance
        color_matches = []
        total_weight = 0
        
        for importance, colors in expected_colors.items():
            weight = importance_weights.get(importance, 0.5)
            for color_name in colors:
                if color_name:  # Ensure color name is not empty
                    all_expected.append((color_name, importance, weight))
                    total_weight += weight
        
        if not all_expected:
            return {
                'score': 50.0,  # Neutral score
                'details': 'No expected colors specified',
                'warning': True
            }
        
        # For each expected color, find the best match
        matched_scores = []
        detailed_results = defaultdict(list)
        
        for expected_color, importance, weight in all_expected:
            best_match = None
            best_score = 0
            
            # Find best match in all extracted colors
            for extracted_color in extracted_colors:
                match_result = self.find_best_color_match(extracted_color, expected_color)
                if match_result.get('match_found') and match_result['score'] > best_score:
                    best_score = match_result['score']
                    best_match = match_result
            
            if best_match:
                weighted_score = best_score * weight
                matched_scores.append(weighted_score)
                detailed_results[importance].append(best_match)
            else:
                # No match found, assign 0 score
                matched_scores.append(0.0)
                detailed_results[importance].append({
                    'color_name': expected_color,
                    'score': 0,
                    'error': f'No match found for {expected_color}'
                })
        
        # Calculate weighted average score
        if total_weight > 0:
            final_score = sum(matched_scores) / total_weight
        else:
            final_score = 0
        
        return {
            'score': final_score,
            'detailed_results': dict(detailed_results),
            'total_expected_colors': len(all_expected),
            'total_extracted_colors': len(extracted_colors),
            'color_coverage': len([s for s in matched_scores if s > 40]) / len(all_expected) if all_expected else 0
        }
    
    def analyze_image_characteristics(self, image_path: str) -> Dict[str, Any]:
        """Analyze image characteristics (brightness, contrast, saturation)"""
        """
        Analyze the overall color characteristics of the image
        
        Args:
            image_path: Image path
            
        Returns:
            Image characteristic analysis results
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not read image'}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to HSV for analysis
            image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Analyze brightness
            brightness = np.mean(image_hsv[:, :, 2])
            brightness_category = 'bright' if brightness > 170 else 'medium' if brightness > 85 else 'dark'
            
            # Calculate saturation and brightness variance
            saturation = np.mean(image_hsv[:, :, 1])
            saturation_category = 'vivid' if saturation > 170 else 'moderate' if saturation > 85 else 'muted'
            
            # Analyze hue distribution
            hue_values = image_hsv[:, :, 0].flatten()
            hue_histogram = np.histogram(hue_values, bins=12)[0]
            dominant_hue_range = np.argmax(hue_histogram)
            
            # Hue temperature analysis
            warm_hues = np.sum(hue_histogram[0:3]) + np.sum(hue_histogram[9:12])  # Red, orange, yellow
            cool_hues = np.sum(hue_histogram[3:9])  # Green, cyan, blue, purple
            
            if warm_hues > cool_hues * 1.2:
                temperature = 'warm'
            elif cool_hues > warm_hues * 1.2:
                temperature = 'cool'
            else:
                temperature = 'neutral'
            
            return {
                'brightness': brightness,
                'brightness_category': brightness_category,
                'saturation': saturation,
                'saturation_category': saturation_category,
                'dominant_temperature': temperature,
                'hue_distribution': hue_histogram.tolist(),
                'dominant_hue_range': int(dominant_hue_range)
            }
            
        except Exception as e:
            return {'error': f'Error analyzing image characteristics: {e}'}
    
    def evaluate_color_harmony(self, extracted_colors: List[Tuple[int, int, int]]) -> Dict[str, Any]:
        """Evaluate color harmony"""
        """
        Evaluate color harmony
        
        Args:
            extracted_colors: List of extracted colors
            
        Returns:
            Color harmony evaluation result
        """
        if len(extracted_colors) < 2:
            return {'score': 50.0, 'details': 'Insufficient colors for harmony analysis'}
        
        try:
            # Convert to HSV for analysis
            hsv_colors = []
            for r, g, b in extracted_colors[:8]:  # Analyze the top 8 main colors
                # Normalize to 0-1 range
                rgb_normalized = np.array([r, g, b]) / 255.0
                # Convert to HSV
                hsv = cv2.cvtColor(np.array([[rgb_normalized]], dtype=np.float32), cv2.COLOR_RGB2HSV)[0, 0]
                hsv_colors.append(hsv)
            
            # Analyze hue relationships
            hues = [hsv[0] for hsv in hsv_colors if hsv[1] > 0.1]  # Exclude gray
            
            if len(hues) < 2:
                return {'score': 60.0, 'details': 'Mostly achromatic colors'}
            
            # Calculate hue distances
            hue_distances = []
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    # Handle circular nature of hues
                    dist = min(abs(hues[i] - hues[j]), 360 - abs(hues[i] - hues[j]))
                    hue_distances.append(dist)
            
            # Score based on harmony type
            harmony_score = 0
            harmony_reasons = []
            
            # Analogous colors (adjacent hues)
            adjacent_pairs = sum(1 for d in hue_distances if 15 <= d <= 45)
            if adjacent_pairs > 0:
                harmony_score += 30
                harmony_reasons.append(f"Adjacent colors ({adjacent_pairs} pairs)")
            
            # Complementary colors
            complementary_pairs = sum(1 for d in hue_distances if 150 <= d <= 210)
            if complementary_pairs > 0:
                harmony_score += 25
                harmony_reasons.append(f"Complementary colors ({complementary_pairs} pairs)")
            
            # Triadic colors
            triadic_groups = sum(1 for d in hue_distances if 100 <= d <= 140)
            if triadic_groups > 0:
                harmony_score += 20
                harmony_reasons.append(f"Triadic relationships ({triadic_groups} groups)")
            
            # Monochromatic harmony (different lightness/saturation of the same hue)
            monochromatic = sum(1 for d in hue_distances if d <= 15)
            if monochromatic > len(hue_distances) * 0.5:
                harmony_score += 15
                harmony_reasons.append("Monochromatic harmony")
            
            # Saturation consistency
            saturations = [hsv[1] for hsv in hsv_colors]
            saturation_std = np.std(saturations)
            if saturation_std < 0.3:
                harmony_score += 10
                harmony_reasons.append("Consistent saturation")
            
            # Ensure score is within a reasonable range
            harmony_score = min(100, max(0, harmony_score))
            
            return {
                'score': harmony_score,
                'reasons': harmony_reasons,
                'hue_analysis': {
                    'unique_hues': len(hues),
                    'adjacent_pairs': adjacent_pairs,
                    'complementary_pairs': complementary_pairs,
                    'triadic_groups': triadic_groups,
                    'monochromatic_ratio': monochromatic / len(hue_distances) if hue_distances else 0
                },
                'saturation_consistency': float(1.0 - min(1.0, saturation_std))
            }
            
        except Exception as e:
            return {'score': 50.0, 'error': f'Error in harmony analysis: {e}'}
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str,
                       bboxes_dict: Optional[Dict[str, List[int]]] = None,
                       visualize: bool = False) -> Dict[str, Any]:
        """Comprehensive color evaluation - Main entry point"""
        # If bboxes are provided, perform object-based color analysis
        if bboxes_dict:
            return self.comprehensive_color_evaluation_with_objects(json_data, image_path, bboxes_dict, visualize=visualize)
        else:
            return self.comprehensive_color_evaluation(json_data, image_path, visualize=visualize)
    
    def comprehensive_color_evaluation(self, json_data: Dict[str, Any],
                                     image_path: str,
                                     visualize: bool = False) -> Dict[str, Any]:
        """
        Comprehensive color evaluation - Main entry point (overall analysis only)

        Args:
            json_data: Artwork JSON data
            image_path: Image file path
            visualize: Whether to generate visualization
            
        Returns:
            Complete color evaluation results
        """
        print(f"🎨 Starting comprehensive color evaluation for: {os.path.basename(image_path)}")
        
        try:
            # 1. Extract dominant colors
            print("   Step 1: Extracting dominant colors...")
            extracted_colors = self.extract_dominant_colors(image_path)
            if not extracted_colors:
                return {'error': 'Failed to extract colors from image'}
            
            print(f"   ✅ Extracted {len(extracted_colors)} dominant colors")
            
            # 2. Extract expected colors from JSON - new two-layer structure
            print("   Step 2: Extracting expected colors from JSON...")
            color_expectations = self.extract_expected_colors(json_data)

            # Count color expectations
            object_colors_count = sum(len(colors['primary']) + len(colors['secondary'])
                                    for colors in color_expectations['object_colors'].values())
            overall_colors_count = sum(len(colors) for colors in color_expectations['overall_colors'].values())

            print(f"   ✅ Found {object_colors_count} object-level colors, {overall_colors_count} overall colors")

            # 3. Evaluate overall color accuracy
            print("   Step 3: Evaluating overall color accuracy...")
            accuracy_result = self.evaluate_overall_color_accuracy(extracted_colors, color_expectations['overall_colors'])
            
            # 4. Calculate overall score (using only color accuracy)
            print("   Step 4: Calculating overall color score...")
            
            # Weight distribution (using only color accuracy)
            overall_score = accuracy_result.get('score', 0)
            
            # 5. Compile final results
            evaluation_result = {
                'overall_score': overall_score,
                'dimension_scores': {
                    'color_accuracy': accuracy_result.get('score', 0)
                },
                'detailed_results': {
                    'accuracy': accuracy_result,
                    'extracted_colors': extracted_colors,
                    'expected_colors': color_expectations
                },
                'evaluation_summary': {
                    'total_colors_extracted': len(extracted_colors),
                    'total_colors_expected': object_colors_count + overall_colors_count,
                    'color_coverage': accuracy_result.get('color_coverage', 0)
                },
                'image_file': image_path,
                'artwork_id': json_data.get('artwork_id', 'unknown')
            }
            
            if visualize:
                output_dir = os.path.join(os.path.dirname(image_path), "color_analysis")
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"color_analysis_{os.path.basename(image_path)}")
                self.create_color_visualization(evaluation_result, save_path)
                evaluation_result['visualization_path'] = save_path

            print(f"   ✅ Color evaluation completed! Overall score: {overall_score:.1f}/100")
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Comprehensive color evaluation failed: {e}"
            print(f"   ❌ {error_msg}")
            return {'error': error_msg}
    
    def comprehensive_color_evaluation_with_objects(self, json_data: Dict[str, Any],
                                                   image_path: str,
                                                   bboxes_dict: Dict[str, List[int]],
                                                   visualize: bool = False) -> Dict[str, Any]:
        """Comprehensive color evaluation based on detected objects"""
        try:
            print(f"   Starting object-based color evaluation...")
            
            # 1. Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Could not load image: {image_path}'}
            
            # 2. Global color analysis
            # Pass visualize=visualize to control visualization generation
            global_result = self.comprehensive_color_evaluation(json_data, image_path, visualize=visualize)
            if 'error' in global_result:
                return global_result
            
            # 3. Object-based color analysis
            object_color_analysis = {}
            
            for obj_name, bbox in bboxes_dict.items():
                print(f"     Analyzing colors for object: {obj_name}")
                
                # Extract object region colors
                if len(bbox) >= 4:
                    # Ensure bbox coordinates are numeric and convert to int
                    x1 = int(float(bbox[0]))
                    y1 = int(float(bbox[1]))
                    x2 = int(float(bbox[2]))
                    y2 = int(float(bbox[3]))
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, image.shape[1] - 1))
                    y1 = max(0, min(y1, image.shape[0] - 1))
                    x2 = max(x1 + 1, min(x2, image.shape[1]))  # Ensure x2 > x1
                    y2 = max(y1 + 1, min(y2, image.shape[0]))  # Ensure y2 > y1

                    obj_region = image[y1:y2, x1:x2]
                    
                    if obj_region.size > 0:
                        # Save object region to a temporary file for color extraction
                        import tempfile
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            cv2.imwrite(tmp_file.name, obj_region)
                            obj_colors = self.extract_dominant_colors(tmp_file.name, k=3)
                            os.unlink(tmp_file.name)  # Delete temporary file
                        
                        # Analyze object expected colors
                        expected_colors = self._extract_object_expected_colors(json_data, obj_name)
                        print(f"     Debug: Expected colors for {obj_name}: {expected_colors}")
                        
                        # If no specific colors are extracted for the object, use overall colors as fallback
                        if not expected_colors:
                            print(f"     Debug: No specific colors found for {obj_name}, trying fallback methods")
                            # Get from overall color expectations
                            color_expectations = self.extract_expected_colors(json_data)
                            if 'object_colors' in color_expectations and obj_name in color_expectations['object_colors']:
                                obj_color_info = color_expectations['object_colors'][obj_name]
                                # Merge primary and secondary colors
                                expected_colors = obj_color_info.get('primary', []) + obj_color_info.get('secondary', [])
                                print(f"     Debug: Colors from object_colors: {expected_colors}")
                            # If still no colors, use overall artistic expression colors
                            if not expected_colors and 'overall_colors' in color_expectations:
                                overall_colors = color_expectations['overall_colors']
                                expected_colors = overall_colors.get('primary', []) + overall_colors.get('secondary', [])
                                print(f"     Debug: Colors from overall_colors: {expected_colors}")
                        
                        # Calculate match score
                        if expected_colors and obj_colors:
                            color_match = self._calculate_object_color_match(obj_colors, expected_colors)
                        else:
                            color_match = {'match_score': 70.0, 'note': 'No color expectations for this object'}
                        
                        object_color_analysis[obj_name] = {
                            'dominant_colors': obj_colors,
                            'expected_colors': expected_colors,
                            'color_match': color_match,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]  # Ensure bbox is an integer list
                        }
            
            # 4. Integrate results (using only object color accuracy and overall color accuracy)
            if object_color_analysis:
                object_scores = [obj['color_match']['match_score'] for obj in object_color_analysis.values()]
                object_based_score = np.mean(object_scores)
                
                # Combine global and object scores (50% global, 50% object)
                final_score = global_result['overall_score'] * 0.5 + object_based_score * 0.5
            else:
                final_score = global_result['overall_score']
                object_based_score = global_result['overall_score']  # If no object analysis, use overall score
            
            # Update results
            enhanced_result = global_result.copy()
            enhanced_result['overall_score'] = final_score
            enhanced_result['object_color_analysis'] = object_color_analysis
            enhanced_result['dimension_scores']['object_color_accuracy'] = object_based_score
            enhanced_result['evaluation_summary']['objects_analyzed'] = len(object_color_analysis)
            
            print(f"   ✅ Object-based color evaluation completed! Final score: {final_score:.1f}/100")
            return enhanced_result
            
        except Exception as e:
            error_msg = f"Object-based color evaluation failed: {e}"
            print(f"   ❌ {error_msg}")
            return {'error': error_msg}
    
    def _extract_object_expected_colors(self, json_data: Dict[str, Any], obj_name: str) -> List[str]:
        """Extract expected colors for object"""
        expected_colors = []
        
        try:
            # Method 1: Extract from objects.enhanced_objects
            objects_data = json_data.get('objects', {})
            if isinstance(objects_data, dict):
                enhanced_objects = objects_data.get('enhanced_objects', [])
                for obj in enhanced_objects:
                    if isinstance(obj, dict):
                        obj_name_json = obj.get('name', '').lower()
                        if obj_name_json == obj_name.lower() or obj_name.lower() in obj_name_json or obj_name_json in obj_name.lower():
                            # Extract colors from color_palette
                            color_palette = obj.get('color_palette', {})
                            if isinstance(color_palette, dict):
                                # Primary colors
                                primary_colors = color_palette.get('primary_colors', [])
                                if isinstance(primary_colors, list):
                                    normalized_colors = [self._normalize_color_name(c) for c in primary_colors if c]
                                    expected_colors.extend([color for color in normalized_colors if color])  # Filter empty strings
                                elif isinstance(primary_colors, str) and primary_colors:
                                    normalized_color = self._normalize_color_name(primary_colors)
                                    if normalized_color:  # Add only non-empty colors
                                        expected_colors.append(normalized_color)
                                
                                # Secondary colors
                                secondary_colors = color_palette.get('secondary_colors', [])
                                if isinstance(secondary_colors, list):
                                    normalized_colors = [self._normalize_color_name(c) for c in secondary_colors if c]
                                    expected_colors.extend([color for color in normalized_colors if color])  # Filter empty strings
                                elif isinstance(secondary_colors, str) and secondary_colors:
                                    normalized_color = self._normalize_color_name(secondary_colors)
                                    if normalized_color:  # Add only non-empty colors
                                        expected_colors.append(normalized_color)
                                
                                # Accent colors
                                accent_colors = color_palette.get('accent_colors', [])
                                if isinstance(accent_colors, list):
                                    normalized_colors = [self._normalize_color_name(c) for c in accent_colors if c]
                                    expected_colors.extend([color for color in normalized_colors if color])  # Filter empty strings
                                elif isinstance(accent_colors, str) and accent_colors:
                                    normalized_color = self._normalize_color_name(accent_colors)
                                    if normalized_color:  # Add only non-empty colors
                                        expected_colors.append(normalized_color)
                            
                            # Extract from color field (if exists)
                            if 'color' in obj:
                                color = obj['color']
                                if isinstance(color, list):
                                    normalized_colors = [self._normalize_color_name(str(c)) for c in color if c]
                                    expected_colors.extend([color for color in normalized_colors if color])  # Filter empty strings
                                elif isinstance(color, str) and color:
                                    normalized_color = self._normalize_color_name(color)
                                    if normalized_color:  # Add only non-empty colors
                                        expected_colors.append(normalized_color)
            
            # Method 2: Extract overall colors from artistic_expression as fallback
            if not expected_colors:
                artistic_expression = json_data.get('artistic_expression', {})
                if isinstance(artistic_expression, dict):
                    art_color_palette = artistic_expression.get('color_palette', {})
                    if isinstance(art_color_palette, dict):
                        # Overall primary colors
                        primary_colors = art_color_palette.get('primary_colors', [])
                        if isinstance(primary_colors, list):
                            normalized_colors = [self._normalize_color_name(c) for c in primary_colors if c]
                            expected_colors.extend([color for color in normalized_colors if color])  # Filter empty strings
                        
                        # Overall secondary colors
                        secondary_colors = art_color_palette.get('secondary_colors', [])
                        if isinstance(secondary_colors, list):
                            normalized_colors = [self._normalize_color_name(c) for c in secondary_colors if c]
                            expected_colors.extend([color for color in normalized_colors if color])  # Filter empty strings
            
            # Deduplicate and filter empty strings
            expected_colors = [color for color in list(set(expected_colors)) if color]
            
        except Exception as e:
            print(f"Warning: Error extracting object colors for {obj_name}: {e}")
        
        return expected_colors
    
    def _calculate_object_color_match(self, actual_colors: List[Tuple[int, int, int]], expected_colors: List[str]) -> Dict[str, Any]:
        """Calculate object color match score"""
        if not expected_colors or not actual_colors:
            return {'match_score': 50.0, 'note': 'Insufficient color data'}
        
        try:
            total_matches = 0
            color_matches = []
            
            for expected_color in expected_colors:
                best_match_score = 0
                best_actual_color = None
                
                for actual_color in actual_colors:
                    # Use existing color matching method
                    match_result = self.find_best_color_match(actual_color, expected_color)
                    if match_result.get('match_found'):
                        match_score = match_result['score'] / 100.0  # Convert to 0-1 range
                        if match_score > best_match_score:
                            best_match_score = match_score
                            best_actual_color = actual_color
                
                color_matches.append({
                    'expected': expected_color,
                    'best_match_score': best_match_score,
                    'best_actual_color': best_actual_color
                })
                total_matches += best_match_score
            
            average_match = (total_matches / len(expected_colors)) * 100
            
            return {
                'match_score': average_match,
                'detailed_matches': color_matches,
                'colors_expected': len(expected_colors),
                'colors_found': len(actual_colors)
            }
            
        except Exception as e:
            return {'match_score': 30.0, 'error': str(e)}
    
    def create_color_visualization(self, evaluation_result: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> None:
        """
        Create visualization of color evaluation results
        
        Args:
            evaluation_result: Evaluation results
            save_path: Save path
        """
        if 'error' in evaluation_result:
            print(f"Cannot create visualization due to error: {evaluation_result['error']}")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Color Evaluation Results (Score: {evaluation_result["overall_score"]:.1f}/100)', 
                        fontsize=16, fontweight='bold')
            
            # 1. Dimension score bar chart
            dimensions = list(evaluation_result['dimension_scores'].keys())
            scores = list(evaluation_result['dimension_scores'].values())
            
            colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in scores]
            axes[0, 0].bar(dimensions, scores, color=colors, alpha=0.7)
            axes[0, 0].set_ylim(0, 100)
            axes[0, 0].set_title('Dimension Scores')
            axes[0, 0].set_ylabel('Score')
            
            # 2. Extracted dominant colors
            extracted_colors = evaluation_result['detailed_results']['extracted_colors'][:12]
            for i, color in enumerate(extracted_colors):
                row, col = divmod(i, 6)
                color_normalized = np.array(color) / 255.0
                rect = plt.Rectangle((col, 1-row), 1, 1, facecolor=color_normalized, 
                                   edgecolor='black', linewidth=1)
                axes[0, 1].add_patch(rect)
            
            axes[0, 1].set_xlim(0, 6)
            axes[0, 1].set_ylim(0, 2)
            axes[0, 1].set_title('Extracted Dominant Colors')
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            
            # 3. Expected colors vs actual match
            accuracy_data = evaluation_result['detailed_results']['accuracy']
            if 'detailed_results' in accuracy_data:
                y_pos = 0
                for importance, matches in accuracy_data['detailed_results'].items():
                    for match in matches:
                        if 'extracted_rgb' in match:
                            color_normalized = np.array(match['extracted_rgb']) / 255.0
                            rect = plt.Rectangle((0, y_pos), 2, 0.8, facecolor=color_normalized,
                                               edgecolor='black', linewidth=1)
                            axes[1, 0].add_patch(rect)
                            
                            # Add score
                            score = match.get('score', 0)
                            axes[1, 0].text(2.1, y_pos + 0.4, f"{score:.0f}", 
                                          ha='left', va='center', fontweight='bold')
                            y_pos += 1
            
            axes[1, 0].set_xlim(0, 4)
            axes[1, 0].set_title('Color Matching Results')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])
            
            # 4. Evaluation summary
            summary_text = f"""Overall Score: {evaluation_result['overall_score']:.1f}/100
            
Accuracy: {evaluation_result['dimension_scores'].get('color_accuracy', 0):.1f}/100
Object Colors: {evaluation_result['dimension_scores'].get('object_color_accuracy', 0):.1f}/100

Colors Extracted: {evaluation_result['evaluation_summary']['total_colors_extracted']}
Colors Expected: {evaluation_result['evaluation_summary']['total_colors_expected']}
Coverage: {evaluation_result['evaluation_summary'].get('color_coverage', 0):.1%}"""
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[1, 1].set_title('Evaluation Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✅ Color visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating color visualization: {e}")


# Export main class
__all__ = ['ColorValidator']


