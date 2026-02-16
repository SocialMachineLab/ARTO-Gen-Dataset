import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans

class ColorAnalyzer:
    """CV-based Color Analyzer - Ported from original project color_validator.py"""
    
    def __init__(self):
        # Default color mapping (Hardcoded to avoid Pandas/CSV dependency)
        # Load GT mapping
        self.gt_mapping = {}
        
        # Try multiple locations for the color mapping file
        # Priority: 1) arto_kg/config, 2) current directory, 3) parent directory
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "config", "gt_color_mapping.json"),  # arto_kg/config/
            os.path.join(os.getcwd(), "gt_color_mapping.json"),  # Current directory
            os.path.join(os.path.dirname(os.getcwd()), "gt_color_mapping.json"),  # Parent directory
        ]
        
        mapping_path = None
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                mapping_path = normalized_path
                break
        
        if mapping_path:
            with open(mapping_path, 'r') as f:
                self.gt_mapping = json.load(f)
        else:
            print(f"Warning: Color mapping not found in any expected location, using default mapping")

        # Default fallback mapping
        self.default_color_mapping = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (165, 42, 42), # Updated brown
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'silver': (192, 192, 192),
            'gold': (255, 215, 0)
        }
        
    def _hex_to_rgb(self, hex_str: str) -> Tuple[int, int, int]:
        """Convert hashless hex string to RGB tuple"""
        hex_str = hex_str.lstrip('#')
        return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

    def evaluate_object_colors(self, image_path: str,
                             bboxes_list: List[Dict],
                             expected_colors: Dict[str, List[str]]) -> Dict[str, Any]:
        """Evaluate object color matching"""
        if not expected_colors:
            return {'average_score': 100.0, 'details': [], 'note': 'No color expectations'}

        # Construct bbox mapping
        name_to_bbox = {}
        for det in bboxes_list:
            label = det.get('label')
            box = det.get('box') # [x1, y1, x2, y2] relative
            if label and box:
                # Keep all detected instances, or only the largest?
                # Assuming we match the largest or first one for each category
                # For simplicity, we use the largest one
                w = box[2] - box[0]
                h = box[3] - box[1]
                area = w * h
                if label not in name_to_bbox or area > name_to_bbox[label].get('area', -1):
                     name_to_bbox[label] = {
                         'box': box,
                         'area': area
                     }

        results = []
        total_score = 0
        count = 0
        
        # Load image once
        image = cv2.imread(image_path)
        if image is None:
            return {'overall_score': 0.0, 'details': [], 'error': 'Cannot read image'}
            
        img_h, img_w = image.shape[:2]

        for obj_name, color_desc_list in expected_colors.items():
            count += 1
            # Find matching BBox
            matched_bbox = None
            for label, info in name_to_bbox.items():
                 if label in obj_name or obj_name in label:
                     matched_bbox = info['box']
                     break
            
            if not matched_bbox:
                results.append({
                    'object': obj_name, 
                    'expected': color_desc_list, 
                    'actual': 'not detected', 
                    'match_score': 0.0
                })
                # continue # Skip scoring if not detected? Or count as 0? 0 is fine.
                continue
                
            # Auto-detect coordinate system
            is_absolute = any(c > 1.0 for c in matched_bbox)
            if is_absolute:
                x1, y1, x2, y2 = map(int, matched_bbox)
            else:
                x1 = int(matched_bbox[0] * img_w)
                y1 = int(matched_bbox[1] * img_h)
                x2 = int(matched_bbox[2] * img_w)
                y2 = int(matched_bbox[3] * img_h)
            
            # Extract colors
            extracted_colors = self.extract_colors_from_region(image, [x1, y1, x2, y2])
            
            # Calculate matching score (Updated Rank-Based Strategy)
            # Find best match for ANY of the color descriptions for this object
            best_obj_score = 0.0
            
            # Normalize color descriptions list
            # Usually input is List[str] e.g. ["warm brown"]
            # Color validator expects target RGB
            
            target_rgbs = []
            for c_desc in color_desc_list: # color_desc_list is ['warm brown']
                # Try map to RGB
                mapped_info = self.gt_mapping.get(c_desc)
                if not mapped_info:
                    # Try normalization if key not found (but keys in json are normalized?)
                    # gt_mapping keys are exactly what was used to generate it?
                    # Actually generate_color_mapping uses RAW descriptions. So exact match should work.
                    # Fallback to default mapping if needed
                    # Or treat as "Unverifiable" -> 100
                    # If ONE description is unverifiable, do we pass?
                    # Strategy: If Unverifiable, assume target is "Any" -> Score 100.
                    target_rgbs.append("UNVERIFIABLE")
                else:
                    # FIX: Handle 'hex' key from mapping file
                    if 'rgb' in mapped_info:
                        target_rgbs.append(tuple(mapped_info['rgb']))
                    elif 'hex' in mapped_info:
                        target_rgbs.append(self._hex_to_rgb(mapped_info['hex']))
                    else:
                        target_rgbs.append("UNVERIFIABLE")

            if not target_rgbs:
                 # No valid targets found (should not happen if input had text)
                 best_obj_score = 100.0
            elif "UNVERIFIABLE" in target_rgbs:
                 # If we can't verify it, we give benefit of doubt
                 best_obj_score = 100.0
            else:
                # Compare Extracted Top-3 vs Target RGBs
                # extracted_colors = [{'rgb': (r,g,b), 'proportion': p, 'name': n}, ...]
                # Only care about Top 3 by proportion
                top_3 = sorted(extracted_colors, key=lambda x: x['proportion'], reverse=True)[:3]
                
                # Formula: Score = Max( Sim(Top1)*1.0, Sim(Top2)*0.9, Sim(Top3)*0.8 )
                # We want maximum similarity between ANY target and ANY top-k (weighted)
                
                for target_rgb in target_rgbs:
                    # target_rgb is (r,g,b)
                    current_target_score = 0.0
                    for i, extracted in enumerate(top_3):
                        weight = 1.0
                        if i == 1: weight = 0.9
                        if i == 2: weight = 0.8
                        
                        sim = self.calculate_similarity(target_rgb, extracted['rgb'])
                        weighted_sim = sim * weight
                        if weighted_sim > current_target_score:
                            current_target_score = weighted_sim
                            
                    if current_target_score > best_obj_score:
                        best_obj_score = current_target_score
            
            match_score = best_obj_score
            
            # Extract color names and RGB values
            top_colors_info = []
            for c in extracted_colors[:3]:
                top_colors_info.append({
                    "name": c['color_name'],
                    "rgb": c['rgb'], 
                    "proportion": c['proportion']
                })
            
            top_color_names = [c['name'] for c in top_colors_info]
            
            results.append({
                'object': obj_name,
                'expected': color_desc_list,
                'actual_detected_colors': top_color_names,
                'detailed_colors': top_colors_info,
                'match_score': round(match_score, 2)
            })
            total_score += match_score
            
        overall = (total_score / count) if count > 0 else 100.0
        return {'overall_score': round(overall, 2), 'details': results}

    def calculate_similarity(self, rgb1, rgb2):
        """Calculate score 0-100 based on Euclidean distance"""
        r1, g1, b1 = rgb1
        r2, g2, b2 = rgb2
        dist = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
        # Max dist is 441.67 (sqrt(255^2 * 3))
        # Use a steeper curve? dist < 50 is good.
        # Linear: 100 * (1 - dist/442)
        score = 100.0 * (1.0 - (dist / 442.0))
        return max(0.0, score)

    def extract_colors_from_region(self, image: np.ndarray, bbox: List[int]) -> List[Dict[str, Any]]:
        """Extract dominant colors from region"""
        x1, y1, x2, y2 = bbox
        
        # Boundary check
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        if x2 <= x1 or y2 <= y1:
            return []
            
        roi = image[y1:y2, x1:x2]
        if roi.size == 0: return []
        
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        pixels = roi_rgb.reshape(-1, 3)
        
        # Downsample for speedup
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
            
        if len(pixels) == 0: return []
            
        n_clusters = min(5, len(pixels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=5) # n_init=5 for speed
        kmeans.fit(pixels)
        
        centers = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        
        color_info = []
        for i, center in enumerate(centers):
            proportion = np.sum(labels == i) / len(labels)
            color_name = self.rgb_to_color_name(tuple(center))
            color_info.append({
                'rgb': tuple(center),
                'proportion': proportion,
                'color_name': color_name
            })
            
        color_info.sort(key=lambda x: x['proportion'], reverse=True)
        return color_info

    def calculate_color_match_score(self, extracted: List[Dict], expected_names: List[str]) -> float:
        """Calculate color match score"""
        if not expected_names or not extracted: return 0.0
        
        normalized_expected = [self._map_color_name(n) for n in expected_names]
        
        max_score = 0.0
        for exp_name in normalized_expected:
            exp_rgb = self.default_color_mapping.get(exp_name)
            if not exp_rgb: continue
            
            # Find best matching extracted color
            for ext in extracted:
                ext_rgb = ext['rgb']
                # Euclidean distance
                dist = np.sqrt(sum((a-b)**2 for a,b in zip(exp_rgb, ext_rgb)))
                # Normalize (max distance is sqrt(3*255^2) ≈ 441.67)
                sim = max(0, 100 * (1 - dist / 442.0))
                
                # Weight (proportion)
                weighted_sim = sim * (0.5 + 0.5 * ext['proportion']) # partially dependent on proportion
                max_score = max(max_score, weighted_sim)
                
        return min(100.0, max_score)

    def _map_color_name(self, name: str) -> str:
        """Simple color name mapping"""
        name = name.lower()
        if 'metal' in name or 'stainless' in name: return 'silver'
        if 'wood' in name: return 'brown'
        if 'skin' in name: return 'beige'
        
        # Try direct match
        if name in self.default_color_mapping: return name
        
        # Try containment match
        for k in self.default_color_mapping:
            if k in name: return k
            
        return 'gray' # Default

    def rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """RGB to nearest color name"""
        min_dist = float('inf')
        best_name = 'unknown'
        
        for name, val in self.default_color_mapping.items():
            dist = np.sqrt(sum((a-b)**2 for a,b in zip(rgb, val)))
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name
