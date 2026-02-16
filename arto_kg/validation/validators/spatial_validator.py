"""
Spatial Relation Evaluation Module
Specializes in spatial relationship, layout rationality, and composition analysis
"""

import cv2
import numpy as np
import json
import os
import sys
import logging
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

logger = logging.getLogger(__name__)


@dataclass
class SpatialRelation:
    """Spatial Relation Data Class"""
    object1: str
    object2: str
    relation_type: str
    confidence: float
    geometric_relation: str
    distance: float
    description: str


class SpatialValidator:
    """Spatial Relation Validator"""

    def __init__(self, vlm_wrapper=None):
        """
        Initialize Spatial Relation Validator
        
        Args:
            vlm_wrapper: VLM wrapper for complex spatial relationship analysis
        """
        self.vlm_wrapper = vlm_wrapper
        self.position_threshold = 50  # Pixel threshold
        
        # Spatial relation mapping
        self.geometric_relations = {
            'left_of_above': 'northwest',
            'left_of_aligned': 'west', 
            'left_of_below': 'southwest',
            'aligned_above': 'north',
            'aligned_aligned': 'center',
            'aligned_below': 'south',
            'right_of_above': 'northeast',
            'right_of_aligned': 'east',
            'right_of_below': 'southeast'
        }
        
        # Complex relation keywords
        self.complex_relation_keywords = {
            'support': ['on', 'sitting on', 'standing on', 'lying on', 'placed on'],
            'containment': ['in', 'inside', 'within', 'contained in'],
            'adjacency': ['next to', 'beside', 'near', 'close to', 'adjacent to'],
            'occlusion': ['behind', 'in front of', 'covering', 'hidden by'],
            'interaction': ['holding', 'touching', 'connected to', 'attached to']
        }
    
    def calculate_geometric_relations(self, bboxes_dict: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate geometric spatial relations between objects
        
        Args:
            bboxes_dict: Object bounding box dict {object_name: [x1, y1, x2, y2]}
            
        Returns:
            Geometric relation dict
        """
        relations = {}
        
        for obj1_name, bbox1 in bboxes_dict.items():
            relations[obj1_name] = {}
            
            for obj2_name, bbox2 in bboxes_dict.items():
                if obj1_name != obj2_name:
                    relation_info = self._analyze_bbox_relationship(bbox1, bbox2, obj1_name, obj2_name)
                    relations[obj1_name][obj2_name] = relation_info
        
        return relations
    
    def _analyze_bbox_relationship(self, bbox1: List[int], bbox2: List[int], 
                                  obj1_name: str, obj2_name: str) -> Dict[str, Any]:
        """
        Analyze relationship between two bounding boxes
        
        Args:
            bbox1, bbox2: Bounding boxes [x1, y1, x2, y2]
            obj1_name, obj2_name: Object names
            
        Returns:
            Relation info dict
        """
        # Calculate center points
        center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
        center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
        
        # Calculate distance
        distance = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Determine relative position
        horizontal_relation = self._get_horizontal_relation(center1[0], center2[0])
        vertical_relation = self._get_vertical_relation(center1[1], center2[1])
        
        geometric_relation = f"{horizontal_relation}_{vertical_relation}"
        direction = self.geometric_relations.get(geometric_relation, geometric_relation)
        
        # Analyze overlap
        overlap_info = self._calculate_overlap(bbox1, bbox2)
        
        # Analyze size relationship
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        size_relation = 'larger' if area1 > area2 * 1.2 else 'smaller' if area1 < area2 * 0.8 else 'similar'
        
        return {
            'geometric_relation': geometric_relation,
            'direction': direction,
            'distance': distance,
            'overlap_ratio': overlap_info['overlap_ratio'],
            'overlap_area': overlap_info['overlap_area'],
            'size_relation': size_relation,
            'relative_position': {
                'horizontal': horizontal_relation,
                'vertical': vertical_relation
            }
        }
    
    def _get_horizontal_relation(self, x1: float, x2: float) -> str:
        """Get horizontal relation"""
        if x1 < x2 - self.position_threshold:
            return "left_of"
        elif x1 > x2 + self.position_threshold:
            return "right_of"
        else:
            return "aligned"
    
    def _get_vertical_relation(self, y1: float, y2: float) -> str:
        """Get vertical relation"""
        if y1 < y2 - self.position_threshold:
            return "above"
        elif y1 > y2 + self.position_threshold:
            return "below"
        else:
            return "aligned"
    
    def _calculate_overlap(self, bbox1: List[int], bbox2: List[int]) -> Dict[str, float]:
        """Calculate overlap between two bounding boxes"""
        # Calculate overlap area
        x_overlap = max(0, min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0]))
        y_overlap = max(0, min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
        overlap_area = x_overlap * y_overlap
        
        # Calculate respective areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate overlap ratio (relative to smaller object)
        min_area = min(area1, area2)
        overlap_ratio = overlap_area / min_area if min_area > 0 else 0
        
        return {
            'overlap_area': overlap_area,
            'overlap_ratio': overlap_ratio
        }
    
    def extract_expected_spatial_relations(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract expected spatial relations from JSON - Optimized, handles duplicates
        
        Args:
            json_data: Artwork JSON data
            
        Returns:
            Expected spatial relation info
        """
        expected_relations = {
            'basic_relations': {},
            'complex_relations': {},
            'composition_rules': [],
            'layout_preferences': {},
            'object_positions': {},
            'object_relationships': []
        }
        
        try:
            # Extract from composition data - Smart deduplication
            composition = json_data.get('composition', {})
            
            # Priority handling: spatial_relationships > scene_framework > spatial_layout
            spatial_data_sources = [
                ('spatial_relationships', composition.get('spatial_relationships', {})),
                ('scene_framework', composition.get('scene_framework', {})),
                ('spatial_layout', composition.get('spatial_layout', {}))
            ]
            
            best_spatial_data = None
            best_source_name = None
            
            # Select most complete data source
            for source_name, spatial_data in spatial_data_sources:
                if isinstance(spatial_data, dict) and spatial_data:
                    # Check data quality
                    if 'parsing_details' in spatial_data:
                        parsing_details = spatial_data['parsing_details']
                        if isinstance(parsing_details, dict) and parsing_details.get('status') == 'success':
                            best_spatial_data = spatial_data
                            best_source_name = source_name
                            break
                    elif 'object_positions' in spatial_data or 'object_relationships' in spatial_data:
                        # No parsing_details but has core data
                        if best_spatial_data is None:
                            best_spatial_data = spatial_data
                            best_source_name = source_name
            
            if best_spatial_data:
                print(f"   Using {best_source_name} as primary spatial data source")
                
                # Extract object position info
                object_positions = best_spatial_data.get('object_positions', {})
                if isinstance(object_positions, dict):
                    expected_relations['object_positions'] = object_positions
                    
                    # Extract layout preferences from position info
                    for obj_name, pos_info in object_positions.items():
                        if isinstance(pos_info, dict):
                            position = pos_info.get('position', '')
                            size_in_comp = pos_info.get('size_in_composition', '')
                            if position:
                                expected_relations['layout_preferences'][obj_name] = {
                                    'position': position,
                                    'size_in_composition': size_in_comp
                                }
                
                # Extract object relationships
                object_relationships = best_spatial_data.get('object_relationships', [])
                if isinstance(object_relationships, list):
                    expected_relations['object_relationships'] = object_relationships
                    
                    # Convert to basic_relations format
                    for i, rel in enumerate(object_relationships):
                        if isinstance(rel, dict):
                            obj1 = rel.get('object1', '')
                            obj2 = rel.get('object2', '')
                            relationship = rel.get('relationship', '')
                            
                            if obj1 and obj2 and relationship:
                                relation_key = f"{obj1}_{obj2}_relation"
                                expected_relations['basic_relations'][relation_key] = {
                                    'object1': obj1,
                                    'object2': obj2,
                                    'description': relationship,
                                    'relation_type': 'spatial',
                                    'source': best_source_name
                                }
                                
                                # Check if contains complex relation
                                complex_relation = self._identify_complex_relation(relationship)
                                if complex_relation:
                                    expected_relations['complex_relations'][relation_key] = complex_relation
                
                # Extract composition type
                composition_type = best_spatial_data.get('composition_type', '')
                if composition_type:
                    expected_relations['composition_rules'].append(f"Composition type: {composition_type}")
                
                # Extract extra info from raw_output (if valid)
                raw_output = best_spatial_data.get('raw_output', '')
                if raw_output and isinstance(raw_output, str) and len(raw_output) > 50:
                    # Try to extract more info from raw_output
                    additional_info = self._parse_raw_spatial_output(raw_output)
                    if additional_info:
                        expected_relations['composition_rules'].append(f"Additional spatial info: {additional_info}")
            
            # Extract position preferences from enhanced_objects
            enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
            for obj in enhanced_objects:
                if isinstance(obj, dict):
                    obj_name = obj.get('name', '')
                    if obj_name and obj_name not in expected_relations['layout_preferences']:
                        # Infer position info from artistic description
                        artistic_desc = obj.get('artistic_description', '')
                        if artistic_desc:
                            inferred_position = self._infer_position_from_description(artistic_desc)
                            if inferred_position:
                                expected_relations['layout_preferences'][obj_name] = {
                                    'position': inferred_position,
                                    'source': 'artistic_description'
                                }
            
            # Extract composition rules from artistic_expression
            artistic_expr = json_data.get('artistic_expression', {})
            if isinstance(artistic_expr, dict):
                composition_notes = artistic_expr.get('composition_notes', '')
                if composition_notes:
                    expected_relations['composition_rules'].append(composition_notes)
        
        except Exception as e:
            print(f"Warning: Error extracting expected spatial relations: {e}")
        
        return expected_relations
    
    def _parse_raw_spatial_output(self, raw_output: str) -> Optional[str]:
        """Parse useful spatial info from raw_output"""
        try:
            # Try parsing JSON format raw_output
            if raw_output.strip().startswith('{'):
                import json
                parsed = json.loads(raw_output)
                if isinstance(parsed, dict):
                    # Extract useful info
                    useful_info = []
                    if 'composition_type' in parsed:
                        useful_info.append(f"type: {parsed['composition_type']}")
                    if 'layout_style' in parsed:
                        useful_info.append(f"style: {parsed['layout_style']}")
                    return ', '.join(useful_info) if useful_info else None
        except:
            pass
        
        # Simple text keyword extraction
        spatial_keywords = ['balanced', 'symmetric', 'asymmetric', 'centered', 'triangular', 'linear', 'clustered']
        found_keywords = [kw for kw in spatial_keywords if kw.lower() in raw_output.lower()]
        return ', '.join(found_keywords) if found_keywords else None
    
    def _infer_position_from_description(self, description: str) -> Optional[str]:
        """Infer position from artistic description"""
        desc_lower = description.lower()
        
        position_keywords = {
            'center': ['center', 'central', 'middle', 'prominently', 'stands'],
            'left': ['left', 'leftward'],
            'right': ['right', 'rightward'], 
            'background': ['background', 'distant', 'behind', 'back'],
            'foreground': ['foreground', 'front', 'prominent', 'close'],
            'top': ['top', 'above', 'upper'],
            'bottom': ['bottom', 'below', 'lower']
        }
        
        for position, keywords in position_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return position
        
        return None
    
    def _parse_relation_description(self, relation_key: str, description: str) -> Optional[Dict[str, Any]]:
        """Parse relation description"""
        try:
            # Try extracting object names from relation key
            objects = relation_key.replace('_relation', '').replace('_', ' ').split()
            if len(objects) >= 2:
                obj1, obj2 = objects[0], objects[1]
                
                return {
                    'object1': obj1,
                    'object2': obj2,
                    'description': description.lower(),
                    'relation_type': 'spatial'
                }
        except Exception:
            pass
        
        return None
    
    def _identify_complex_relation(self, description: str) -> Optional[Dict[str, Any]]:
        """Identify complex spatial relation"""
        description_lower = description.lower()
        
        for relation_type, keywords in self.complex_relation_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return {
                        'type': relation_type,
                        'keyword': keyword,
                        'description': description,
                        'requires_vlm': True
                    }
        
        return None
    
    def analyze_layout_composition(self, bboxes_dict: Dict[str, List[int]], 
                                  image_path: str) -> Dict[str, Any]:
        """
        Analyze overall layout and composition
        
        Args:
            bboxes_dict: Object bounding box dict
            image_path: Image path
            
        Returns:
            Layout composition analysis result
        """
        if not bboxes_dict:
            return {'error': 'No objects for layout analysis'}
        
        try:
            # Get image size
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not read image'}
            
            height, width = image.shape[:2]
            
            # Analyze object distribution
            centers = []
            areas = []
            
            for obj_name, bbox in bboxes_dict.items():
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                
                centers.append([center_x, center_y])
                areas.append(area)
            
            centers = np.array(centers)
            areas = np.array(areas)
            
            # Calculate composition metrics
            composition_analysis = {
                'balance_analysis': self._analyze_balance(centers, areas, width, height),
                'symmetry_analysis': self._analyze_symmetry(centers, width, height),
                'rule_of_thirds': self._analyze_rule_of_thirds(centers, width, height),
                'visual_weight': self._analyze_visual_weight(bboxes_dict, areas),
                'spacing_analysis': self._analyze_spacing(centers),
                'composition_score': 0
            }
            
            # Calculate overall composition score
            weights = {
                'balance': 0.3,
                'symmetry': 0.2,
                'rule_of_thirds': 0.2,
                'visual_weight': 0.15,
                'spacing': 0.15
            }
            
            composition_score = (
                composition_analysis['balance_analysis']['score'] * weights['balance'] +
                composition_analysis['symmetry_analysis']['score'] * weights['symmetry'] +
                composition_analysis['rule_of_thirds']['score'] * weights['rule_of_thirds'] +
                composition_analysis['visual_weight']['score'] * weights['visual_weight'] +
                composition_analysis['spacing_analysis']['score'] * weights['spacing']
            )
            
            composition_analysis['composition_score'] = composition_score
            
            return composition_analysis
            
        except Exception as e:
            return {'error': f'Layout analysis failed: {e}'}
    
    def _analyze_balance(self, centers: np.ndarray, areas: np.ndarray, 
                        width: int, height: int) -> Dict[str, Any]:
        """Analyze balance"""
        # Calculate visual center of gravity
        weighted_center_x = np.sum(centers[:, 0] * areas) / np.sum(areas)
        weighted_center_y = np.sum(centers[:, 1] * areas) / np.sum(areas)
        
        # Calculate degree of deviation from center
        center_x_offset = abs(weighted_center_x - width / 2) / (width / 2)
        center_y_offset = abs(weighted_center_y - height / 2) / (height / 2)
        
        # Calculate left-right balance
        left_weight = np.sum(areas[centers[:, 0] < width / 2])
        right_weight = np.sum(areas[centers[:, 0] > width / 2])
        horizontal_balance = 1 - abs(left_weight - right_weight) / (left_weight + right_weight + 1e-6)
        
        # Calculate top-bottom balance
        top_weight = np.sum(areas[centers[:, 1] < height / 2])
        bottom_weight = np.sum(areas[centers[:, 1] > height / 2])
        vertical_balance = 1 - abs(top_weight - bottom_weight) / (top_weight + bottom_weight + 1e-6)
        
        # Overall balance score
        balance_score = (
            (1 - center_x_offset) * 0.3 +
            (1 - center_y_offset) * 0.3 +
            horizontal_balance * 0.2 +
            vertical_balance * 0.2
        ) * 100
        
        return {
            'score': balance_score,
            'weighted_center': [weighted_center_x, weighted_center_y],
            'horizontal_balance': horizontal_balance,
            'vertical_balance': vertical_balance,
            'center_offset': [center_x_offset, center_y_offset]
        }
    
    def _analyze_symmetry(self, centers: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Analyze symmetry"""
        # Horizontal symmetry analysis
        center_x = width / 2
        left_points = centers[centers[:, 0] < center_x]
        right_points = centers[centers[:, 0] > center_x]
        
        # Calculate if each left point has corresponding right point
        symmetry_matches = 0
        total_checks = len(left_points)
        
        if total_checks > 0:
            for left_point in left_points:
                # Mirror point
                mirror_x = 2 * center_x - left_point[0]
                mirror_point = np.array([mirror_x, left_point[1]])
                
                # Find nearest right point
                if len(right_points) > 0:
                    distances = np.linalg.norm(right_points - mirror_point, axis=1)
                    min_distance = np.min(distances)
                    
                    # If distance less than threshold, consider symmetric
                    if min_distance < width * 0.1:  # 10% of image width as threshold
                        symmetry_matches += 1
        
        horizontal_symmetry = symmetry_matches / max(1, total_checks)
        
        # Simplified symmetry score
        symmetry_score = horizontal_symmetry * 100
        
        return {
            'score': symmetry_score,
            'horizontal_symmetry': horizontal_symmetry,
            'symmetry_matches': symmetry_matches,
            'total_points': len(centers)
        }
    
    def _analyze_rule_of_thirds(self, centers: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Analyze rule of thirds"""
        # Thirds line positions
        third_lines_x = [width / 3, 2 * width / 3]
        third_lines_y = [height / 3, 2 * height / 3]
        
        # Intersection points
        power_points = [
            [width / 3, height / 3], [2 * width / 3, height / 3],
            [width / 3, 2 * height / 3], [2 * width / 3, 2 * height / 3]
        ]
        
        # Calculate how many points are close to thirds lines or intersections
        line_proximity_count = 0
        point_proximity_count = 0
        
        for center in centers:
            # Check proximity to thirds lines
            min_line_distance = min(
                abs(center[0] - third_lines_x[0]),
                abs(center[0] - third_lines_x[1]),
                abs(center[1] - third_lines_y[0]),
                abs(center[1] - third_lines_y[1])
            )
            
            if min_line_distance < min(width, height) * 0.05:  # 5% threshold
                line_proximity_count += 1
            
            # Check proximity to intersections
            for power_point in power_points:
                distance = np.linalg.norm(center - power_point)
                if distance < min(width, height) * 0.1:  # 10% threshold
                    point_proximity_count += 1
                    break
        
        # Calculate rule of thirds score
        line_score = line_proximity_count / len(centers) * 60
        point_score = point_proximity_count / len(centers) * 40
        rule_of_thirds_score = line_score + point_score
        
        return {
            'score': rule_of_thirds_score,
            'line_proximity_count': line_proximity_count,
            'point_proximity_count': point_proximity_count,
            'total_objects': len(centers)
        }
    
    def _analyze_visual_weight(self, bboxes_dict: Dict[str, List[int]], 
                              areas: np.ndarray) -> Dict[str, Any]:
        """Analyze visual weight distribution"""
        # Calculate standard deviation of area distribution (uniformity metric)
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        area_cv = area_std / area_mean if area_mean > 0 else 0  # Coefficient of variation
        
        # Ideal CV range (some large objects, some small objects)
        ideal_cv = 0.5
        cv_score = max(0, 100 - abs(area_cv - ideal_cv) * 100)
        
        # Analyze ratio of max and min objects
        max_area = np.max(areas)
        min_area = np.min(areas)
        size_ratio = max_area / min_area if min_area > 0 else float('inf')
        
        # Reasonable size ratio range (2-10x)
        if 2 <= size_ratio <= 10:
            ratio_score = 100
        elif size_ratio < 2:
            ratio_score = 50  # All objects similar size
        else:
            ratio_score = max(0, 100 - (size_ratio - 10) * 5)  # Ratio too large
        
        visual_weight_score = (cv_score + ratio_score) / 2
        
        return {
            'score': visual_weight_score,
            'area_coefficient_of_variation': area_cv,
            'size_ratio': size_ratio,
            'area_distribution': {
                'mean': float(area_mean),
                'std': float(area_std),
                'min': float(min_area),
                'max': float(max_area)
            }
        }
    
    def _analyze_spacing(self, centers: np.ndarray) -> Dict[str, Any]:
        """Analyze object spacing"""
        if len(centers) < 2:
            return {'score': 100, 'note': 'Single object'}
        
        # Calculate distances between all objects
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                distance = np.linalg.norm(centers[i] - centers[j])
                distances.append(distance)
        
        distances = np.array(distances)
        
        # Analyze distance distribution
        distance_mean = np.mean(distances)
        distance_std = np.std(distances)
        distance_cv = distance_std / distance_mean if distance_mean > 0 else 0
        
        # Good spacing should be relatively uniform (low CV)
        # But not completely identical (needs some variation)
        ideal_cv_range = (0.2, 0.6)
        if ideal_cv_range[0] <= distance_cv <= ideal_cv_range[1]:
            spacing_score = 100
        else:
            if distance_cv < ideal_cv_range[0]:
                spacing_score = 70  # Too regular
            else:
                spacing_score = max(0, 100 - (distance_cv - ideal_cv_range[1]) * 100)
        
        return {
            'score': spacing_score,
            'distance_stats': {
                'mean': float(distance_mean),
                'std': float(distance_std),
                'coefficient_of_variation': float(distance_cv),
                'min_distance': float(np.min(distances)),
                'max_distance': float(np.max(distances))
            },
            'total_pairs': len(distances)
        }
    
    def analyze_spatial_semantic_consistency(self, image_path: str,
                                                   bboxes_dict: Dict[str, List[int]],
                                                   json_data: Dict[str, Any],
                                                   geometric_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze spatial semantic consistency - Combine rule-based analysis and VLM assessment

        Args:
            image_path: Image path
            bboxes_dict: Object bounding box dict
            json_data: Original JSON data
            geometric_analysis: Geometric relation analysis result

        Returns:
            Spatial semantic consistency analysis result
        """
        try:
            # 1. Rule-based spatial semantic consistency analysis (70%)
            semantic_analysis = self._analyze_semantic_consistency_rules(
                bboxes_dict, json_data, geometric_analysis
            )

            # 2. VLM overall spatial quality assessment (30%)
            vlm_quality_analysis = self._analyze_overall_spatial_quality_with_vlm(
                image_path, json_data, geometric_analysis, bboxes_dict
            )

            # 3. Overall score
            semantic_score = semantic_analysis.get('score', 70.0)
            vlm_score = vlm_quality_analysis.get('score', 70.0)

            overall_score = semantic_score * 0.7 + vlm_score * 0.3

            return {
                'score': overall_score,
                'semantic_consistency': semantic_analysis,
                'vlm_quality_assessment': vlm_quality_analysis,
                # 'evaluation_method': 'spatial_semantic_consistency',
                'note': 'Combined rule-based semantic analysis and VLM quality assessment'
            }

        except Exception as e:
            logger.error(f"Spatial semantic consistency analysis failed: {e}")
            return {
                'score': 65.0,
                'error': str(e),
                'note': 'Fallback score due to analysis failure'
            }

    def _analyze_semantic_consistency_rules(self, bboxes_dict: Dict[str, List[int]],
                                          json_data: Dict[str, Any],
                                          geometric_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based spatial semantic consistency analysis
        """
        try:
            scores = {}

            # 1. Spatial logic rationality (25%)
            spatial_logic_score = self._evaluate_spatial_logic(bboxes_dict, json_data)
            scores['spatial_logic'] = spatial_logic_score

            # 2. Proportion coordination (25%)
            proportion_score = self._evaluate_proportion_coordination(bboxes_dict, json_data)
            scores['proportion_coordination'] = proportion_score

            # 3. Position expectation matching (25%)
            position_matching_score = self._evaluate_position_expectations(bboxes_dict, json_data)
            scores['position_matching'] = position_matching_score

            # 4. Visual hierarchy clarity (25%)
            visual_hierarchy_score = self._evaluate_visual_hierarchy(bboxes_dict, geometric_analysis)
            scores['visual_hierarchy'] = visual_hierarchy_score

            # Calculate overall score
            weights = [0.25, 0.25, 0.25, 0.25]
            overall_score = sum(score * weight for score, weight in zip(scores.values(), weights))

            return {
                'score': overall_score,
                'detailed_scores': scores,
                'evaluation_aspects': ['spatial_logic', 'proportion_coordination', 'position_matching', 'visual_hierarchy']
            }

        except Exception as e:
            return {
                'score': 70.0,
                'error': str(e),
                'note': 'Rule-based analysis failed, using fallback score'
            }

    def _evaluate_spatial_logic(self, bboxes_dict: Dict[str, List[int]],
                               json_data: Dict[str, Any]) -> float:
        """
        Evaluate spatial logic rationality
        """
        # Check for unreasonable overlaps
        overlap_penalties = 0
        total_pairs = 0

        objects = list(bboxes_dict.keys())
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                total_pairs += 1
                bbox1 = bboxes_dict[objects[i]]
                bbox2 = bboxes_dict[objects[j]]

                overlap_info = self._calculate_overlap(bbox1, bbox2)
                overlap_ratio = overlap_info['overlap_ratio']

                # Penalty if too much overlap (>50%)
                if overlap_ratio > 0.5:
                    overlap_penalties += 1

        # Calculate spatial logic score
        if total_pairs == 0:
            return 80.0

        overlap_score = max(0, 100 - (overlap_penalties / total_pairs) * 100)
        return overlap_score

    def _evaluate_proportion_coordination(self, bboxes_dict: Dict[str, List[int]],
                                        json_data: Dict[str, Any]) -> float:
        """
        Evaluate proportion coordination
        """
        if len(bboxes_dict) < 2:
            return 85.0

        # Calculate object areas
        areas = []
        for bbox in bboxes_dict.values():
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            areas.append(area)

        areas = np.array(areas)

        # Calculate rationality of area ratio
        max_area = np.max(areas)
        min_area = np.min(areas)
        area_ratio = max_area / min_area if min_area > 0 else 1

        # Reasonable area ratio range (1.5-8x)
        if 1.5 <= area_ratio <= 8:
            ratio_score = 100
        elif area_ratio < 1.5:
            ratio_score = 70  # All objects too similar in size
        else:
            ratio_score = max(30, 100 - (area_ratio - 8) * 5)  # Ratio gap too large

        return ratio_score

    def _evaluate_position_expectations(self, bboxes_dict: Dict[str, List[int]],
                                      json_data: Dict[str, Any]) -> float:
        """
        Evaluate position expectation matching
        """
        # Extract position expectations from JSON
        expected_positions = {}

        # Extract position info from enhanced_objects
        enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
        for obj in enhanced_objects:
            if isinstance(obj, dict):
                obj_name = obj.get('name', '')
                artistic_desc = obj.get('artistic_description', '')
                if obj_name and artistic_desc:
                    inferred_pos = self._infer_position_from_description(artistic_desc)
                    if inferred_pos:
                        expected_positions[obj_name] = inferred_pos

        if not expected_positions:
            return 75.0  # No position expectations, give medium score

        # Check if actual position matches expectations
        match_scores = []
        for obj_name, expected_pos in expected_positions.items():
            if obj_name in bboxes_dict:
                actual_pos = self._determine_actual_position(bboxes_dict[obj_name])
                match_score = self._calculate_position_match(expected_pos, actual_pos)
                match_scores.append(match_score)

        return np.mean(match_scores) * 100 if match_scores else 75.0

    def _evaluate_visual_hierarchy(self, bboxes_dict: Dict[str, List[int]],
                                 geometric_analysis: Dict[str, Any]) -> float:
        """
        Evaluate visual hierarchy clarity
        """
        if len(bboxes_dict) < 2:
            return 80.0

        # Evaluate hierarchy based on object size and position
        hierarchy_score = 75.0

        # Check for clear primary/secondary relationship
        areas = []
        centers_y = []

        for bbox in bboxes_dict.values():
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            center_y = (bbox[1] + bbox[3]) / 2
            areas.append(area)
            centers_y.append(center_y)

        areas = np.array(areas)
        centers_y = np.array(centers_y)

        # Check size hierarchy
        area_variance = np.var(areas) / np.mean(areas) if np.mean(areas) > 0 else 0
        if area_variance > 0.3:  # Significant size difference
            hierarchy_score += 15

        # Check vertical hierarchy (foreground/background)
        y_variance = np.var(centers_y)
        if y_variance > 5000:  # Significant vertical hierarchy
            hierarchy_score += 10

        return min(100, hierarchy_score)

    def _determine_actual_position(self, bbox: List[int]) -> str:
        """
        Determine actual position from bbox
        """
        # Simplified position determination logic
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # Assuming image size, should use actual image size
        img_width = 1024  # 默认值，实际应该从图像获取
        img_height = 1024

        if center_x < img_width / 3:
            h_pos = "left"
        elif center_x > 2 * img_width / 3:
            h_pos = "right"
        else:
            h_pos = "center"

        if center_y < img_height / 3:
            v_pos = "top"
        elif center_y > 2 * img_height / 3:
            v_pos = "bottom"
        else:
            v_pos = "middle"

        return f"{h_pos}_{v_pos}"

    def _calculate_position_match(self, expected_pos: str, actual_pos: str) -> float:
        """
        Calculate position match score
        """
        # Simple matching logic
        if expected_pos.lower() in actual_pos.lower() or actual_pos.lower() in expected_pos.lower():
            return 1.0

        # Partial match
        expected_parts = expected_pos.lower().split('_')
        actual_parts = actual_pos.lower().split('_')

        matches = 0
        for part in expected_parts:
            if part in actual_parts:
                matches += 1

        return matches / len(expected_parts) if expected_parts else 0.0

    def _analyze_overall_spatial_quality_with_vlm(self, image_path: str,
                                                 json_data: Dict[str, Any],
                                                 geometric_analysis: Dict[str, Any],
                                                 bboxes_dict: Dict[str, List[int]]) -> Dict[str, Any]:
        """
        Evaluate overall spatial quality using VLM
        """
        if self.vlm_wrapper is None:
            return {
                'score': 70.0,
                'note': 'VLM not available, using fallback score',
                'method': 'fallback'
            }

        try:
            # Prepare analysis context
            context_info = self._prepare_spatial_quality_context(json_data, geometric_analysis, bboxes_dict)

            prompt = f"""
Analyze the overall spatial quality of this artwork image.

Context Information:
{context_info}

Please evaluate the following aspects on a scale of 0.0-1.0:
1. Spatial Logic Reasonableness: Are the spatial arrangements logical and make sense?
2. Visual Balance: Is the composition visually balanced and harmonious?
3. Composition Harmony: Do the elements work well together spatially?
4. Overall Spatial Quality: What is the overall spatial arrangement quality?

IMPORTANT: Provide your response in VALID JSON format only:

{{
  "spatial_logic": 0.0,
  "visual_balance": 0.0,
  "composition_harmony": 0.0,
  "overall_quality": 0.0,
  "analysis": "detailed explanation of your evaluation"
}}

Replace the 0.0 values with your actual scores. Ensure the JSON is valid and parseable.
"""

            # Call VLM
            vlm_result = self.vlm_wrapper.generate_response(prompt, image_path)

            if vlm_result.get('success', False):
                response = vlm_result.get('response', '')
                parsed_result = self._parse_vlm_spatial_quality_response(response)

                # Calculate overall score
                scores = []
                for key in ['spatial_logic', 'visual_balance', 'composition_harmony', 'overall_quality']:
                    if key in parsed_result:
                        scores.append(parsed_result[key])

                overall_score = np.mean(scores) * 100 if scores else 70.0

                return {
                    'score': overall_score,
                    'detailed_scores': parsed_result,
                    'method': 'vlm_analysis',
                    'analysis_explanation': parsed_result.get('analysis', '')
                }
            else:
                logger.warning(f"VLM spatial quality analysis failed: {vlm_result.get('error', 'Unknown error')}")
                return {
                    'score': 70.0,
                    'note': 'VLM analysis failed, using fallback score',
                    'error': vlm_result.get('error', 'Unknown error'),
                    'method': 'fallback'
                }

        except Exception as e:
            logger.error(f"VLM spatial quality analysis failed: {e}")
            return {
                'score': 65.0,
                'error': str(e),
                'method': 'error_fallback'
            }

    def _prepare_spatial_quality_context(self, json_data: Dict[str, Any],
                                       geometric_analysis: Dict[str, Any],
                                       bboxes_dict: Dict[str, List[int]]) -> str:
        """
        Prepare context info for spatial quality analysis
        """
        context_parts = []

        # Object info
        objects_info = []
        for obj_name, bbox in bboxes_dict.items():
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            objects_info.append(f"{obj_name}: center({center_x},{center_y}), size({width}x{height})")

        context_parts.append(f"Objects: {'; '.join(objects_info)}")

        # Expected composition info
        composition = json_data.get('composition', {})
        if composition:
            composition_type = composition.get('spatial_relationships', {}).get('composition_type', '')
            if composition_type:
                context_parts.append(f"Expected composition: {composition_type}")

        # Geometric analysis summary
        if geometric_analysis:
            total_relations = len([rel for obj_rels in geometric_analysis.values() for rel in obj_rels.values()])
            context_parts.append(f"Geometric relations analyzed: {total_relations}")

        return "\n".join(context_parts)

    def _parse_vlm_spatial_quality_response(self, response: str) -> Dict[str, Any]:
        """
        Parse VLM spatial quality assessment response
        """
        try:
            import json
            import re

            # Try parsing JSON directly
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                json_text = json_match.group(0)
                parsed_json = json.loads(json_text)

                result = {}
                # Ensure all scores are within reasonable range
                for key in ['spatial_logic', 'visual_balance', 'composition_harmony', 'overall_quality']:
                    if key in parsed_json:
                        score = float(parsed_json[key])
                        result[key] = max(0.0, min(1.0, score))
                    else:
                        result[key] = 0.7  # Default value

                if 'analysis' in parsed_json:
                    result['analysis'] = parsed_json['analysis']

                return result

            # Fallback parsing method
            return {
                'spatial_logic': 0.7,
                'visual_balance': 0.7,
                'composition_harmony': 0.7,
                'overall_quality': 0.7,
                'analysis': 'Failed to parse VLM response',
                'parse_error': 'JSON parsing failed'
            }

        except Exception as e:
            logger.error(f"Failed to parse VLM spatial quality response: {e}")
            return {
                'spatial_logic': 0.6,
                'visual_balance': 0.6,
                'composition_harmony': 0.6,
                'overall_quality': 0.6,
                'analysis': f'Parse failed: {str(e)}',
                'error': str(e)
            }
    
    def _prepare_spatial_context(self, bboxes_dict: Dict[str, List[int]]) -> str:
        """Prepare spatial context info for VLM"""
        context_parts = []
        
        for obj_name, bbox in bboxes_dict.items():
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            
            context_parts.append(
                f"{obj_name}: Center({center_x},{center_y}), Size({width}x{height})"
            )
        
        return "; ".join(context_parts)
    
    def _parse_vlm_spatial_response(self, response: str, relation_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse VLM spatial relation response"""
        try:
            lines = response.strip().split('\n')
            
            relation_present = False
            accuracy_score = 0.5
            observation = ""
            
            for line in lines:
                if 'Relation present:' in line:
                    relation_present = 'Yes' in line
                elif 'Accuracy score:' in line:
                    score_text = line.split('Accuracy score:')[1].strip()
                    try:
                        accuracy_score = float(score_text)
                    except ValueError:
                        accuracy_score = 0.5
                elif 'Observation:' in line:
                    observation = line.split('Observation:')[1].strip()
            
            return {
                'relation_present': relation_present,
                'score': accuracy_score,
                'observation': observation,
                'relation_type': relation_info.get('type', 'unknown'),
                'original_description': relation_info.get('description', '')
            }
            
        except Exception as e:
            return {
                'score': 0.3,
                'error': f'Failed to parse VLM response: {e}',
                'original_response': response
            }
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str,
                       bboxes_dict: Dict[str, List[int]],
                       visualize: bool = False) -> Dict[str, Any]:
        """Main spatial relation evaluation method"""
        return self.comprehensive_spatial_evaluation(json_data, image_path, bboxes_dict, visualize=visualize)
    
    def comprehensive_spatial_evaluation(self, json_data: Dict[str, Any], 
                                       image_path: str,
                                       bboxes_dict: Dict[str, List[int]],
                                       visualize: bool = False) -> Dict[str, Any]:
        """
        Comprehensive spatial relation evaluation
        
        Args:
            json_data: Artwork JSON data
            image_path: Image file path
            bboxes_dict: Object bounding box dict
            
        Returns:
            Complete spatial relation evaluation result
        """
        print(f"📐 Starting spatial relationship evaluation for: {os.path.basename(image_path)}")
        
        try:
            # 1. Extract expected spatial relations
            print("   Step 1: Extracting expected spatial relationships...")
            expected_relations = self.extract_expected_spatial_relations(json_data)
            
            # 2. Calculate geometric spatial relations
            print("   Step 2: Calculating geometric spatial relationships...")
            geometric_relations = self.calculate_geometric_relations(bboxes_dict)
            
            # 3. Analyze layout and composition
            print("   Step 3: Analyzing layout and composition...")
            layout_analysis = self.analyze_layout_composition(bboxes_dict, image_path)
            
            # 4. Analyze spatial semantic consistency
            # print("   Step 4: Analyzing spatial semantic consistency...")
            # spatial_semantic_result = self.analyze_spatial_semantic_consistency(
            #     image_path, bboxes_dict, json_data, geometric_relations
            # )
            
            # 5. Calculate basic spatial matching scores
            print("   Step 5: Calculating basic spatial matching scores...")
            basic_matching_score = self._calculate_basic_spatial_matching(
                geometric_relations, expected_relations['basic_relations']
            )
            
            # 6. Calculate overall spatial score
            print("   Step 6: Calculating overall spatial score...")
            
            weights = {
                'basic_relations': 0.5,
                'layout_composition': 0.5,
                # 'spatial_semantic': 0.3
            }

            # Get scores for each dimension
            basic_score = basic_matching_score.get('score', 50.0)
            layout_score = layout_analysis.get('composition_score', 50.0)
            # semantic_score = spatial_semantic_result.get('score', 50.0)

            overall_score = (
                basic_score * weights['basic_relations'] +
                layout_score * weights['layout_composition'] 
                # semantic_score * weights['spatial_semantic']
            )
            
            # 7. Compile final result
            evaluation_result = {
                'overall_score': overall_score,
                'dimension_scores': {
                    'basic_spatial_relations': basic_score,
                    'layout_composition': layout_score,
                    # 'spatial_semantic_consistency': semantic_score
                },
                'detailed_results': {
                    'expected_relations': expected_relations,
                    'geometric_relations': geometric_relations,
                    'layout_analysis': layout_analysis,
                    # 'spatial_semantic_consistency': spatial_semantic_result,
                    'basic_matching': basic_matching_score
                },
                'evaluation_summary': {
                    'total_objects': len(bboxes_dict),
                    'basic_relations_count': len(expected_relations['basic_relations']),
                    'semantic_analysis_enabled': True,
                    'vlm_available': self.vlm_wrapper is not None,
                    'evaluation_weights': weights
                },
                'image_file': image_path,
                'artwork_id': json_data.get('artwork_id', 'unknown')
            }
            
            print(f"   Spatial evaluation completed! Overall score: {overall_score:.1f}/100")
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Spatial relationship evaluation failed: {e}"
            print(f"   {error_msg}")
            return {'error': error_msg}
    
    def _calculate_basic_spatial_matching(self, geometric_relations: Dict[str, Dict[str, Any]], 
                                        expected_basic_relations: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate basic spatial relation matching score"""
        if not expected_basic_relations:
            return {'score': 70.0, 'note': 'No basic spatial relations to evaluate'}
        
        try:
            matching_scores = []
            detailed_matches = {}
            
            for relation_key, expected_relation in expected_basic_relations.items():
                obj1 = expected_relation.get('object1', '')
                obj2 = expected_relation.get('object2', '')
                expected_desc = expected_relation.get('description', '').lower()
                
                if obj1 in geometric_relations and obj2 in geometric_relations[obj1]:
                    actual_relation = geometric_relations[obj1][obj2]
                    
                    # Simple keyword matching evaluation
                    match_score = self._evaluate_spatial_description_match(
                        expected_desc, actual_relation
                    )
                    
                    matching_scores.append(match_score)
                    detailed_matches[relation_key] = {
                        'expected': expected_desc,
                        'actual': actual_relation,
                        'score': match_score
                    }
                else:
                    # Object not found or relation cannot be analyzed
                    matching_scores.append(0.0)
                    detailed_matches[relation_key] = {
                        'expected': expected_desc,
                        'actual': 'objects_not_found',
                        'score': 0.0
                    }
            
            average_score = np.mean(matching_scores) * 100 if matching_scores else 50.0
            
            return {
                'score': average_score,
                'detailed_matches': detailed_matches,
                'total_relations': len(expected_basic_relations),
                'successful_matches': len([s for s in matching_scores if s > 0.5])
            }
            
        except Exception as e:
            return {'score': 30.0, 'error': f'Basic spatial matching failed: {e}'}
    
    def _evaluate_spatial_description_match(self, expected_desc: str, 
                                          actual_relation: Dict[str, Any]) -> float:
        """Evaluate spatial description match"""
        # Simplified matching logic
        direction = actual_relation.get('direction', '')
        geometric_rel = actual_relation.get('geometric_relation', '')
        
        # Keyword mapping
        keyword_mapping = {
            'left': ['left', 'west'],
            'right': ['right', 'east'], 
            'above': ['above', 'north', 'top'],
            'below': ['below', 'south', 'bottom'],
            'next to': ['adjacent', 'beside', 'near'],
            'far from': ['far', 'distant']
        }
        
        match_score = 0.0
        
        for keyword, synonyms in keyword_mapping.items():
            if keyword in expected_desc:
                if any(syn in direction.lower() for syn in synonyms):
                    match_score += 0.5
                if any(syn in geometric_rel.lower() for syn in synonyms):
                    match_score += 0.3
        
        # Distance-related matching
        distance = actual_relation.get('distance', 0)
        if 'near' in expected_desc or 'close' in expected_desc:
            if distance < 100:  # Assume 100 pixels as near distance
                match_score += 0.2
        elif 'far' in expected_desc:
            if distance > 200:  # Assume 200 pixels as far distance
                match_score += 0.2
        
        return min(1.0, match_score)
    
    def create_spatial_visualization(self, evaluation_result: Dict[str, Any], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create visualization for spatial relation evaluation result
        
        Args:
            evaluation_result: Evaluation result
            save_path: Save path
        """
        if 'error' in evaluation_result:
            print(f"Cannot create spatial visualization due to error: {evaluation_result['error']}")
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Spatial Relationship Analysis (Score: {evaluation_result["overall_score"]:.1f}/100)', 
                        fontsize=16, fontweight='bold')
            
            # 1. Dimension scores
            dimensions = list(evaluation_result['dimension_scores'].keys())
            scores = list(evaluation_result['dimension_scores'].values())
            
            bar_colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in scores]
            axes[0, 0].bar(range(len(dimensions)), scores, color=bar_colors, alpha=0.7)
            
            for i, score in enumerate(scores):
                axes[0, 0].text(i, score + 1, f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
            
            axes[0, 0].set_ylim(0, 105)
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Spatial Dimension Scores')
            axes[0, 0].set_xticks(range(len(dimensions)))
            axes[0, 0].set_xticklabels([d.replace('_', '\n') for d in dimensions], fontsize=9)
            
            # 2. Layout analysis radar chart (simplified)
            layout_analysis = evaluation_result['detailed_results']['layout_analysis']
            if 'error' not in layout_analysis:
                layout_aspects = ['Balance', 'Symmetry', 'Rule of Thirds', 'Visual Weight', 'Spacing']
                layout_scores = [
                    layout_analysis['balance_analysis']['score'],
                    layout_analysis['symmetry_analysis']['score'],
                    layout_analysis['rule_of_thirds']['score'],
                    layout_analysis['visual_weight']['score'],
                    layout_analysis['spacing_analysis']['score']
                ]
                
                angles = np.linspace(0, 2 * np.pi, len(layout_aspects), endpoint=False)
                layout_scores += [layout_scores[0]]  # Close the shape
                angles = np.concatenate([angles, [angles[0]]])
                
                axes[0, 1].plot(angles, layout_scores, 'o-', linewidth=2)
                axes[0, 1].fill(angles, layout_scores, alpha=0.25)
                axes[0, 1].set_xticks(angles[:-1])
                axes[0, 1].set_xticklabels(layout_aspects)
                axes[0, 1].set_ylim(0, 100)
                axes[0, 1].set_title('Layout Composition Analysis')
                axes[0, 1].grid(True)
            
            # 3. Relation matching results
            basic_matching = evaluation_result['detailed_results'].get('basic_matching', {})
            if 'detailed_matches' in basic_matching:
                relations = list(basic_matching['detailed_matches'].keys())
                match_scores = [match['score'] * 100 for match in basic_matching['detailed_matches'].values()]
                
                if relations:
                    y_pos = range(len(relations))
                    colors = ['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in match_scores]
                    
                    axes[1, 0].barh(y_pos, match_scores, color=colors, alpha=0.7)
                    axes[1, 0].set_yticks(y_pos)
                    axes[1, 0].set_yticklabels([r.replace('_', ' ') for r in relations], fontsize=8)
                    axes[1, 0].set_xlabel('Match Score')
                    axes[1, 0].set_title('Spatial Relations Matching')
                    
                    for i, score in enumerate(match_scores):
                        axes[1, 0].text(score + 1, i, f'{score:.0f}', va='center', fontweight='bold')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No spatial relations to analyze', 
                                   ha='center', va='center', transform=axes[1, 0].transAxes)
                    axes[1, 0].set_title('Spatial Relations Matching')
            
            # 4. Evaluation summary
            summary = evaluation_result['evaluation_summary']
            
            # Safely get complex results
            complex_results = evaluation_result.get('detailed_results', {}).get('complex_relations', {})
            
            summary_text = f"""Overall Score: {evaluation_result['overall_score']:.1f}/100

Basic Relations: {evaluation_result['dimension_scores'].get('basic_spatial_relations', 0):.1f}/100
Layout Composition: {evaluation_result['dimension_scores'].get('layout_composition', 0):.1f}/100
Complex Relations: {evaluation_result['dimension_scores'].get('complex_relations', 0):.1f}/100

Total Objects: {summary.get('total_objects', 0)}
Basic Relations Count: {summary.get('basic_relations_count', 0)}
Complex Relations Count: {summary.get('complex_relations_count', 0)}
VLM Available: {"Yes" if summary.get('vlm_available', False) else "No"}

Layout Composition Score: {layout_analysis.get('composition_score', 0):.1f}/100"""
            
            if complex_results and 'error' not in complex_results:
                conf = complex_results.get('average_confidence', 0)
                # Handle case where confidence might be None
                if conf is None: 
                    conf = 0
                summary_text += f"\nComplex Relations Confidence: {conf:.2f}"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
            axes[1, 1].set_title('Evaluation Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Spatial visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"Error creating spatial visualization: {e}")


# Export main classes
__all__ = ['SpatialValidator', 'SpatialRelation']


