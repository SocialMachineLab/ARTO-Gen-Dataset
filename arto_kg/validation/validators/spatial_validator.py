"""Spatial validation: checks declared spatial relations against detected boxes.

Layout composition metrics are computed and reported alongside, but do not
enter the score.
"""

import cv2
import numpy as np
import json
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import math

from .. import spatial_predicates as sp
from .. import instance_assignment as ia

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)


@dataclass
class SpatialRelation:
    """A spatial relation between two objects."""
    object1: str
    object2: str
    relation_type: str
    confidence: float
    geometric_relation: str
    distance: float
    description: str


class SpatialValidator:
    """Spatial relation validator."""

    def __init__(self, vlm_wrapper=None):
        """Initialize thresholds and keyword tables; `vlm_wrapper` is optional."""
        self.vlm_wrapper = vlm_wrapper
        self.position_threshold = 50  # Pixel threshold
        
        # Relative-position pair to compass direction
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
        """Compute pairwise geometric relations from boxes {key: [x1, y1, x2, y2]}."""
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
        """Describe direction, distance, overlap and size between two boxes."""
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
        """Classify x1 relative to x2 within the position threshold."""
        if x1 < x2 - self.position_threshold:
            return "left_of"
        elif x1 > x2 + self.position_threshold:
            return "right_of"
        else:
            return "aligned"
    
    def _get_vertical_relation(self, y1: float, y2: float) -> str:
        """Classify y1 relative to y2 within the position threshold."""
        if y1 < y2 - self.position_threshold:
            return "above"
        elif y1 > y2 + self.position_threshold:
            return "below"
        else:
            return "aligned"
    
    def _calculate_overlap(self, bbox1: List[int], bbox2: List[int]) -> Dict[str, float]:
        """Return overlap area and its ratio to the smaller box."""
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
        """Collect declared positions, relationships and composition notes.

        Of spatial_relationships, scene_framework and spatial_layout, uses the
        first whose parsing_details status is success, else the first that has
        no parsing_details and holds object_positions or object_relationships.
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
            composition = json_data.get('composition', {})
            
            # Priority: spatial_relationships > scene_framework > spatial_layout
            spatial_data_sources = [
                ('spatial_relationships', composition.get('spatial_relationships', {})),
                ('scene_framework', composition.get('scene_framework', {})),
                ('spatial_layout', composition.get('spatial_layout', {}))
            ]
            
            best_spatial_data = None
            best_source_name = None
            
            # A source that parsed successfully, else the first without parsing
            # details that holds positions or relationships
            for source_name, spatial_data in spatial_data_sources:
                if isinstance(spatial_data, dict) and spatial_data:
                    if 'parsing_details' in spatial_data:
                        parsing_details = spatial_data['parsing_details']
                        if isinstance(parsing_details, dict) and parsing_details.get('status') == 'success':
                            best_spatial_data = spatial_data
                            best_source_name = source_name
                            break
                    elif 'object_positions' in spatial_data or 'object_relationships' in spatial_data:
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
                                
                                complex_relation = self._identify_complex_relation(relationship)
                                if complex_relation:
                                    expected_relations['complex_relations'][relation_key] = complex_relation
                
                # Extract composition type
                composition_type = best_spatial_data.get('composition_type', '')
                if composition_type:
                    expected_relations['composition_rules'].append(f"Composition type: {composition_type}")
                
                # Extract extra info from a non-trivial raw_output
                raw_output = best_spatial_data.get('raw_output', '')
                if raw_output and isinstance(raw_output, str) and len(raw_output) > 50:
                    additional_info = self._parse_raw_spatial_output(raw_output)
                    if additional_info:
                        expected_relations['composition_rules'].append(f"Additional spatial info: {additional_info}")
            
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
        """Summarise layout info from raw_output (JSON fields or keywords)."""
        try:
            if raw_output.strip().startswith('{'):
                import json
                parsed = json.loads(raw_output)
                if isinstance(parsed, dict):
                    useful_info = []
                    if 'composition_type' in parsed:
                        useful_info.append(f"type: {parsed['composition_type']}")
                    if 'layout_style' in parsed:
                        useful_info.append(f"style: {parsed['layout_style']}")
                    return ', '.join(useful_info) if useful_info else None
        except:
            pass
        
        # Fall back to keyword matching
        spatial_keywords = ['balanced', 'symmetric', 'asymmetric', 'centered', 'triangular', 'linear', 'clustered']
        found_keywords = [kw for kw in spatial_keywords if kw.lower() in raw_output.lower()]
        return ', '.join(found_keywords) if found_keywords else None
    
    def _identify_complex_relation(self, description: str) -> Optional[Dict[str, Any]]:
        """Return the complex relation type named in a description, if any."""
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
        """Score layout composition: balance, symmetry, thirds, weight, spacing."""
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
        """Score how evenly area-weighted objects are distributed."""
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
        """Score left-right mirror symmetry of object centres."""
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
        
        symmetry_score = horizontal_symmetry * 100
        
        return {
            'score': symmetry_score,
            'horizontal_symmetry': horizontal_symmetry,
            'symmetry_matches': symmetry_matches,
            'total_points': len(centers)
        }
    
    def _analyze_rule_of_thirds(self, centers: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Score proximity of object centres to thirds lines and intersections."""
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
        """Score the spread of object areas."""
        # Calculate standard deviation of area distribution (uniformity metric)
        area_std = np.std(areas)
        area_mean = np.mean(areas)
        area_cv = area_std / area_mean if area_mean > 0 else 0  # Coefficient of variation
        
        # Target CV: a mix of large and small objects
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
        """Score the regularity of pairwise centre distances."""
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
        
        # Good spacing is fairly uniform but not identical (moderate CV)
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
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str,
                       bboxes_dict: Dict[str, List[int]],
                       visualize: bool = False) -> Dict[str, Any]:
        """Run the full spatial evaluation."""
        return self.comprehensive_spatial_evaluation(json_data, image_path, bboxes_dict, visualize=visualize)
    
    def comprehensive_spatial_evaluation(self, json_data: Dict[str, Any], 
                                       image_path: str,
                                       bboxes_dict: Dict[str, List[int]],
                                       visualize: bool = False) -> Dict[str, Any]:
        """Verify declared relations and report layout metrics alongside.

        The overall score is the declared-relation verification score only.
        """
        print(f"Starting spatial relationship evaluation for: {os.path.basename(image_path)}")
        
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
            
            # 4. Verify the declared relations against the boxes
            print("   Step 4: Verifying declared spatial relations...")
            verification = self.verify_declared_relations(json_data, bboxes_dict)

            # 5. Compile the result. Layout composition is reported separately
            # and does not affect the score.
            overall_score = verification['score']

            evaluation_result = {
                # None when no relation was checkable.
                'overall_score': overall_score,
                'verification': verification,
                'dimension_scores': {
                    'declared_relations': overall_score,
                    'layout_composition': layout_analysis.get('composition_score'),
                },
                'detailed_results': {
                    'expected_relations': expected_relations,
                    'geometric_relations': geometric_relations,
                    'layout_analysis': layout_analysis,
                },
                'evaluation_summary': {
                    'total_objects': len(bboxes_dict),
                    'declared_relations': verification['declared'],
                    'checked': verification['checked'],
                    'verified': verification['verified'],
                    'deferred_to_vlm': verification['deferred_to_vlm'],
                    'not_checked': verification['not_checked'],
                    'unsupported': verification['unsupported'],
                    'vlm_available': self.vlm_wrapper is not None,
                },
                'image_file': image_path,
                'artwork_id': json_data.get('artwork_id', 'unknown')
            }

            if visualize:
                output_dir = os.path.join(os.path.dirname(image_path), "spatial_analysis")
                os.makedirs(output_dir, exist_ok=True)
                viz_path = os.path.join(
                    output_dir, f"spatial_analysis_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                evaluation_result['visualization_path'] = self.create_spatial_visualization(
                    evaluation_result, viz_path)

            if overall_score is None:
                print("   Spatial evaluation completed. No relation was checkable.")
            else:
                print(f"   Spatial evaluation completed. Verified "
                      f"{verification['verified']}/{verification['checked']} "
                      f"({overall_score:.1f}%)")
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Spatial relationship evaluation failed: {e}"
            print(f"   {error_msg}")
            return {'error': error_msg}
    
    def verify_declared_relations(self, json_data: Dict[str, Any],
                                  bboxes_dict: Dict[str, List[int]]) -> Dict[str, Any]:
        """Check each declared spatial relation against the two boxes.

        Relations needing depth, occlusion or part-whole reasoning are deferred
        to the VLM; relations without both boxes are `not_checked`. Neither
        counts as a failure.
        """
        composition = json_data.get('composition') or {}
        declared = composition.get('spatial_relations') or []

        names = ia.names_by_instance(json_data)

        def box_for(endpoint):
            """Look up a box by instance id, falling back to object name.

            The name fallback covers triples that name objects directly.
            """
            if endpoint is None:
                return None
            box = bboxes_dict.get(endpoint)
            if box is not None:
                return box
            key = str(endpoint).lower()
            for instance_id, name in names.items():
                if str(name).lower() == key:
                    return bboxes_dict.get(instance_id)
            return None

        results = []
        for relation in declared:
            if isinstance(relation, dict):
                subject = relation.get('subject')
                predicate = relation.get('relation', '')
                obj = relation.get('object')
            elif isinstance(relation, (list, tuple)) and len(relation) == 3:
                # Triple layout: [subject, "relation", object].
                subject, predicate, obj = relation
            else:
                continue

            subject_box, object_box = box_for(subject), box_for(obj)
            record = sp.evaluate(predicate, subject_box, object_box)
            record.update({
                'subject': subject,
                'object': obj,
                # Carried so the VLM channel reuses the boxes resolved here.
                'subject_box': subject_box,
                'object_box': object_box,
                'subject_name': names.get(subject, str(subject)),
                'object_name': names.get(obj, str(obj)),
            })
            results.append(record)

        summary = sp.summarise(results)
        summary['declared'] = len(declared)
        return summary

    def create_spatial_visualization(self, evaluation_result: Dict[str, Any],
                                   save_path: Optional[str] = None) -> Optional[str]:
        """Plot the spatial result; save to `save_path` and return it, or show it."""
        if 'error' in evaluation_result:
            print(f"Cannot create spatial visualization: {evaluation_result['error']}")
            return None

        try:
            score = evaluation_result.get('overall_score')
            summary = evaluation_result.get('evaluation_summary', {})
            verification = evaluation_result.get('verification', {})
            layout = evaluation_result.get('detailed_results', {}).get('layout_analysis', {})

            fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            fig.suptitle('Spatial Relations (score: ' + ('-' if score is None else f'{score:.1f}/100') + ')',
                         fontsize=16, fontweight='bold')

            # 1. Declared relations by outcome
            checked = summary.get('checked', 0)
            verified = summary.get('verified', 0)
            labels = ['verified', 'not verified', 'deferred to VLM', 'not checked', 'unsupported']
            counts = [verified, checked - verified, summary.get('deferred_to_vlm', 0),
                      summary.get('not_checked', 0), summary.get('unsupported', 0)]
            axes[0].barh(labels, counts, color=['green', 'red', 'orange', 'grey', 'lightgrey'], alpha=0.7)
            axes[0].invert_yaxis()
            axes[0].set_xlabel('Relations')
            axes[0].set_title('Declared relations')

            # 2. Layout metrics (reported, not scored)
            aspects = [('Balance', 'balance_analysis'), ('Symmetry', 'symmetry_analysis'),
                       ('Rule of thirds', 'rule_of_thirds'), ('Visual weight', 'visual_weight'),
                       ('Spacing', 'spacing_analysis')]
            if 'error' not in layout and all(key in layout for _, key in aspects):
                axes[1].barh([name for name, _ in aspects],
                             [layout[key].get('score', 0) for _, key in aspects], alpha=0.7)
                axes[1].invert_yaxis()
                axes[1].set_xlim(0, 100)
            else:
                axes[1].text(0.5, 0.5, 'No layout data', ha='center', va='center',
                             transform=axes[1].transAxes)
            axes[1].set_title('Layout (not scored)')

            # 3. Each checked relation
            lines = []
            for record in verification.get('details', [])[:15]:
                status = record.get('status')
                outcome = ('ok' if record.get('verified') else 'no') if status == 'checked' else status
                lines.append(f"{record.get('subject_name')} {record.get('relation', '')} "
                             f"{record.get('object_name')}: {outcome}")
            axes[2].text(0.02, 0.98, "\n".join(lines) or "No declared relations",
                         transform=axes[2].transAxes, fontsize=10, verticalalignment='top',
                         family='monospace')
            axes[2].set_title('Relations')
            axes[2].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                return save_path
            plt.show()
            return None

        except Exception as e:
            print(f"Error creating spatial visualization: {e}")
            return None


__all__ = ['SpatialValidator', 'SpatialRelation']


