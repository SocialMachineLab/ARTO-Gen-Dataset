   
            

"""
Size Relation Evaluation Module
Specializes in object size analysis and size relation validation
"""

import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class SizeValidator:
    """Object size relation validator"""
    
    def __init__(self):
        """Initialize size validator"""
        
        # standard object sizes (relative to human size 1.0)
        self.standard_object_sizes = {
            # Animals
            'person': 1.0, 'human': 1.0, 'man': 1.0, 'woman': 1.0,
            'cat': 0.3, 'dog': 0.4, 'horse': 1.2, 'elephant': 2.5,
            'bird': 0.2, 'eagle': 0.5, 'butterfly': 0.05, 'zebra': 1.1,
            'lion': 1.0, 'tiger': 1.0, 'bear': 1.3, 'wolf': 0.8,
            'rabbit': 0.3, 'fox': 0.4, 'deer': 1.0, 'sheep': 0.8,
            
            # Vehicles
            'car': 1.5, 'truck': 2.5, 'bicycle': 1.0, 'motorcycle': 1.2,
            'airplane': 10.0, 'boat': 3.0, 'ship': 15.0,
            
            # Buildings
            'house': 5.0, 'building': 8.0, 'tower': 15.0, 'bridge': 10.0,
            'church': 12.0, 'castle': 20.0, 'temple': 15.0,
            
            # Natural objects
            'tree': 3.0, 'flower': 0.1, 'mountain': 20.0, 'rock': 1.0,
            'sun': 50.0, 'moon': 30.0, 'cloud': 5.0,
            'forest': 25.0, 'river': 8.0, 'lake': 10.0,
            
            # Daily items
            'chair': 0.8, 'table': 1.2, 'book': 0.15, 'cup': 0.08,
            'phone': 0.06, 'laptop': 0.3, 'tv': 1.5, 'knife': 0.2,
            'fork': 0.15, 'spoon': 0.18, 'plate': 0.25, 'bowl': 0.2,
            'microwave': 0.4, 'refrigerator': 2.0, 'oven': 1.5,
            'bed': 2.0, 'sofa': 2.5, 'lamp': 1.2, 'clock': 0.3,
            'vase': 0.4, 'painting': 1.0, 'sculpture': 1.5,
            
            # Clothing and accessories
            'hat': 0.3, 'shirt': 0.8, 'dress': 1.0, 'shoes': 0.3,
            'gloves': 0.2, 'belt': 0.4, 'scarf': 0.5,
            
            # Food
            'apple': 0.1, 'banana': 0.15, 'bread': 0.3, 'cake': 0.4,
            'cheese': 0.2, 'wine': 0.3, 'beer': 0.2,
            
            # Tools and equipment
            'hammer': 0.4, 'screwdriver': 0.3, 'wrench': 0.35,
            'camera': 0.25, 'telescope': 0.8, 'microscope': 0.5,
            
            # Weapons
            'sword': 1.2, 'dagger': 0.3, 'bow': 1.5, 'arrow': 0.8,
            
            # Musical instruments
            'guitar': 1.0, 'piano': 1.8, 'violin': 0.6, 'drum': 1.2,
            
            # Others
            'fireplace': 2.0, 'fountain': 3.0, 'statue': 2.5,
            'fence': 1.5, 'gate': 2.0, 'window': 1.0, 'door': 2.0
        }
        
        # Size descriptor to value mapping
        self.size_descriptors = {
            'tiny': 0.1, 'very_small': 0.2, 'small': 0.3,
            'medium': 0.5, 'normal': 0.5, 'average': 0.5,
            'large': 0.8, 'big': 0.8, 'very_large': 1.0,
            'huge': 1.2, 'giant': 1.5, 'enormous': 2.0
        }
    
    def extract_expected_size_relations(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract expected size relations - Supports new JSON structure"""
        size_relations = {
            'absolute_sizes': {},
            'relative_sizes': {},
            'size_constraints': []
        }
        
        try:
            # Extract size info from enhanced_objects (new structure)
            enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
            for obj in enhanced_objects:
                if isinstance(obj, dict):
                    obj_name = obj.get('name', '')
                    if not obj_name:
                        continue
                    
                    # Absolute size (new structure: directly in object)
                    if 'size' in obj:
                        size_relations['absolute_sizes'][obj_name] = obj['size']
                    
                    # Relative size relations
                    if 'size_relation' in obj:
                        size_relations['relative_sizes'][obj_name] = obj['size_relation']
                    
                    # Size constraints
                    if 'size_constraint' in obj:
                        size_relations['size_constraints'].append({
                            'object': obj_name,
                            'constraint': obj['size_constraint']
                        })
                    
                    # Infer size info from artistic_description
                    artistic_desc = obj.get('artistic_description', '')
                    if artistic_desc:
                        inferred_size = self._infer_size_from_description(artistic_desc, obj_name)
                        if inferred_size and obj_name not in size_relations['absolute_sizes']:
                            size_relations['absolute_sizes'][obj_name] = inferred_size
            
            # Compatible with old structure - Extract from direct objects array
            objects = json_data.get('objects', [])
            if isinstance(objects, list):
                for obj in objects:
                    if isinstance(obj, dict):
                        obj_name = obj.get('name', '')
                        if not obj_name:
                            continue
                        
                        # Absolute size
                        if 'size' in obj:
                            size_relations['absolute_sizes'][obj_name] = obj['size']
                        
                        # Relative size relations
                        if 'size_relation' in obj:
                            size_relations['relative_sizes'][obj_name] = obj['size_relation']
            
            # Extract spatial size relations from composition
            composition = json_data.get('composition', {})
            if isinstance(composition, dict):
                # Extract size relations from spatial_relationships
                spatial_rels = composition.get('spatial_relationships', {})
                if isinstance(spatial_rels, dict):
                    object_positions = spatial_rels.get('object_positions', {})
                    for obj_name, pos_info in object_positions.items():
                        if isinstance(pos_info, dict) and 'size_in_composition' in pos_info:
                            # If no absolute size info yet, use size from composition
                            if obj_name not in size_relations['absolute_sizes']:
                                size_relations['absolute_sizes'][obj_name] = pos_info['size_in_composition']
            
            # Extract from global size rules
            if 'size_rules' in json_data:
                rules = json_data['size_rules']
                if isinstance(rules, dict):
                    size_relations.update(rules)
        
        except Exception as e:
            logger.warning(f"Failed to extract size relations: {e}")
        
        return size_relations
    
    def _infer_size_from_description(self, description: str, obj_name: str) -> Optional[str]:
        """Infer object size from artistic description"""
        desc_lower = description.lower()
        obj_lower = obj_name.lower()
        
        # Size keyword mapping
        size_keywords = {
            'tiny': ['tiny', 'minute', 'minuscule'],
            'small': ['small', 'little', 'compact', 'delicate'],
            'medium': ['medium', 'moderate', 'average'],
            'large': ['large', 'big', 'substantial', 'prominent', 'majestic', 'noble'],
            'huge': ['huge', 'massive', 'enormous', 'gigantic', 'towering', 'dominant']
        }
        
        # Check for size keywords in description
        for size, keywords in size_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    return size.title()  # Convert to title case: Large, Small, etc.
        
        # Infer based on standard object size
        standard_sizes = {
            'elephant': 'Large',
            'horse': 'Large', 
            'zebra': 'Large',
            'lion': 'Large',
            'tiger': 'Large',
            'person': 'Medium',
            'human': 'Medium',
            'dog': 'Medium',
            'cat': 'Small',
            'bird': 'Small',
            'knife': 'Small',
            'spoon': 'Small',
            'fork': 'Small',
            'cup': 'Small',
            'wine glass': 'Small',
            'bottle': 'Small',
            'book': 'Small',
            'chair': 'Medium',
            'table': 'Large',
            'building': 'Huge',
            'tree': 'Large',
            'flower': 'Small'
        }
        
        # Check if object name is in standard sizes
        for obj_type, size in standard_sizes.items():
            if obj_type in obj_lower:
                return size
        
        return None
    
    def calculate_object_areas(self, bboxes_dict: Dict[str, List[int]]) -> Dict[str, float]:
        """Calculate object bounding box areas"""
        areas = {}
        
        for obj_name, bbox in bboxes_dict.items():
            if len(bbox) >= 4:
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                area = width * height
                areas[obj_name] = max(area, 1)  # Avoid division by zero
            else:
                areas[obj_name] = 1
        
        return areas
    
    def analyze_absolute_sizes(self, bboxes_dict: Dict[str, List[int]], 
                             expected_sizes: Dict[str, str]) -> Dict[str, Any]:
        """Analyze absolute size consistency"""
        if not expected_sizes:
            return {'score': 80.0, 'note': 'No absolute size expectations'}
        
        try:
            areas = self.calculate_object_areas(bboxes_dict)
            
            if not areas:
                return {'score': 50.0, 'error': 'No valid bounding boxes'}
            
            # Use max area as baseline
            max_area = max(areas.values())
            
            size_analysis = {}
            consistency_scores = []
            
            for obj_name, expected_size in expected_sizes.items():
                if obj_name not in areas:
                    continue
                
                # Actual relative size
                actual_relative_size = areas[obj_name] / max_area
                
                # Expected relative size (mapped from size descriptor)
                expected_relative_size = self.size_descriptors.get(
                    str(expected_size).lower(), 0.5
                )
                
                # Standard object size (from predefined dict)
                standard_size = self.standard_object_sizes.get(
                    obj_name.lower(), 0.5
                )
                
                # Combined expected size (combining three factors)
                # 1. User specified size description weight 40%
                # 2. Standard object size weight 40%
                # 3. Default value 20%
                combined_expectation = (
                    expected_relative_size * 0.4 + 
                    standard_size * 0.4 + 
                    0.5 * 0.2
                )
                
                # Calculate consistency score
                size_ratio = min(actual_relative_size, combined_expectation) / max(actual_relative_size, combined_expectation)
                consistency_score = size_ratio * 100
                
                consistency_scores.append(consistency_score)
                
                size_analysis[obj_name] = {
                    'actual_relative_size': actual_relative_size,
                    'expected_size': expected_size,
                    'expected_relative_size': expected_relative_size,
                    'standard_size': standard_size,
                    'combined_expectation': combined_expectation,
                    'consistency_score': consistency_score
                }
            
            overall_score = np.mean(consistency_scores) if consistency_scores else 50.0
            
            return {
                'score': overall_score,
                'detailed_analysis': size_analysis,
                'average_consistency': overall_score / 100,
                'objects_analyzed': len(size_analysis)
            }
            
        except Exception as e:
            logger.error(f"Absolute size analysis failed: {e}")
            return {'score': 30.0, 'error': str(e)}
    
    def analyze_relative_sizes(self, bboxes_dict: Dict[str, List[int]], 
                             relative_relations: Dict[str, str]) -> Dict[str, Any]:
        """Analyze relative size relations"""
        if not relative_relations:
            return {'score': 80.0, 'note': 'No relative size relations to analyze'}
        
        try:
            areas = self.calculate_object_areas(bboxes_dict)
            
            relation_scores = []
            detailed_relations = {}
            
            for obj_name, relation_desc in relative_relations.items():
                if obj_name not in areas:
                    continue
                
                # Parse relation description
                relation_info = self._parse_size_relation(relation_desc)
                if not relation_info:
                    continue
                
                target_obj = relation_info.get('target_object')
                relation_type = relation_info.get('relation_type')
                
                if target_obj not in areas:
                    continue
                
                # Calculate actual size ratio
                obj_area = areas[obj_name]
                target_area = areas[target_obj]
                actual_ratio = obj_area / target_area
                
                # Determine expected ratio based on relation type
                expected_ratio = self._get_expected_size_ratio(relation_type)
                
                # Calculate match score
                ratio_match = min(actual_ratio, expected_ratio) / max(actual_ratio, expected_ratio)
                match_score = ratio_match * 100
                
                relation_scores.append(match_score)
                
                detailed_relations[f"{obj_name}_{target_obj}"] = {
                    'object1': obj_name,
                    'object2': target_obj,
                    'relation_type': relation_type,
                    'actual_ratio': actual_ratio,
                    'expected_ratio': expected_ratio,
                    'match_score': match_score
                }
            
            overall_score = np.mean(relation_scores) if relation_scores else 80.0
            
            return {
                'score': overall_score,
                'detailed_relations': detailed_relations,
                'relations_analyzed': len(detailed_relations),
                'average_match_score': overall_score / 100
            }
            
        except Exception as e:
            logger.error(f"Relative size analysis failed: {e}")
            return {'score': 30.0, 'error': str(e)}
    
    def _parse_size_relation(self, relation_desc: str) -> Optional[Dict[str, str]]:
        """Parse size relation description"""
        relation_desc = str(relation_desc).lower()
        
        # Simple relation parsing
        if 'larger than' in relation_desc or 'bigger than' in relation_desc:
            parts = relation_desc.split('than')
            if len(parts) > 1:
                return {
                    'relation_type': 'larger_than',
                    'target_object': parts[1].strip()
                }
        
        elif 'smaller than' in relation_desc or 'littler than' in relation_desc:
            parts = relation_desc.split('than')
            if len(parts) > 1:
                return {
                    'relation_type': 'smaller_than',
                    'target_object': parts[1].strip()
                }
        
        elif 'same size as' in relation_desc or 'equal to' in relation_desc:
            parts = relation_desc.replace('same size as', 'as').replace('equal to', 'as').split('as')
            if len(parts) > 1:
                return {
                    'relation_type': 'equal_to',
                    'target_object': parts[1].strip()
                }
        
        return None
    
    def _get_expected_size_ratio(self, relation_type: str) -> float:
        """Get expected size ratio"""
        ratios = {
            'much_larger_than': 3.0,
            'larger_than': 1.5,
            'slightly_larger_than': 1.2,
            'equal_to': 1.0,
            'slightly_smaller_than': 0.8,
            'smaller_than': 0.6,
            'much_smaller_than': 0.3
        }
        
        return ratios.get(relation_type, 1.0)
    
    def create_size_visualization(self, evaluation_result: Dict[str, Any], 
                                save_path: Optional[str] = None) -> Optional[str]:
        """Create size analysis visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Size Analysis Results (Score: {evaluation_result.get("overall_score", 0):.1f}/100)', 
                        fontsize=16, fontweight='bold')
            
            # 1. Absolute size consistency
            abs_analysis = evaluation_result.get('absolute_size_analysis', {})
            if 'detailed_analysis' in abs_analysis:
                objects = list(abs_analysis['detailed_analysis'].keys())
                consistency_scores = [abs_analysis['detailed_analysis'][obj]['consistency_score'] 
                                    for obj in objects]
                
                colors = ['green' if s > 70 else 'orange' if s > 50 else 'red' for s in consistency_scores]
                axes[0, 0].bar(objects, consistency_scores, color=colors, alpha=0.7)
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].set_ylabel('Consistency Score')
                axes[0, 0].set_title('Absolute Size Consistency')
                axes[0, 0].tick_params(axis='x', rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, 'No absolute size data', ha='center', va='center', 
                               transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('Absolute Size Consistency')
            
            # 2. Relative size relations
            rel_analysis = evaluation_result.get('relative_size_analysis', {})
            if 'detailed_relations' in rel_analysis:
                relations = list(rel_analysis['detailed_relations'].keys())[:5]  # Top 5
                match_scores = [rel_analysis['detailed_relations'][rel]['match_score'] 
                              for rel in relations]
                
                colors = ['green' if s > 70 else 'orange' if s > 50 else 'red' for s in match_scores]
                axes[0, 1].barh(relations, match_scores, color=colors, alpha=0.7)
                axes[0, 1].set_xlim(0, 100)
                axes[0, 1].set_xlabel('Match Score')
                axes[0, 1].set_title('Relative Size Relations')
            else:
                axes[0, 1].text(0.5, 0.5, 'No relative size data', ha='center', va='center', 
                               transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Relative Size Relations')
            
            # 3. Score distribution
            scores = []
            if 'absolute_size_analysis' in evaluation_result:
                scores.append(evaluation_result['absolute_size_analysis'].get('score', 0))
            if 'relative_size_analysis' in evaluation_result:
                scores.append(evaluation_result['relative_size_analysis'].get('score', 0))
            
            if scores:
                labels = ['Absolute Size', 'Relative Size']
                colors = ['lightblue', 'lightgreen']
                axes[1, 0].pie(scores, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                axes[1, 0].set_title('Score Distribution')
            else:
                axes[1, 0].text(0.5, 0.5, 'No score data', ha='center', va='center', 
                               transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Score Distribution')
            
            # 4. Evaluation summary
            summary_text = f"""Overall Score: {evaluation_result.get('overall_score', 0):.1f}/100

Absolute Size Score: {abs_analysis.get('score', 0):.1f}/100
Relative Size Score: {rel_analysis.get('score', 0):.1f}/100

Objects Analyzed: {abs_analysis.get('objects_analyzed', 0)}
Relations Analyzed: {rel_analysis.get('relations_analyzed', 0)}"""
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[1, 1].set_title('Size Analysis Summary')
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
            logger.error(f"Size visualization failed: {e}")
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
                       bboxes_dict: Dict[str, List[int]],
                       visualize: bool = False) -> Dict[str, Any]:
        """Main size evaluation method"""
        try:
            logger.info(f"Starting size validation for: {image_path}")
            
            # 1. Extract expected size relations
            expected_size_relations = self.extract_expected_size_relations(json_data)
            
            # 2. Automatically infer missing relative size relations
            self._infer_missing_relative_relations(expected_size_relations, json_data)
            
            # 3. Analyze absolute size consistency
            absolute_analysis = self.analyze_absolute_sizes(
                bboxes_dict, expected_size_relations.get('absolute_sizes', {})
            )
            
            # 4. Analyze relative size relations
            relative_analysis = self.analyze_relative_sizes(
                bboxes_dict, expected_size_relations.get('relative_sizes', {})
            )
            
            # 5. Calculate overall score
            abs_score = absolute_analysis.get('score', 50.0)
            rel_score = relative_analysis.get('score', 50.0)
            
            # Weights: Absolute size 60%, Relative size 40%
            overall_score = abs_score * 0.6 + rel_score * 0.4
            
            # 6. Generate visualization
            viz_path = None
            if visualize:
                output_dir = os.path.join(os.path.dirname(image_path), "size_analysis")
                os.makedirs(output_dir, exist_ok=True)
                
                viz_path = os.path.join(output_dir, f"size_analysis_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                self.create_size_visualization(result, viz_path)
            
            # Update result with visualization path (or None)
            result = {
                'overall_score': overall_score,
                'dimension_scores': {
                    'absolute_size_consistency': abs_score,
                    'relative_size_relations': rel_score
                },
                'absolute_size_analysis': absolute_analysis,
                'relative_size_analysis': relative_analysis,
                'evaluation_summary': {
                    'total_objects': len(bboxes_dict),
                    'objects_with_size_expectations': len(expected_size_relations.get('absolute_sizes', {})),
                    'relative_relations_count': len(expected_size_relations.get('relative_sizes', {})),
                    'expected_relations': expected_size_relations
                },
                'image_path': image_path,
                'visualization_path': viz_path
            }
            
            logger.info(f"Size validation completed. Overall score: {overall_score:.1f}/100")
            
            return result
            
        except Exception as e:
            logger.error(f"Size validation failed: {e}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'image_path': image_path
            }
    
    def _infer_missing_relative_relations(self, size_relations: Dict[str, Any], json_data: Dict[str, Any]):
        """Automatically infer missing relative size relations"""
        try:
            absolute_sizes = size_relations.get('absolute_sizes', {})
            relative_sizes = size_relations.get('relative_sizes', {})
            
            # If no relative size relations, infer from absolute sizes
            if not relative_sizes and len(absolute_sizes) > 1:
                object_names = list(absolute_sizes.keys())
                
                # Create relative relations for each pair of objects
                for i in range(len(object_names)):
                    for j in range(i + 1, len(object_names)):
                        obj1 = object_names[i]
                        obj2 = object_names[j]
                        
                        # Get size descriptions for both objects
                        size1 = str(absolute_sizes.get(obj1, 'medium')).lower()
                        size2 = str(absolute_sizes.get(obj2, 'medium')).lower()
                        
                        # Infer relative relation
                        relation = self._infer_relative_size_relation(size1, size2)
                        if relation:
                            # Add bidirectional relations
                            if obj1 not in relative_sizes:
                                relative_sizes[obj1] = f"{relation} than {obj2}"
                            if obj2 not in relative_sizes:
                                inverse_relation = self._get_inverse_relation(relation)
                                relative_sizes[obj2] = f"{inverse_relation} than {obj1}"
            
            size_relations['relative_sizes'] = relative_sizes
            
        except Exception as e:
            logger.warning(f"Failed to infer relative relations: {e}")
    
    def _infer_relative_size_relation(self, size1: str, size2: str) -> Optional[str]:
        """Infer relative relation from two size descriptions"""
        # Size rank mapping
        size_ranking = {
            'tiny': 1, 'very_small': 2, 'small': 3,
            'medium': 4, 'normal': 4, 'average': 4,
            'large': 5, 'big': 5, 'very_large': 6,
            'huge': 7, 'giant': 8, 'enormous': 9
        }
        
        rank1 = size_ranking.get(size1, 4)
        rank2 = size_ranking.get(size2, 4)
        
        if rank1 > rank2 + 1:
            return "larger"
        elif rank1 < rank2 - 1:
            return "smaller"
        else:
            return None  # Similar size, do not establish relation
    
    def _get_inverse_relation(self, relation: str) -> str:
        """Get inverse relation"""
        inverse_map = {
            'larger': 'smaller',
            'smaller': 'larger',
            'bigger': 'smaller',
            'tinier': 'larger'
        }
        return inverse_map.get(relation, 'different')
    
    
    