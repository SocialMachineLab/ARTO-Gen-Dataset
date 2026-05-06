"""
State Evaluation Module
Specializes in object state analysis and verification
"""

import cv2
import numpy as np
import json
import os
import re
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class StateValidator:
    """Object State Validator"""
    
    def __init__(self, vlm_wrapper=None):
        """
        Initialize State Validator
        
        Args:
            vlm_wrapper: VLM wrapper for state recognition
        """
        self.vlm_wrapper = vlm_wrapper
        
        # Common state categories
        self.state_categories = {
            'position': ['sitting', 'standing', 'lying', 'flying', 'floating', 'hanging'],
            'activity': ['running', 'walking', 'jumping', 'sleeping', 'eating', 'drinking'],
            'condition': ['open', 'closed', 'broken', 'intact', 'full', 'empty'],
            'appearance': ['bright', 'dim', 'colorful', 'faded', 'shiny', 'dull'],
            'movement': ['moving', 'stationary', 'rotating', 'swaying', 'falling'],
            'interaction': ['touching', 'separate', 'connected', 'adjacent', 'distant']
        }
        
        # State confidence threshold
        self.confidence_threshold = 0.6
    
    def extract_expected_states(self, json_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Extract expected object states - Supports new JSON structure"""
        expected_states = {}
        
        try:
            # Extract state information from enhanced_objects (new structure)
            enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
            for obj in enhanced_objects:
                if isinstance(obj, dict):
                    obj_name = obj.get('name', '')
                    if not obj_name:
                        continue
                    
                    # Directly extract state field (new structure)
                    if 'state' in obj:
                        state_info = obj['state']
                        expected_states[obj_name] = {
                            'description': str(state_info),
                            'category': self._categorize_state(str(state_info)),
                            'confidence_required': 0.7,
                            'source': 'direct_state_field'
                        }
                    
                    # Infer state from artistic_description
                    if obj_name not in expected_states and 'artistic_description' in obj:
                        artistic_desc = obj['artistic_description']
                        inferred_states = self._extract_states_from_description(artistic_desc)
                        if inferred_states:
                            expected_states[obj_name] = {
                                'description': inferred_states,
                                'category': self._categorize_state(inferred_states),
                                'confidence_required': 0.6,
                                'source': 'artistic_description'
                            }
                    
                    # Infer potential state from material
                    if 'material' in obj:
                        material_state = self._infer_state_from_material(obj['material'], obj_name)
                        if material_state and obj_name not in expected_states:
                            expected_states[obj_name] = {
                                'description': material_state,
                                'category': 'condition',
                                'confidence_required': 0.5,
                                'source': 'material_inference'
                            }
            
            # Compatible with old structure - Extract from direct objects array
            objects = json_data.get('objects', [])
            if isinstance(objects, list):
                for obj in objects:
                    if isinstance(obj, dict):
                        obj_name = obj.get('name', '')
                        if not obj_name:
                            continue
                        
                        if 'state' in obj:
                            state_info = obj['state']
                            expected_states[obj_name] = {
                                'description': str(state_info),
                                'category': self._categorize_state(str(state_info)),
                                'confidence_required': 0.7,
                                'source': 'legacy_state_field'
                            }
                        
                        # Infer state from other fields
                        if obj_name not in expected_states:
                            if 'position' in obj:
                                expected_states[obj_name] = {
                                    'description': f"positioned {obj['position']}",
                                    'category': 'position',
                                    'confidence_required': 0.6,
                                    'source': 'position_field'
                                }
                            elif 'activity' in obj:
                                activity_desc = obj['activity']
                                expected_states[obj_name] = {
                                    'description': str(activity_desc),
                                    'category': 'activity',
                                    'confidence_required': 0.8,
                                    'source': 'activity_field'
                                }
            
            # Extract position state from composition
            composition = json_data.get('composition', {})
            if isinstance(composition, dict):
                spatial_rels = composition.get('spatial_relationships', {})
                if isinstance(spatial_rels, dict):
                    object_positions = spatial_rels.get('object_positions', {})
                    for obj_name, pos_info in object_positions.items():
                        if isinstance(pos_info, dict) and obj_name not in expected_states:
                            position = pos_info.get('position', '')
                            if position:
                                expected_states[obj_name] = {
                                    'description': f"positioned in {position}",
                                    'category': 'position',
                                    'confidence_required': 0.6,
                                    'source': 'composition_position'
                                }
        
        except Exception as e:
            logger.warning(f"Failed to extract expected states: {e}")
        
        return expected_states
    
    def _extract_states_from_description(self, description: str) -> str:
        """Extract state information from artistic description"""
        desc_lower = description.lower()
        
        # State keywords
        state_keywords = {
            'position': ['standing', 'sitting', 'lying', 'positioned', 'placed', 'resting'],
            'activity': ['running', 'walking', 'flying', 'swimming', 'eating', 'drinking'],
            'condition': ['alive', 'vibrant', 'pristine', 'polished', 'weathered', 'aged', 'broken', 'intact'],
            'appearance': ['bright', 'dim', 'shining', 'glowing', 'faded', 'colorful', 'naturalistic'],
            'movement': ['dynamic', 'static', 'moving', 'stationary', 'flowing', 'posed']
        }
        
        found_states = []
        for category, keywords in state_keywords.items():
            for keyword in keywords:
                if keyword in desc_lower:
                    found_states.append(keyword)
        
        return ', '.join(found_states) if found_states else 'naturalistic rendering'
    
    def _infer_state_from_material(self, material: str, obj_name: str) -> Optional[str]:
        """Infer object state from material"""
        material_lower = material.lower()
        obj_lower = obj_name.lower()
        
        # Material state inference
        material_states = {
            'steel': 'polished, metallic finish',
            'wood': 'natural grain, smooth finish', 
            'fur': 'soft, natural texture',
            'hair': 'flowing, natural',
            'glass': 'transparent, reflective',
            'metal': 'lustrous, solid',
            'fabric': 'draped, textured',
            'leather': 'supple, worn',
            'stone': 'solid, weathered',
            'crystal': 'clear, prismatic'
        }
        
        for mat, state in material_states.items():
            if mat in material_lower:
                return state
        
        # Default state based on object type
        if any(animal in obj_lower for animal in ['horse', 'zebra', 'elephant', 'lion', 'tiger', 'cat', 'dog']):
            return 'alive, alert, naturalistic'
        elif any(tool in obj_lower for tool in ['knife', 'sword', 'tool']):
            return 'ready for use, well-maintained'
        elif any(vessel in obj_lower for vessel in ['cup', 'glass', 'bottle', 'bowl']):
            return 'empty, clean, positioned'
        
        return None
    
    def _categorize_state(self, state_description: str) -> str:
        """Categorize state description"""
        state_desc_lower = state_description.lower()

        # Priority check: More specific state categories
        category_keywords = {
            'condition': ['alive', 'vibrant', 'pristine', 'polished', 'ready', 'intact', 'broken', 'weathered', 'aged'],
            'appearance': ['bright', 'dim', 'shiny', 'dull', 'colorful', 'faded', 'naturalistic', 'realistic'],
            'position': ['standing', 'sitting', 'lying', 'positioned', 'placed', 'resting'],
            'activity': ['running', 'walking', 'flying', 'moving', 'eating', 'drinking'],
            'movement': ['dynamic', 'static', 'flowing', 'posed', 'moving', 'stationary'],
            'interaction': ['touching', 'separate', 'connected', 'adjacent', 'distant']
        }

        # Check by priority
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in state_desc_lower:
                    return category

        # If no match, use original logic
        for category, states in self.state_categories.items():
            for state in states:
                if state in state_desc_lower:
                    return category

        return 'condition'  # Default to condition instead of general
    
    def analyze_states_with_vlm(self, image_path: str, 
                              bboxes_dict: Dict[str, List[int]], 
                              expected_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze object states using VLM"""
        if self.vlm_wrapper is None:
            return {
                'score': 50.0,
                'note': 'VLM not available, using basic state analysis',
                'method': 'fallback',
                'average_confidence': 0.5
            }
        
        if not expected_states:
            return {
                'score': 80.0,
                'note': 'No state expectations to analyze',
                'method': 'no_expectations',
                'average_confidence': 0.8
            }
        
        try:
            state_analysis_results = {}
            state_scores = []
            
            logger.info(f"Starting VLM state analysis for image: {image_path}")
            for obj_name, state_info in expected_states.items():
                if obj_name not in bboxes_dict:
                    logger.info(f"Object '{obj_name}' not found in bounding boxes, skipping.")
                    continue
                
                bbox = bboxes_dict[obj_name]
                expected_description = state_info['description']
                category = state_info['category']
                
                # Analyze single object state
                analysis_result = self._analyze_single_object_state_vlm(
                    image_path, bbox, obj_name, expected_description, category
                )
                
                state_analysis_results[obj_name] = analysis_result
                state_scores.append(analysis_result.get('match_score', 0.5))
            
            overall_score = np.mean(state_scores) * 100 if state_scores else 50.0
            
            # Calculate average confidence, avoid NaN
            confidences = [r.get('confidence', 0.5) for r in state_analysis_results.values()]
            average_confidence = float(np.mean(confidences)) if confidences else 0.5
            
            return {
                'score': overall_score,
                'detailed_results': state_analysis_results,
                'objects_analyzed': len(state_analysis_results),
                'average_confidence': average_confidence,
                'method': 'vlm_analysis'
            }
            
        except Exception as e:
            logger.error(f"VLM state analysis failed: {e}")
            return {
                'score': 30.0,
                'error': str(e),
                'method': 'vlm_failed',
                'average_confidence': 0.3
            }
    
    def _analyze_single_object_state_vlm(self, image_path: str, bbox: List[int], 
                                       obj_name: str, expected_description: str,
                                       category: str) -> Dict[str, Any]:
        """Analyze single object state using VLM"""
        try:
            # Build VLM prompt with JSON format for reliable parsing
            prompt = f"""
Carefully observe the {obj_name} in the image (located approximately at coordinates {bbox}).

Expected State Description: {expected_description}
State Category: {category}

Please analyze the actual state of this object and evaluate how well it matches the expected state.

Please respond strictly in the following JSON format:
```json
{{
  "actual_state": "[detailed description of the object's actual state]",
  "match_score": [value between 0-1],
  "analysis": "[detailed explanation of why you gave this score]",
  "state_matches": [true/false]
}}
```

Notes:
- actual_state should provide a detailed description of the state you observe in the image
- match_score ranges from 0-1, where 1 indicates a perfect match
- analysis should explain the reasoning behind your score
- state_matches indicates whether the state basically matches the expected state
"""
            
            # Call VLM for actual analysis
            vlm_result = self.vlm_wrapper.generate_response(prompt, image_path)
            
            if vlm_result.get('success', False):
                response = vlm_result.get('response', '')
                logger.info(f"🔍 VLM State Analysis for {obj_name}:")
                logger.info(f"   Response length: {len(response)} chars")
                logger.info(f"   Response preview: {response[:200]}...")
                logger.debug(f"   Full response: {response}")
            else:
                # Fallback response when VLM call fails
                logger.warning(f"VLM call failed for {obj_name}: {vlm_result.get('error', 'Unknown error')}")
                response = f"""```json
{{
  "actual_state": "{obj_name} state analysis failed, using fallback",
  "match_score": 0.4,
  "analysis": "VLM unavailable, estimating based on geometry and expected description",
  "state_matches": false
}}
```"""
            
            # Parse VLM response
            return self._parse_state_analysis_response(response, expected_description)
            
        except Exception as e:
            logger.error(f"Single object state analysis failed for {obj_name}: {e}")
            return {
                'match_score': 0.3,
                'confidence': 0.3,
                'actual_state': 'analysis_failed',
                'error': str(e)
            }
    
    def _parse_state_analysis_response(self, response: str, expected_description: str) -> Dict[str, Any]:
        """Parse VLM state analysis response"""
        try:
            # First try extracting JSON from response
            json_response = self._extract_json_from_response(response)

            if json_response:
                actual_state = json_response.get('actual_state', 'unknown')
                match_score = float(json_response.get('match_score', 0.5))
                analysis = json_response.get('analysis', '')
                state_matches = json_response.get('state_matches', False)

                # Ensure match_score is within reasonable range
                match_score = max(0.0, min(1.0, match_score))

                # Calculate confidence
                confidence = min(match_score + 0.1, 1.0)

                logger.info(f"✅ Successfully parsed JSON VLM response: {actual_state[:50]}...")

                return {
                    'actual_state': actual_state,
                    'expected_state': expected_description,
                    'match_score': match_score,
                    'confidence': confidence,
                    'analysis': analysis,
                    'state_matches': state_matches,
                    'status': 'success'
                }

            # If JSON parsing fails, try legacy format parsing
            logger.warning("JSON parsing failed, trying fallback text parsing...")
            return self._parse_legacy_response(response, expected_description)

        except Exception as e:
            logger.error(f"Failed to parse VLM response: {e}")
            logger.debug(f"Raw response: {response[:200]}...")
            return {
                'match_score': 0.3,
                'confidence': 0.3,
                'actual_state': 'parse_failed',
                'analysis': f'Response parsing failed: {str(e)}',
                'error': str(e)
            }

    def _extract_json_from_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract JSON data from response - Enhanced version supporting multiple formats"""
        try:
            import re
            import json

            # Pattern 1: Standard JSON code block ```json {...} ```
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)

            # Pattern 2: Code block without language tag ``` {...} ```
            json_match = re.search(r'```\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)

            # Pattern 3: Find JSON object containing actual_state (compact format)
            json_match = re.search(r'\{[^{}]*"actual_state"[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                return json.loads(json_str)
            
            # Pattern 4: Multi-line JSON object (looser matching, supports nesting)
            json_match = re.search(r'\{[\s\S]*?"actual_state"[\s\S]*?\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Clean potential comments
                json_str = re.sub(r'//.*?\n', '', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

            # Pattern 5: Entire response is JSON
            if response.strip().startswith('{') and response.strip().endswith('}'):
                return json.loads(response.strip())

            return None

        except Exception as e:
            logger.debug(f"JSON extraction failed: {e}")
            return None

    def _parse_legacy_response(self, response: str, expected_description: str) -> Dict[str, Any]:
        """Parse legacy format response (fallback method)"""
        lines = response.strip().split('\n')

        actual_state = 'unknown'
        match_score = 0.5
        analysis = ''

        for line in lines:
            line = line.strip()
            if line.startswith('Actual state:'):
                actual_state = line.split(':', 1)[1]
            elif line.startswith('Match score:'):
                score_text = line.split('：', 1)[1] if '：' in line else line.split(':', 1)[1]
                try:
                    match_score = float(score_text.strip())
                except:
                    match_score = 0.5
            elif line.startswith('Analysis:'):
                analysis = line.split(':', 1)[1]

        # Calculate confidence
        confidence = min(match_score + 0.1, 1.0)

        return {
            'actual_state': actual_state,
            'expected_state': expected_description,
            'match_score': match_score,
            'confidence': confidence,
            'analysis': analysis,
            'status': 'success'
        }
    
    def analyze_states_fallback(self, bboxes_dict: Dict[str, List[int]], 
                              expected_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback state analysis method (without VLM)"""
        if not expected_states:
            return {
                'score': 80.0,
                'note': 'No state expectations to analyze',
                'method': 'fallback_no_expectations',
                'average_confidence': 0.8
            }
        
        try:
            # Simple heuristic-based state analysis
            fallback_results = {}
            scores = []
            
            for obj_name, state_info in expected_states.items():
                if obj_name not in bboxes_dict:
                    continue
                
                bbox = bboxes_dict[obj_name]
                expected_description = state_info['description']
                category = state_info['category']
                
                # Simple geometric analysis
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                aspect_ratio = width / height if height > 0 else 1
                
                # Infer state based on geometric features
                estimated_score = self._estimate_state_from_geometry(
                    aspect_ratio, category, expected_description
                )
                
                scores.append(estimated_score)
                
                fallback_results[obj_name] = {
                    'estimated_state': f"geometry-based estimation for {category}",
                    'match_score': estimated_score,
                    'confidence': 0.4,  # Lower confidence
                    'method': 'geometric_heuristic',
                    'bbox_aspect_ratio': aspect_ratio
                }
            
            overall_score = np.mean(scores) * 100 if scores else 60.0
            
            # Calculate average confidence, avoid NaN
            confidences = [r.get('confidence', 0.4) for r in fallback_results.values()]
            average_confidence = float(np.mean(confidences)) if confidences else 0.4
            
            return {
                'score': overall_score,
                'detailed_results': fallback_results,
                'objects_analyzed': len(fallback_results),
                'average_confidence': average_confidence,
                'method': 'fallback_geometric',
                'note': 'Used geometric heuristics due to VLM unavailability'
            }
            
        except Exception as e:
            logger.error(f"Fallback state analysis failed: {e}")
            return {
                'score': 40.0,
                'error': str(e),
                'method': 'fallback_failed',
                'average_confidence': 0.3
            }
    
    def _estimate_state_from_geometry(self, aspect_ratio: float, category: str, description: str) -> float:
        """Estimate state match score based on geometric features"""
        description_lower = description.lower()
        
        # Simple heuristic based on aspect ratio
        if category == 'position':
            if 'standing' in description_lower and aspect_ratio < 0.8:
                return 0.7
            elif 'lying' in description_lower and aspect_ratio > 1.2:
                return 0.7
            elif 'sitting' in description_lower and 0.8 <= aspect_ratio <= 1.2:
                return 0.6
        
        elif category == 'activity':
            # Activity state is hard to judge from geometric features
            return 0.5
        
        elif category == 'condition':
            # Condition state requires semantic understanding
            return 0.4
        
        # Default medium match
        return 0.5
    
    def create_state_visualization(self, evaluation_result: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> Optional[str]:
        """Create state analysis visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'State Analysis Results (Score: {evaluation_result.get("overall_score", 0):.1f}/100)', 
                        fontsize=16, fontweight='bold')
            
            # 1. Object state match scores
            state_analysis = evaluation_result.get('state_analysis', {})
            if 'detailed_results' in state_analysis:
                objects = list(state_analysis['detailed_results'].keys())
                match_scores = [state_analysis['detailed_results'][obj].get('match_score', 0) * 100 
                              for obj in objects]
                
                colors = ['green' if s > 70 else 'orange' if s > 50 else 'red' for s in match_scores]
                bars = axes[0, 0].bar(objects, match_scores, color=colors, alpha=0.7)
                
                # Add score labels
                for bar, score in zip(bars, match_scores):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2, height + 1,
                                   f'{score:.1f}', ha='center', va='bottom', fontsize=10)
                
                axes[0, 0].set_ylim(0, 105)
                axes[0, 0].set_ylabel('Match Score (%)')
                axes[0, 0].set_title('State Match Scores')
                axes[0, 0].tick_params(axis='x', rotation=45)
            else:
                axes[0, 0].text(0.5, 0.5, 'No state analysis data', ha='center', va='center', 
                               transform=axes[0, 0].transAxes)
                axes[0, 0].set_title('State Match Scores')
            
            # 2. Confidence distribution
            if 'detailed_results' in state_analysis:
                confidences = [state_analysis['detailed_results'][obj].get('confidence', 0) 
                             for obj in state_analysis['detailed_results'].keys()]
                
                if confidences:
                    axes[0, 1].hist(confidences, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[0, 1].set_xlabel('Confidence')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].set_title('Confidence Distribution')
                    axes[0, 1].axvline(np.mean(confidences), color='red', linestyle='--', 
                                      label=f'Mean: {np.mean(confidences):.2f}')
                    axes[0, 1].legend()
            else:
                axes[0, 1].text(0.5, 0.5, 'No confidence data', ha='center', va='center', 
                               transform=axes[0, 1].transAxes)
                axes[0, 1].set_title('Confidence Distribution')
            
            # 3. Analysis method pie chart
            method = state_analysis.get('method', 'unknown')
            method_labels = {'vlm_analysis': 'VLM Analysis', 'fallback_geometric': 'Geometric Fallback', 
                           'fallback_no_expectations': 'No Expectations', 'vlm_failed': 'VLM Failed'}
            method_name = method_labels.get(method, method)
            
            axes[1, 0].pie([1], labels=[method_name], autopct='', startangle=90,
                          colors=['lightgreen' if 'vlm' in method else 'lightcoral'])
            axes[1, 0].set_title('Analysis Method')
            
            # 4. Evaluation summary
            summary_text = f"""Overall Score: {evaluation_result.get('overall_score', 0):.1f}/100

Analysis Method: {method_name}
Objects Analyzed: {state_analysis.get('objects_analyzed', 0)}
Average Confidence: {state_analysis.get('average_confidence', 0):.2f}

VLM Available: {"Yes" if self.vlm_wrapper else "No"}"""
            
            if 'note' in state_analysis:
                summary_text += f"\n\nNote: {state_analysis['note']}"
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            axes[1, 1].set_title('State Analysis Summary')
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
            logger.error(f"State visualization failed: {e}")
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
        """Main state evaluation method"""
        try:
            logger.info(f"Starting state validation for: {image_path}")
            
            # 1. Extract expected states
            expected_states = self.extract_expected_states(json_data)
            
            # 2. Select analysis method
            if self.vlm_wrapper is not None and expected_states:
                state_analysis = self.analyze_states_with_vlm(
                    image_path, bboxes_dict, expected_states
                )
            else:
                state_analysis = self.analyze_states_fallback(
                    bboxes_dict, expected_states
                )
            
            overall_score = state_analysis.get('score', 50.0)
            
            # 3. Generate visualization
            viz_path = None
            result = {
                'overall_score': overall_score,
                'dimension_scores': {
                    'state_consistency': overall_score
                },
                'state_analysis': state_analysis,
                'evaluation_summary': {
                    'total_objects': len(bboxes_dict),
                    'objects_with_state_expectations': len(expected_states),
                    'vlm_available': self.vlm_wrapper is not None,
                    'analysis_method': state_analysis.get('method', 'unknown'),
                    'expected_states': expected_states
                },
                'image_path': image_path,
                'visualization_path': viz_path # This will be updated below
            }

            if visualize:
                output_dir = os.path.join(os.path.dirname(image_path), "state_analysis")
                os.makedirs(output_dir, exist_ok=True)
                
                viz_path = os.path.join(output_dir, f"state_analysis_{os.path.splitext(os.path.basename(image_path))[0]}.png")
                self.create_state_visualization(result, viz_path)
            
            # Update result with visualization path (or None)
            result['visualization_path'] = viz_path
            
            logger.info(f"State validation completed. Overall score: {overall_score:.1f}/100")
            
            return result
            
        except Exception as e:
            logger.error(f"State validation failed: {e}")
            return {
                'error': str(e),
                'overall_score': 0.0,
                'image_path': image_path
            }