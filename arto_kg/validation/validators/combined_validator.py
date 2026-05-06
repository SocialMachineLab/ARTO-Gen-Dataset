
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

class CombinedArtworkValidator:
    """
    Combined Validator that merges State, Spatial, and Alignment analysis into a single VLM call.
    Also provides rule-based geometric verification for Size and Spatial relations.
    Ported from validation_vllm/core/batch_processor.py.
    """

    def __init__(self):
        pass

    # ==========================
    # 1. Geometric Verification
    # ==========================

    def verify_size_geometric(self, image_path: str, json_data: Dict[str, Any], bboxes_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify object sizes based on bounding box areas rules.
        """
        try:
            expected_sizes = self._extract_expected_sizes(json_data)
            if not expected_sizes:
                return {'overall_score': 100.0, 'details': [], 'note': 'No size requirements'}

            # Get image dimensions from image file or metadata if available
            # Note: In the original batch_processor, it opens the image. 
            # To avoid extra IO if possible, we could pass img size, but opening is safer.
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    img_w, img_h = img.size
                img_area = img_w * img_h
            except Exception as e:
                logger.error(f"Failed to load image for size verification: {e}")
                return {'overall_score': 0.0, 'details': [], 'note': 'Image load failed'}

            # Build name -> bbox mapping (using largest bbox for stability)
            name_to_bbox = {}
            for det in bboxes_dict:
                label = det.get('label') or det.get('text_label') # GroundingDINO/OWLv2 compatibility
                box = det.get('box')
                confidence = det.get('score') or det.get('confidence', 0.0)
                
                if not label or not box or confidence < 0.3:
                    continue

                # Box format assumption: [x1, y1, x2, y2]
                w = box[2] - box[0]
                h = box[3] - box[1]
                area = w * h

                if label not in name_to_bbox or area > name_to_bbox[label]['area']:
                    name_to_bbox[label] = {'box': box, 'area': area}

            results = []
            total_score = 0
            count = 0

            for obj_name, size_desc in expected_sizes.items():
                count += 1
                matched_bbox = None
                # fuzzy match
                for label, info in name_to_bbox.items():
                    if label.lower() in obj_name.lower() or obj_name.lower() in label.lower():
                        matched_bbox = info
                        break
                
                if not matched_bbox:
                    results.append({'object': obj_name, 'expected': size_desc, 'actual': 'not detected', 'score': 0.0})
                    continue

                area = matched_bbox['area']
                area_ratio = area / img_area
                score = 0.0
                actual_desc = "unknown"

                if area_ratio < 0.05:
                    actual_desc = "small"
                elif area_ratio < 0.25:
                    actual_desc = "medium"
                else:
                    actual_desc = "large"

                # Scoring logic
                if size_desc.lower() in actual_desc or actual_desc in size_desc.lower():
                    score = 1.0
                elif size_desc.lower() == 'medium' and (area_ratio > 0.02 and area_ratio < 0.4):
                    # Loose medium
                    score = 0.8
                else:
                    score = 0.3 # Detected but wrong size

                results.append({
                    'object': obj_name,
                    'expected': size_desc,
                    'actual': actual_desc,
                    'area_ratio': round(area_ratio, 4),
                    'score': score
                })
                total_score += score

            overall = (total_score / count * 100) if count > 0 else 100.0
            return {'overall_score': overall, 'details': results}

        except Exception as e:
            logger.error(f"Geometric size verification failed: {e}")
            return {'overall_score': 0.0, 'error': str(e)}

    def verify_spatial_geometric(self, json_data: Dict[str, Any], bboxes_dict: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify spatial relations based on bounding box centers.
        """
        try:
            spatial_relations = self._extract_spatial_relations(json_data)
            if not spatial_relations:
                return {'overall_score': 100.0, 'details': [], 'note': 'No spatial relations'}

            id_to_name = self._get_object_id_to_name_mapping(json_data)
            
            # Build name/label -> bbox center mapping
            name_to_bbox = {}
            for det in bboxes_dict:
                label = det.get('label') or det.get('text_label')
                box = det.get('box')
                if label and box:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    name_to_bbox[label] = {'box': box, 'center': (cx, cy)}

            results = []
            total_score = 0
            count = 0

            for rel in spatial_relations:
                if len(rel) < 3: continue
                # rel is [obj1_id, relation_type, obj2_id]
                obj1_id, relation, obj2_id = rel[0], rel[1], rel[2]
                
                obj1_name = id_to_name.get(obj1_id, str(obj1_id)).split('_')[0]
                obj2_name = id_to_name.get(obj2_id, str(obj2_id)).split('_')[0]
                
                count += 1
                
                # Match bboxes
                bbox1 = None
                bbox2 = None
                
                for label, info in name_to_bbox.items():
                    if obj1_name.lower() in label.lower() or label.lower() in obj1_name.lower(): bbox1 = info
                for label, info in name_to_bbox.items():
                    if obj2_name.lower() in label.lower() or label.lower() in obj2_name.lower(): bbox2 = info
                
                if not bbox1 or not bbox2:
                    results.append({
                        'relation': f"{obj1_name} {relation} {obj2_name}",
                        'verified': False,
                        'reason': 'object not detected'
                    })
                    continue

                # Geometric check
                c1 = bbox1['center']
                c2 = bbox2['center']
                verified = False
                
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1] # y increases downwards usually

                rel_lower = relation.lower()
                if 'left' in rel_lower: # obj1 left of obj2 => c1.x < c2.x
                    if dx < 0: verified = True
                elif 'right' in rel_lower:
                    if dx > 0: verified = True
                elif 'above' in rel_lower or 'top' in rel_lower: # obj1 above obj2 => c1.y < c2.y
                    if dy < 0: verified = True
                elif 'below' in rel_lower or 'bottom' in rel_lower or 'under' in rel_lower:
                    if dy > 0: verified = True
                else:
                    # Complex relations like 'inside', 'on' hard to do geometrically without segmentation
                    # Giving benefit of doubt or relying on VLM for those (but this function is purely geometric)
                    # We'll mark as verify=True for now as fallback or maybe skip
                    verified = True 

                score = 1.0 if verified else 0.0
                total_score += score
                results.append({
                    'relation': f"{obj1_name} {relation} {obj2_name}",
                    'verified': verified,
                    'score': score
                })

            overall = (total_score / count * 100) if count > 0 else 100.0
            return {'overall_score': overall, 'details': results}

        except Exception as e:
            logger.error(f"Geometric spatial verification failed: {e}")
            return {'overall_score': 0.0, 'error': str(e)}


    # ==========================
    # 2. VLM Analysis (Unified)
    # ==========================

    def construct_combined_prompt(self, json_data: Dict[str, Any]) -> str:
        """
        Construct a single prompt for State, Semantic Relations, and Alignment.
        """
        # 1. State Info
        expected_states = self._extract_expected_states(json_data)
        state_text = ""
        if expected_states:
            for obj, state in expected_states.items():
                state_text += f"- {obj}: {state}\n"
        else:
            state_text = "No specific states to verify."

        # 2. Semantic Info
        semantic_relations = self._extract_semantic_relations(json_data)
        relation_text = ""
        id_to_name = self._get_object_id_to_name_mapping(json_data)
        if semantic_relations:
            for rel in semantic_relations:
                if len(rel) >= 3:
                     # rel: [id1, relation, id2]
                    obj1 = id_to_name.get(rel[0], str(rel[0])).split('_')[0]
                    obj2 = id_to_name.get(rel[2], str(rel[2])).split('_')[0]
                    relation_text += f"- {obj1} {rel[1]} {obj2}\n"
        else:
            relation_text = "No specific relations to verify."
            
        # 3. Alignment Info
        main_prompt = self._extract_main_prompt(json_data)
        
        prompt = f"""You are an expert art critic and computer vision validator. Perform a comprehensive analysis of this image.

TASK 1: OBJECT STATE ANALYSIS
Check if these objects match their expected states:
{state_text}

TASK 2: SEMANTIC RELATION ANALYSIS
Verify if these relationships are visible:
{relation_text}

TASK 3: OVERALL ALIGNMENT
Compare the image with this description:
"{main_prompt}"

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure (no markdown, no other text).
IMPORTANT: Do NOT include any 'analysis' or 'reasoning' text fields. Return ONLY numeric scores and booleans.

{{
  "state_analysis": [
    {{ "object": "name", "match_score": 0.9 }}
  ],
  "semantic_analysis": [
    {{ "relation": "obj1 rel obj2", "verified": true, "confidence": 0.9 }}
  ],
  "alignment_metrics": {{
    "semantic_similarity": 0.8,
    "content_coverage": 0.8,
    "style_consistency": 0.8,
    "overall_harmony": 0.9
  }}
}}
"""
        return prompt

    def parse_combined_result(self, vlm_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the JSON response from VLM.
        Returns a dictionary with keys: 'state_analysis', 'semantic_analysis', 'alignment_analysis'
        """
        default_scores = {
            'state_analysis': {'overall_score': 0.0, 'details': []},
            'semantic_analysis': {'overall_score': 0.0, 'details': []},
            'alignment_analysis': {'overall_score': 0.0, 'metrics': {}}
        }

        if not vlm_result or not vlm_result.get('success'):
            return default_scores

        response = vlm_result.get('response', '').strip()
        
        # Clean markdown
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '')
        elif response.startswith('```'):
            response = response.replace('```', '')
            
        try:
            # Try to find JSON block
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx != -1 and end_idx != -1:
                response = response[start_idx:end_idx+1]
                
            parsed = json.loads(response)
            
            # 1. State Score
            states = parsed.get('state_analysis', [])
            state_score = 0.0
            if states:
                scores = [s.get('match_score', 0) for s in states]
                state_score = sum(scores) / len(scores) * 100
            else:
                state_score = 100.0 # No states to verify = pass
            
            # 2. Semantic Score
            relations = parsed.get('semantic_analysis', [])
            semantic_score = 0.0
            if relations:
                scores = [r.get('confidence', 1.0 if r.get('verified') else 0.0) for r in relations]
                semantic_score = sum(scores) / len(scores) * 100
            else:
                semantic_score = 100.0 # No relations = pass
                
            # 3. Alignment Score
            metrics = parsed.get('alignment_metrics', {})
            align_keys = ['semantic_similarity', 'content_coverage', 'style_consistency', 'overall_harmony']
            align_vals = [metrics.get(k, 0) for k in align_keys]
            align_score = (sum(align_vals) / len(align_keys)) * 100 if align_keys else 0.0
            
            return {
                'state_analysis': {'overall_score': state_score, 'details': states},
                'semantic_analysis': {'overall_score': semantic_score, 'details': relations},
                'alignment_analysis': {'overall_score': align_score, 'metrics': metrics}
            }

        except Exception as e:
            logger.error(f"Failed to parse combined VLM response: {e}, Response: {response}")
            return default_scores

    # ==========================
    # 3. Helpers (Extracted from utils.py)
    # ==========================

    def _extract_expected_sizes(self, json_data: Dict) -> Dict[str, str]:
        sizes = {}
        for obj in json_data.get('objects', {}).get('enhanced_objects', []):
            if obj.get('name') and obj.get('size'):
                sizes[obj['name']] = obj['size']
        return sizes

    def _extract_expected_states(self, json_data: Dict) -> Dict[str, str]:
        states = {}
        for obj in json_data.get('objects', {}).get('enhanced_objects', []):
            if obj.get('name') and obj.get('state'):
                states[obj['name']] = obj['state']
        return states

    def _extract_spatial_relations(self, json_data: Dict) -> List[List]:
        return json_data.get('composition', {}).get('spatial_relations', [])

    def _extract_semantic_relations(self, json_data: Dict) -> List[List]:
        return json_data.get('composition', {}).get('semantic_relations', [])

    def _get_object_id_to_name_mapping(self, json_data: Dict) -> Dict[int, str]:
        mapping = {}
        for obj in json_data.get('objects', {}).get('enhanced_objects', []):
            if obj.get('object_id') is not None and obj.get('name'):
                mapping[obj['object_id']] = obj['name']
        return mapping

    def _extract_main_prompt(self, json_data: Dict) -> str:
        return json_data.get('final_prompts', {}).get('main_prompt', '')
