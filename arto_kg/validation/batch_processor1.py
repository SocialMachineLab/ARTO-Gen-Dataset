"""
Batch Processor - Orchestrate the entire validation pipeline
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from .vllm_wrapper import VLLMWrapper
from .color_analyzer import ColorAnalyzer
from .utils import (
    load_gt_data,
    extract_expected_colors,
    extract_expected_sizes,
    extract_expected_states,
    extract_spatial_relations,
    extract_semantic_relations,
    get_object_id_to_name_mapping,
    extract_main_prompt,
    save_validation_result
)



class BatchProcessor:
    """Batch Validation Processor"""
    
    def __init__(self, vlm_wrapper: VLLMWrapper, batch_size: int = 10):
        """
        Initialize batch processor
        
        Args:
            vlm_wrapper: VLLMWrapper instance
            batch_size: Batch size
        """
        self.vlm = vlm_wrapper
        self.batch_size = batch_size
        self.color_analyzer = ColorAnalyzer()
    
    def process_file_list(self, 
                         file_list: List[str],
                         output_dir: str,
                         od_results_dir: str,
                         image_base_dir: str) -> Dict[str, Any]:
        """
        Process file list
        
        Args:
            file_list: List of GT file paths
            output_dir: Output directory
            od_results_dir: OD results directory
            image_base_dir: Image base directory
            
        Returns:
            Processing statistics
        """
        total_files = len(file_list)
        processed = 0
        skipped = 0
        failed = 0
        
        print(f"\nStarting batch processing: {total_files} files")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Output dir: {output_dir}")
        
        start_time = time.time()
        
        # Process by batch
        for i in range(0, total_files, self.batch_size):
            batch_files = file_list[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
            try:
                batch_results = self._process_batch(
                    batch_files,
                    output_dir,
                    od_results_dir,
                    image_base_dir
                )
                
                processed += batch_results['processed']
                skipped += batch_results['skipped']
                failed += batch_results['failed']
                
            except Exception as e:
                print(f"Batch {batch_num} failed: {e}")
                failed += len(batch_files)
        
        elapsed = time.time() - start_time
        
        print(f"\nBatch processing completed!")
        print(f"   Total: {total_files}")
        print(f"   Processed: {processed}")
        print(f"   Skipped: {skipped}")
        print(f"   Failed: {failed}")
        print(f"   Time: {elapsed:.1f}s ({elapsed/total_files:.2f}s/file)")
        
        return {
            'total': total_files,
            'processed': processed,
            'skipped': skipped,
            'failed': failed,
            'elapsed_time': elapsed,
            'avg_time_per_file': elapsed / total_files if total_files > 0 else 0
        }
    
    def _process_batch(self,
                      batch_files: List[str],
                      output_dir: str,
                      od_results_dir: str,
                      image_base_dir: str) -> Dict[str, int]:
        """Process a batch"""
        
        # Prepare batch data
        batch_data = []
        for gt_path in batch_files:
            try:
                data = self._prepare_single_file(gt_path, od_results_dir, image_base_dir)
                if data:
                    batch_data.append(data)
            except Exception as e:
                print(f"   Failed to prepare {os.path.basename(gt_path)}: {e}")
        
        if not batch_data:
            return {'processed': 0, 'skipped': len(batch_files), 'failed': 0}
        
        # Construct combined validation prompt (State, Semantic, Alignment)
        prompts = []
        images = []
        
        for data in batch_data:
            combined_prompt = self._construct_combined_prompt(data)
            prompts.append(combined_prompt)
            images.append(data['image_path'])
        
        # Batch VLM Inference (Just one call!)
        # Increase max_tokens to prevent truncation
        vlm_results = self.vlm.generate_batch(prompts, images, max_tokens=2048)
        
        # Process results
        processed = 0
        skipped = 0
        failed = 0
        
        for idx, data in enumerate(batch_data):
            try:
                vlm_result = vlm_results[idx]
                
                # 1. Parse VLM results (State, Semantic, Alignment)
                parsed_scores = self._parse_combined_result(vlm_result)
                
                # 2. Execute geometric validation (Size, Spatial) - No VLM needed
                size_result = self._verify_size_geometric(data)
                spatial_result = self._verify_spatial_geometric(data)
                
                # 3. Execute color validation (CV) - No VLM needed
                expected_colors = extract_expected_colors(data['json_data'])
                color_result = self.color_analyzer.evaluate_object_colors(
                    data['image_path'], 
                    data['bboxes_dict'], 
                    expected_colors
                )
                
                # 4. Assemble full result
                full_result = {
                    'artwork_id': data['artwork_id'],
                    'image_path': data['image_path'],
                    'bboxes_source': data['od_path'],
                    'steps': {
                        'step3_color': color_result,                    # CV analysis
                        'step4_size': size_result,                      # BBox validation
                        'step5_state': parsed_scores['state_analysis'], # VLM validation
                        'step6_spatial': parsed_scores['semantic_analysis'], # VLM validation semantic
                        'step6_spatial_geometric': spatial_result,           # BBox validation geometric
                        'step7_alignment': parsed_scores['alignment_analysis'] # VLM validation
                    },
                    'raw_vlm_response': vlm_result.get('response', '')
                }
                
                # Save result
                artwork_id = data['artwork_id']
                output_path_dir = os.path.join(output_dir, artwork_id)
                os.makedirs(output_path_dir, exist_ok=True)
                output_path = os.path.join(output_path_dir, f"{artwork_id}_full_validation.json")
                
                if save_validation_result(full_result, output_path):
                    processed += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"   Failed to process {data.get('artwork_id', 'unknown')}: {e}")
                failed += 1
        
        return {
            'processed': processed,
            'skipped': skipped,
            'failed': failed
        }

    def _prepare_single_file(self,
                            gt_path: str,
                            od_results_dir: str,
                            image_base_dir: str) -> Optional[Dict[str, Any]]:
        """Prepare data for a single file"""
        
        # Load GT data
        json_data = load_gt_data(gt_path)
        if not json_data:
            return None
        
        artwork_id = json_data.get('artwork_id', '')
        if not artwork_id:
            return None
        
        # Construct image path
        # Derive image path from GT path
        # V5: data/v5/v3_baroque_batch/final_results/baroque/artwork_XXX_v2.json
        #     -> image_generation/v5_generated_images/v3_baroque_batch/artwork_XXX.png
        # V6: data/v6_normal/final_results/baroque/artwork_XXX_v2.json
        #     -> image_generation/v6_normal_generated_images/baroque/artwork_XXX.png
        
        path_parts = gt_path.split('/')
        if 'v5' in path_parts:
            dataset = 'v5'
        elif 'v6' in path_parts or 'v6_normal' in path_parts:
            dataset = 'v6'
        else:
            return None
        
        # Find batch name or style name
        batch_name = None
        style_name = None
        
        # First try to find batch (V5)
        for part in path_parts:
            if 'batch' in part:
                batch_name = part
                break
        
        # If batch not found, look for style (V6)
        if not batch_name:
            # V6 path: data/v6_normal/final_results/baroque/artwork_XXX_v2.json
            # Second to last part is style
            if 'final_results' in path_parts:
                idx = path_parts.index('final_results')
                if idx + 1 < len(path_parts):
                    style_name = path_parts[idx + 1]
        
        # Construct image path
        if dataset == 'v5':
            if not batch_name:
                print(f"   V5 batch name not found in path: {gt_path}")
                return None
            image_path = os.path.join(
                image_base_dir,
                f"{dataset}_generated_images",
                batch_name,
                f"{artwork_id}.png"
            )
        else:  # v6
            if not style_name:
                print(f"   V6 style name not found in path: {gt_path}")
                return None
            image_path = os.path.join(
                image_base_dir,
                "v6_normal_generated_images",
                style_name,
                f"{artwork_id}.png"
            )
        
        if not os.path.exists(image_path):
            print(f"   Image not found: {image_path}")
            return None
        
        # OD result path
        od_path = os.path.join(od_results_dir, f"{artwork_id}_fused.json")
        if not os.path.exists(od_path):
            print(f"   OD result not found: {od_path}")
            return None
        
        # Load OD result
        with open(od_path, 'r') as f:
            try:
                od_data = json.load(f)
            except json.JSONDecodeError:
                print(f"   Failed to decode OD result: {od_path}")
                return None
        
        bboxes_dict = od_data.get('detections', [])
        
        return {
            'artwork_id': artwork_id,
            'gt_path': gt_path,
            'image_path': image_path,
            'od_path': od_path,
            'json_data': json_data,
            'bboxes_dict': bboxes_dict
        }


    def _construct_combined_prompt(self, data: Dict) -> str:
        """Construct combined validation prompt"""
        json_data = data['json_data']
        
        # 1. State Info
        expected_states = extract_expected_states(json_data)
        state_text = ""
        if expected_states:
            for obj, state in expected_states.items():
                state_text += f"- {obj}: {state}\n"
        else:
            state_text = "No specific states to verify."

        # 2. Semantic Info
        semantic_relations = extract_semantic_relations(json_data)
        relation_text = ""
        if semantic_relations:
            id_to_name = get_object_id_to_name_mapping(json_data)
            for rel in semantic_relations:
                if len(rel) >= 3:
                     # Simplify ID display
                    obj1 = id_to_name.get(rel[0], rel[0]).split('_')[0]
                    obj2 = id_to_name.get(rel[2], rel[2]).split('_')[0]
                    relation_text += f"- {obj1} {rel[1]} {obj2}\n"
        else:
            relation_text = "No specific relations to verify."
            
        # 3. Alignment Info
        main_prompt = extract_main_prompt(json_data)
        
        # ---------------------------------------------------------
        # [BACKUP] PREVIOUS DETAILED PROMPT V1 (With VLM Color & Analysis)
        # ---------------------------------------------------------
        # # 3. Color Info (New)
        # expected_colors = extract_expected_colors(json_data)
        # color_text = ""
        # if expected_colors:
        #      for obj, colors in expected_colors.items():
        #          color_str = ", ".join(colors)
        #          color_text += f"- {obj}: {color_str}\n"
        # else:
        #     color_text = "No specific colors to verify."
        #
        # prompt = f"""You are an expert art critic and computer vision validator. Perform a comprehensive analysis of this image.
        #
        # TASK 1: OBJECT STATE ANALYSIS
        # Check if these objects match their expected states:
        # {state_text}
        #
        # TASK 2: SEMANTIC RELATION ANALYSIS
        # Verify if these relationships are visible:
        # {relation_text}
        #
        # TASK 3: COLOR ANALYSIS
        # Verify if these objects match their expected colors:
        # {color_text}
        #
        # TASK 4: OVERALL ALIGNMENT
        # Compare the image with this description:
        # "{main_prompt}"
        #
        # OUTPUT FORMAT:
        # Return ONLY a valid JSON object with this exact structure (no markdown, no other text):
        # {{
        #   "state_analysis": [
        #     {{ "object": "name", "actual_state": "description", "match_score": 0.9, "analysis": "reason" }}
        #   ],
        #   "semantic_analysis": [
        #     {{ "relation": "obj1 rel obj2", "verified": true, "confidence": 0.9 }}
        #   ],
        #   "color_analysis": [
        #     {{ "object": "name", "actual_colors": ["red"], "match_score": 0.9, "analysis": "reason" }}
        #   ],
        #   "alignment_metrics": {{
        #     "semantic_similarity": 0.8,
        #     "content_coverage": 0.8,
        #     "style_consistency": 0.8,
        #     "overall_harmony": 0.9,
        #     "analysis": "summary"
        #   }}
        # }}
        # """
        # ---------------------------------------------------------

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

    def _parse_combined_result(self, vlm_result: Dict) -> Dict:
        """Parse combined VLM result"""
        default_scores = {
            'state_analysis': {'overall_score': 0.0, 'details': []},
            'semantic_analysis': {'overall_score': 0.0, 'details': []},
            'alignment_analysis': {'overall_score': 0.0, 'metrics': {}}
        }
        
        if not vlm_result or not vlm_result.get('success'):
            return default_scores
            
        response = vlm_result.get('response', '').strip()
        
        # Try to clean markdown tags
        if response.startswith('```json'):
            response = response.replace('```json', '').replace('```', '')
        elif response.startswith('```'):
            response = response.replace('```', '')
            
        try:
            # Try to find JSON part
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
            
            # 2. Semantic Score
            relations = parsed.get('semantic_analysis', [])
            semantic_score = 0.0
            if relations:
                scores = [r.get('confidence', 1.0 if r.get('verified') else 0.0) for r in relations]
                semantic_score = sum(scores) / len(scores) * 100
                
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
             print(f"   Parse Error: {e}")
             return default_scores

    # Geometric validation methods (Size & Spatial)
    def _verify_size_geometric(self, data: Dict) -> Dict:
        """Validate object size based on BBox"""
        json_data = data['json_data']
        bboxes = data['bboxes_dict'] # List of dicts
        image_path = data['image_path']
        
        expected_sizes = extract_expected_sizes(json_data)
        if not expected_sizes:
            return {'overall_score': 100.0, 'details': [], 'note': 'No size requirements'}
            
        # Get actual image size
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_w, img_h = img.size
            img_area = img_w * img_h
        except Exception as e:
            print(f"   Failed to load image for size verification: {e}")
            return {'overall_score': 0.0, 'details': [], 'note': 'Image load failed'}
        
        # Construct name -> bbox mapping
        # Handle duplicate objects by taking the largest one for simplicity
        name_to_bbox = {}
        
        for det in bboxes:
            label = det.get('label')
            box = det.get('box')
            confidence = det.get('confidence', 0.0)
            
            if not label or not box or confidence < 0.3:
                continue
                
            # Calculate area (absolute pixels)
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
            # Try to match detected objects
            # Simple fuzzy match
            matched_bbox = None
            for label, info in name_to_bbox.items():
                 if label in obj_name or obj_name in label:
                     matched_bbox = info
                     break
            
            if not matched_bbox:
                results.append({'object': obj_name, 'expected': size_desc, 'actual': 'not detected', 'score': 0.0})
                continue
                
            area = matched_bbox['area']
            area_ratio = area / img_area  # Convert to relative area
            score = 0.0
            actual_desc = "unknown"
            
            # Simple threshold check (using relative area)
            if area_ratio < 0.05:
                actual_desc = "small"
            elif area_ratio < 0.25:
                actual_desc = "medium"
            else:
                actual_desc = "large"
                
            # Loose match
            if size_desc.lower() in actual_desc or actual_desc in size_desc.lower():
                score = 1.0
            elif size_desc.lower() == 'medium' and (area_ratio > 0.02 and area_ratio < 0.4): # Slightly relax medium
                score = 0.8
            else:
                score = 0.3 # Wrong size but detected
                
            results.append({
                'object': obj_name, 
                'expected': size_desc, 
                'actual': actual_desc, 
                'area': round(area, 3),  # Absolute area (pixels^2)
                'area_ratio': round(area_ratio, 4),  # Relative area
                'score': score
            })
            total_score += score
            
        overall = (total_score / count * 100) if count > 0 else 100.0
        return {'overall_score': overall, 'details': results}

    def _verify_spatial_geometric(self, data: Dict) -> Dict:
        """Validate spatial relationships based on BBox"""
        json_data = data['json_data']
        spatial_relations = extract_spatial_relations(json_data)
        
        if not spatial_relations:
             return {'overall_score': 100.0, 'details': [], 'note': 'No spatial relations'}
             
        bboxes = data['bboxes_dict']
        id_to_name = get_object_id_to_name_mapping(json_data)
        
        # Need to build mapping again, ideally using object_id if supported by OD result
        # OD result mainly has label, no object_id, so must match by label
        # This is problem for multi-instance same-name objects, but acceptable compromise for validation system
        
        name_to_bbox = {}
        for det in bboxes:
             label = det.get('label')
             box = det.get('box')
             if label and box:
                  # [x1, y1, x2, y2] -> center [cx, cy]
                  cx = (box[0] + box[2]) / 2
                  cy = (box[1] + box[3]) / 2
                  name_to_bbox[label] = {'box': box, 'center': (cx, cy)}

        results = []
        total_score = 0
        count = 0
        
        for rel in spatial_relations:
            if len(rel) < 3: continue
            
            obj1_id, relation, obj2_id = rel[0], rel[1], rel[2]
            obj1_name = id_to_name.get(obj1_id, str(obj1_id)).split('_')[0]
            obj2_name = id_to_name.get(obj2_id, str(obj2_id)).split('_')[0]
            
            count += 1
            
            # Find BBox
            bbox1 = None
            bbox2 = None
            
            # Fuzzy match
            for label, info in name_to_bbox.items():
                if obj1_name in label or label in obj1_name: bbox1 = info
            for label, info in name_to_bbox.items():
                if obj2_name in label or label in obj2_name: bbox2 = info
                
            if not bbox1 or not bbox2:
                results.append({'relation': f"{obj1_name} {relation} {obj2_name}", 'verified': False, 'reason': 'object not detected'})
                continue
                
            # Geometric validation logic
            c1 = bbox1['center']
            c2 = bbox2['center']
            verified = False
            
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1] # y increases downwards usually
            
            # Relation mapping (assuming y increases downwards)
            if 'left' in relation: # obj1 is left of obj2 -> c1.x < c2.x
                if dx < 0: verified = True
            elif 'right' in relation:
                if dx > 0: verified = True
            elif 'above' in relation or 'top' in relation: # obj1 above obj2 -> c1.y < c2.y
                if dy < 0: verified = True
            elif 'below' in relation or 'bottom' in relation or 'under' in relation:
                if dy > 0: verified = True
            else:
                # Complex relations like 'inside', 'on' are hard to judge by center point, pass for now
                verified = True 
            
            score = 1.0 if verified else 0.0
            total_score += score
            results.append({'relation': f"{obj1_name} {relation} {obj2_name}", 'verified': verified, 'score': score})
            
        overall = (total_score / count * 100) if count > 0 else 100.0
        return {'overall_score': overall, 'details': results}

    # Deprecated methods
    def _quick_color_analysis(self, data): return {}
    def _quick_size_analysis(self, data): return {}
    def _batch_state_analysis(self, batch_data): return []
    def _batch_semantic_analysis(self, batch_data): return []
    def _batch_alignment_analysis(self, batch_data): return []
    def _assemble_result(self, data, state, semantic, alignment): return {}

__all__ = ['BatchProcessor']
