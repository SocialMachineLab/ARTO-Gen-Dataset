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
    
    def __init__(self, vlm_wrapper: VLLMWrapper, batch_size: int = 10, enable_od: bool = True):
        """
        Initialize batch processor
        
        Args:
            vlm_wrapper: VLLMWrapper instance
            batch_size: Batch size
            enable_od: Enable automatic object detection
        """
        self.vlm = vlm_wrapper
        self.batch_size = batch_size
        self.enable_od = enable_od
        
        # Initialize all modular validators
        from .validators import (
            ObjectDetectionValidator,
            SizeValidator,
            SpatialValidator,
            StateValidator,
            AlignmentValidator,
            ColorValidator
        )
        
        self.object_detector = ObjectDetectionValidator() if enable_od else None
        self.color_validator = ColorValidator()  # No vlm_wrapper needed
        self.size_validator = SizeValidator()  # No parameters
        self.spatial_validator = SpatialValidator(vlm_wrapper=vlm_wrapper)
        self.state_validator = StateValidator(vlm_wrapper=vlm_wrapper)
        self.alignment_validator = AlignmentValidator(vlm_wrapper=vlm_wrapper)  # device auto-detected
        self.color_analyzer = ColorAnalyzer()  # Legacy compatibility
    
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
        
        print(f"Starting batch processing: {total_files} files")
        print(f"Batch size: {self.batch_size}")
        print(f"Output dir: {output_dir}")
        
        start_time = time.time()
        
        # Process by batch
        for i in range(0, total_files, self.batch_size):
            batch_files = file_list[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
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
        
        print(f"Batch processing completed.")
        print(f"Total: {total_files}")
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Time: {elapsed:.1f}s ({elapsed/total_files:.2f}s/file)")
        
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
        skipped_count = 0
        
        for gt_path in batch_files:
            try:
                data = self._prepare_single_file(gt_path, od_results_dir, image_base_dir)
                if not data:
                    continue
                
                # Check if output file exists (supports resuming)
                artwork_id = data['artwork_id']
                output_path = os.path.join(output_dir, artwork_id, f"{artwork_id}_full_validation.json")
                
                if os.path.exists(output_path):
                    # print(f"Skipping {artwork_id} (already processed)")
                    skipped_count += 1
                    continue
                
                batch_data.append(data)
                
            except Exception as e:
                print(f"Failed to prepare {os.path.basename(gt_path)}: {e}")
        
        if not batch_data:
            return {'processed': 0, 'skipped': skipped_count, 'failed': 0}
        
        # Construct combined validation prompt (State, Semantic, Alignment)
        prompts = []
        images = []
        
        # Process each item with modular validators
        processed = 0
        skipped = 0
        failed = 0
        
        for data in batch_data:
            try:
                json_data = data['json_data']
                image_path = data['image_path']
                bboxes_dict = data.get('bboxes_dict', {})
                
                # Convert bboxes_dict format if needed (list of dicts -> dict)
                if isinstance(bboxes_dict, list):
                    bboxes_dict = {det.get('label') or det.get('text_label'): det.get('box') 
                                   for det in bboxes_dict if det.get('box')}
                
                # 1. Color Analysis (CV-based, no VLM)
                print(f"  - Running color validation...")
                color_result = self.color_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )
                
                # 2. Size Analysis (Geometric)
                print(f"  - Running size validation...")
                size_result = self.size_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )
                
                # 3. State Analysis (VLM)
                print(f"  - Running state validation...")
                state_result = self.state_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )
                
                # 4. Spatial Analysis (Geometric + VLM)
                print(f"  - Running spatial validation...")
                spatial_result = self.spatial_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )
                
                # 5. Alignment Analysis (VLM)
                print(f"  - Running alignment validation...")
                alignment_result = self.alignment_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )
                
                # 6. Extract semantic relations from spatial result
                semantic_result = {
                    'overall_score': spatial_result.get('overall_score', 0),
                    'relations': spatial_result.get('relation_analysis', {})
                }
                
                # 4. Assemble full result
                full_result = {
                    'artwork_id': data['artwork_id'],
                    'image_path': image_path,
                    'bboxes_source': data['od_path'],
                    'steps': {
                        'step3_color': color_result,
                        'step4_size': size_result,
                        'step5_state': state_result,
                        'step6_semantic': semantic_result,
                        'step6_spatial': spatial_result,
                        'step7_alignment': alignment_result
                    },
                    'validation_method': 'modular_validators'
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
                print(f"Failed to process {data.get('artwork_id', 'unknown')}: {e}")
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
            print(f"Failed to load JSON data from {gt_path}")
            return None
        
        artwork_id = json_data.get('artwork_id', '')
        if not artwork_id:
            print(f"No artwork_id found in {gt_path}")
            return None
        
        # Construct image path
        flat_image_path = os.path.join(image_base_dir, f"{artwork_id}.png")
        if os.path.exists(flat_image_path):
            image_path = flat_image_path
        else:
            print(f"Flat image not found at {flat_image_path} checking legacy structure...")
            # Fallback to legacy structured paths
            path_parts = gt_path.split('/')
            if 'v5' in path_parts:
                dataset = 'v5'
            elif 'v6' in path_parts or 'v6_normal' in path_parts:
                dataset = 'v6'
            else:
                print(f"Path structure not recognized (no v5/v6) and flat image missing: {gt_path}")
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
            
                if 'final_results' in path_parts:
                    idx = path_parts.index('final_results')
                    if idx + 1 < len(path_parts):
                        style_name = path_parts[idx + 1]
            
           
            if dataset == 'v5':
                if not batch_name:
                    print(f"Batch name not found in path for v5 dataset: {gt_path}")
                    # If flattened structure used but flat check failed...
                    return None
                image_path = os.path.join(
                    image_base_dir,
                    f"{dataset}_generated_images",
                    batch_name,
                    f"{artwork_id}.png"
                )
            else:  
                if not style_name:
                    print(f"Style name not found in path for v6 dataset: {gt_path}")
                    return None
                image_path = os.path.join(
                    image_base_dir,
                    "v6_normal_generated_images",
                    style_name,
                    f"{artwork_id}.png"
                )
        
        if not os.path.exists(image_path):
            print(f"Image not found at constructed path: {image_path}")
            return None
        
        # OD result path
        od_path = os.path.join(od_results_dir, f"{artwork_id}_fused.json")
        if not os.path.exists(od_path):
            # Try _combined.json suffix
            od_path_combined = os.path.join(od_results_dir, f"{artwork_id}_combined.json")
            if os.path.exists(od_path_combined):
                od_path = od_path_combined
            else:
                # Auto-run OD if enabled
                if self.enable_od and self.object_detector:
                    print(f"OD result not found, running automatic detection for {artwork_id}...")
                    
                    # Extract expected objects
                    expected_objects = self._extract_object_names(json_data)
                    
                    if expected_objects:
                        try:
                            # Run detection
                            detection_result = self.object_detector.main_evaluation(
                                json_data, image_path
                            )
                            
                            # Save result
                            os.makedirs(od_results_dir, exist_ok=True)
                            with open(od_path, 'w') as f:
                                json.dump(detection_result, f, indent=2)
                            
                            print(f"Auto-detection completed for {artwork_id}")
                        except Exception as e:
                            print(f"Auto-detection failed for {artwork_id}: {e}")
                            return None
                    else:
                        print(f"No expected objects found for {artwork_id}")
                        return None
                else:
                    print(f"OD result not found: {od_path} (or _combined.json)")
                    return None

        
        # Load OD result
        with open(od_path, 'r') as f:
            try:
                od_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Failed to decode OD result: {od_path}")
                return None
        
        # Extract bboxes from fused JSON structure
        # Fused JSON has structure: detailed_results.detection.detected_objects
        bboxes_list = []
        if 'detailed_results' in od_data:
            # New fused JSON format
            detection_data = od_data.get('detailed_results', {}).get('detection', {})
            bboxes_list = detection_data.get('detected_objects', [])
        elif 'detections' in od_data:
            # Legacy format
            bboxes_list = od_data.get('detections', [])
        else:
            print(f"[BatchProcessor] Warning: No detection data found in {od_path}")
        
        # Convert list of detections to dict format: {object_name: bbox}
        bboxes_dict = {}
        for det in bboxes_list:
            obj_name = det.get('class_name') or det.get('label') or det.get('text_label')
            bbox = det.get('bbox') or det.get('box')
            if obj_name and bbox:
                bboxes_dict[obj_name] = bbox
        
        print(f"[BatchProcessor] Loaded {len(bboxes_dict)} object bboxes from {os.path.basename(od_path)}")
        
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
                # Handle both list [s,r,o] and dict {'subject':s, 'relation':r, 'object':o} formats
                try:
                    if isinstance(rel, dict):
                        subj = rel.get('subject')
                        relation = rel.get('relation') 
                        obj = rel.get('object')
                    elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                        subj = rel[0]
                        relation = rel[1]
                        obj = rel[2]
                    else:
                        continue

                    if subj and relation and obj:
                        # Resolve IDs to names if they are IDs
                        obj1 = id_to_name.get(subj, str(subj)).split('_')[0]
                        obj2 = id_to_name.get(obj, str(obj)).split('_')[0]
                        relation_text += f"- {obj1} {relation} {obj2}\n"
                except Exception as e:
                    print(f"Warning: Failed to parse relation {rel}: {e}")
                    continue
        else:
            relation_text = "No specific relations to verify."
            
        # 3. Alignment Info
        main_prompt = extract_main_prompt(json_data)
    

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

    def _parse_combined_result(self, vlm_result: Dict, gt_data: Optional[Dict] = None) -> Dict:
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
            
            # FIX: Check if GT expected relations
            has_expected_relations = False
            if gt_data:
                from .utils import extract_semantic_relations
                expected_rels = extract_semantic_relations(gt_data)
                if expected_rels:
                    has_expected_relations = True
            
            if not has_expected_relations:
                # If GT didn't expect any relations, we give perfect score (ignore VLM hallucinations)
                semantic_score = 100.0
            elif relations:
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
             print(f"Parse Error: {e}")
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
            print(f"Failed to load image for size verification: {e}")
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
            # Handle both list [s,r,o] and dict {'subject':s, 'relation':r, 'object':o} formats
            try:
                if isinstance(rel, dict):
                    obj1_id = rel.get('subject')
                    relation = rel.get('relation')
                    obj2_id = rel.get('object')
                elif isinstance(rel, (list, tuple)) and len(rel) >= 3:
                    obj1_id = rel[0]
                    relation = rel[1]
                    obj2_id = rel[2]
                else:
                    continue

                if not (obj1_id and relation and obj2_id):
                    continue

                obj1_name = id_to_name.get(obj1_id, str(obj1_id)).split('_')[0]
                obj2_name = id_to_name.get(obj2_id, str(obj2_id)).split('_')[0]
            except Exception as e:
                print(f"Warning: Failed to parse spatial relation {rel}: {e}")
                continue
            
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
    
    def _extract_object_names(self, json_data: Dict[str, Any]) -> List[str]:
        """Extract object names from JSON data"""
        objects = []
        
        # Try enhanced_objects first (new structure)
        enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
        for obj in enhanced_objects:
            if isinstance(obj, dict) and 'name' in obj:
                objects.append(obj['name'])
        
        # Fallback to object_names (legacy structure)
        if not objects:
            object_names = json_data.get('objects', {}).get('object_names', [])
            if isinstance(object_names, list):
                objects.extend(object_names)
        
        # Fallback to direct objects array (oldest structure)
        if not objects:
            objects_list = json_data.get('objects', [])
            if isinstance(objects_list, list):
                for obj in objects_list:
                    if isinstance(obj, dict) and 'name' in obj:
                        objects.append(obj['name'])
                    elif isinstance(obj, str):
                        objects.append(obj)
        
        return objects

    # Deprecated methods

    def _quick_color_analysis(self, data): return {}
    def _quick_size_analysis(self, data): return {}
    def _batch_state_analysis(self, batch_data): return []
    def _batch_semantic_analysis(self, batch_data): return []
    def _batch_alignment_analysis(self, batch_data): return []
    def _assemble_result(self, data, state, semantic, alignment): return {}

__all__ = ['BatchProcessor']
