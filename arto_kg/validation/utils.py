"""
Utility functions
"""

import os
import json
from typing import Dict, Any, List, Optional


def load_gt_data(gt_path: str) -> Dict[str, Any]:
    """Load GT data"""
    try:
        # If relative path, convert to absolute
        if not os.path.isabs(gt_path):
            # Assume relative to current working directory
            project_root = os.getcwd()
            gt_path = os.path.join(project_root, gt_path)
        
        # Normalize path, handle ../ etc.
        gt_path = os.path.normpath(gt_path)
        
        with open(gt_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading GT data from {gt_path}: {e}")
        return {}


def extract_expected_colors(json_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract expected colors from GT data"""
    object_colors = {}
    
    enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
    for obj in enhanced_objects:
        obj_name = obj.get('name', '')
        if obj_name:
            primary_colors = obj.get('primary_colors', [])
            object_colors[obj_name] = primary_colors
    
    return object_colors


def extract_expected_sizes(json_data: Dict[str, Any]) -> Dict[str, str]:
    """Extract expected sizes from GT data"""
    object_sizes = {}
    
    enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
    for obj in enhanced_objects:
        obj_name = obj.get('name', '')
        size = obj.get('size', '')
        if obj_name and size:
            object_sizes[obj_name] = size
    
    return object_sizes


def extract_expected_states(json_data: Dict[str, Any]) -> Dict[str, str]:
    """Extract expected states from GT data"""
    object_states = {}
    
    enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
    for obj in enhanced_objects:
        obj_name = obj.get('name', '')
        state = obj.get('state', '')
        if obj_name and state:
            object_states[obj_name] = state
    
    return object_states


def extract_spatial_relations(json_data: Dict[str, Any]) -> List[List]:
    """Extract spatial relations from GT data"""
    composition = json_data.get('composition', {})
    spatial_relations = composition.get('spatial_relations', [])
    return spatial_relations


def extract_semantic_relations(json_data: Dict[str, Any]) -> List[List]:
    """Extract semantic relations from GT data"""
    composition = json_data.get('composition', {})
    semantic_relations = composition.get('semantic_relations', [])
    return semantic_relations


def get_object_id_to_name_mapping(json_data: Dict[str, Any]) -> Dict[int, str]:
    """Get object ID to name mapping"""
    mapping = {}
    
    enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
    for obj in enhanced_objects:
        obj_id = obj.get('object_id')
        obj_name = obj.get('name', '')
        if obj_id is not None and obj_name:
            mapping[obj_id] = obj_name
    
    return mapping


def extract_main_prompt(json_data: Dict[str, Any]) -> str:
    """Extract main prompt"""
    final_prompts = json_data.get('final_prompts', {})
    return final_prompts.get('main_prompt', '')


def make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON serializable format"""
    import numpy as np
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(make_json_serializable(item) for item in obj)
    else:
        return obj


def save_validation_result(result: Dict[str, Any], output_path: str):
    """Save validation result"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Ensure serializable
        serializable_result = make_json_serializable(result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving result to {output_path}: {e}")
        return False


__all__ = [
    'load_gt_data',
    'extract_expected_colors',
    'extract_expected_sizes',
    'extract_expected_states',
    'extract_spatial_relations',
    'extract_semantic_relations',
    'get_object_id_to_name_mapping',
    'extract_main_prompt',
    'make_json_serializable',
    'save_validation_result'
]
