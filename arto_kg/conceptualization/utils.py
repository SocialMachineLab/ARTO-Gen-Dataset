"""
Utility Functions Module
Contains common functions for JSON parsing, logging, file operations, etc.
"""

import json
import re
import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Configure logging
def setup_logger(name: str = "artwork_pipeline", log_dir: str = "data/output/logs") -> logging.Logger:
    """Setup logger"""
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# JSON Parsing Functions (Reused and optimized from original code)
def parse_json_response(response_text: str, patterns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Parse JSON content from LLM response
    Reuse parsing logic from original code
    """
    if patterns is None:
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        ]
    
    # Save original full text (including <think> tags)
    original_text = response_text
    
    # Remove thinking tags for JSON parsing
    cleaned_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
    
    # Try various patterns to parse JSON
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                continue
    
    # Return original full text (including <think>) if parsing fails
    return {"error": "JSON parsing failed", "raw_response": original_text}

def extract_response_content(response: Any) -> str:
    """
    Extract content from different response formats
    Reuse response handling logic from original code
    """
    if hasattr(response, 'message'):
        return response.message.content.strip()
    elif isinstance(response, dict) and 'message' in response:
        return response['message']['content'].strip()
    elif isinstance(response, dict) and 'content' in response:
        return response['content'].strip()
    else:
        raise ValueError(f"Unexpected response format: {type(response)}")


# File Operation Functions
def ensure_dir(directory: str) -> None:
    """Ensure directory exists"""
    os.makedirs(directory, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: str, ensure_ascii: bool = False) -> None:
    """Save JSON data to file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=ensure_ascii)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_text(content: str, filepath: str) -> None:
    """Save text content to file"""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

# New Staged Saving Functions
class StageOutputManager:
    """Stage Output Manager"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = base_output_dir
        self.stage_mapping = {
            "object_selection": "stage_1_object_selection",
            "compatibility_check": "stage_1_object_selection", 
            "scene_framework": "stage_2_scene_framework",
            "spatial_layout": "stage_3_spatial_layout",
            "object_enhancement": "stage_4_object_enhancement",
            "environment_details": "stage_5_environment_details",
            "artistic_expression": "stage_6_artistic_expression"
        }
        
    def create_stage_directories(self, stage: str) -> Dict[str, str]:
        """Create stage directory structure"""
        stage_name = self.stage_mapping.get(stage, f"stage_{stage}")
        stage_dir = os.path.join(self.base_output_dir, stage_name)
        
        dirs = {
            "raw_outputs": os.path.join(stage_dir, "raw_outputs"),
            "parsed_results": os.path.join(stage_dir, "parsed_results"), 
            "processing_logs": os.path.join(stage_dir, "processing_logs"),
            "base": stage_dir
        }
        
        for dir_path in dirs.values():
            ensure_dir(dir_path)
            
        return dirs
    
    def save_stage_output(self, stage: str, artwork_id: str, 
                         raw_output: str, parsed_result: Dict[str, Any],
                         processing_info: Dict[str, Any]) -> Dict[str, str]:
        """Save complete information for stage output"""
        dirs = self.create_stage_directories(stage)
        saved_files = {}
        
        # Save raw LLM output
        raw_file = os.path.join(dirs["raw_outputs"], f"{artwork_id}_raw.txt")
        save_text(raw_output, raw_file)
        saved_files["raw_output"] = raw_file
        
        # Extract and save thinking process (if <think> tag exists)
        think_content = self._extract_think_content(raw_output)
        if think_content:
            think_file = os.path.join(dirs["raw_outputs"], f"{artwork_id}_think.txt")
            save_text(think_content, think_file)
            saved_files["think_output"] = think_file
        
        # Save parsed result
        parsed_file = os.path.join(dirs["parsed_results"], f"{artwork_id}_parsed.json")
        save_json(parsed_result, parsed_file)
        saved_files["parsed_result"] = parsed_file
        
        # Save processing info (including parsing status, errors, etc.)
        processing_file = os.path.join(dirs["processing_logs"], f"{artwork_id}_processing.json")
        processing_data = {
            "artwork_id": artwork_id,
            "stage": stage,
            "timestamp": get_timestamp(),
            "processing_info": processing_info,
            "parsing_success": "error" not in parsed_result,
            "has_think_content": think_content is not None,
            "think_content_length": len(think_content) if think_content else 0,
            "files": {
                "raw_output": raw_file,
                "parsed_result": parsed_file
            }
        }
        
        if think_content:
            processing_data["files"]["think_output"] = saved_files["think_output"]
            
        save_json(processing_data, processing_file)
        saved_files["processing_info"] = processing_file
        
        return saved_files
    
    def _extract_think_content(self, raw_output: str) -> Optional[str]:
        """Extract content within <think> tags"""
        think_matches = re.findall(r'<think>(.*?)</think>', raw_output, re.DOTALL)
        if think_matches:
            # If multiple <think> blocks, merge them
            return '\n\n--- Think Block ---\n\n'.join(match.strip() for match in think_matches)
        return None
    
    def save_batch_stage_output(self, stage: str, batch_results: List[Dict[str, Any]]) -> str:
        """Save batch processing stage output"""
        dirs = self.create_stage_directories(stage)
        
        # Save batch summary info
        batch_summary = {
            "stage": stage,
            "timestamp": get_timestamp(),
            "total_items": len(batch_results),
            "successful_items": len([r for r in batch_results if not r.get("has_error", False)]),
            "failed_items": len([r for r in batch_results if r.get("has_error", False)]),
            "batch_results": batch_results
        }
        
        batch_file = os.path.join(dirs["base"], f"batch_summary_{stage}.json")
        save_json(batch_summary, batch_file)
        
        return batch_file

def create_batch_output_structure(output_base_dir: str) -> Tuple[str, StageOutputManager]:
    """Create batch output directory structure"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = os.path.join(output_base_dir, f"batch_{timestamp}")
    
    # Create final results directory
    ensure_dir(os.path.join(batch_dir, "final_results"))
    
    # Create output manager
    output_manager = StageOutputManager(batch_dir)
    
    return batch_dir, output_manager

def generate_artwork_id() -> str:
    """Generate unique artwork ID"""
    return f"artwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"

def get_timestamp() -> str:
    """Get current timestamp"""
    return datetime.now().isoformat()


# Batch Processing Helper Functions
def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# Progress Tracking Functions
def print_progress(current: int, total: int, prefix: str = "Progress") -> None:
    """Print progress info"""
    percentage = (current / total) * 100
    print(f"{prefix}: {current}/{total} ({percentage:.1f}%)")

def format_duration(seconds: float) -> str:
    """Format duration"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

# Random Number and Selection Functions
import random

def set_random_seed(seed: int) -> None:
    """Set random seed"""
    random.seed(seed)

def weighted_random_choice(items: List[Any], weights: List[float]) -> Any:
    """Weighted random choice"""
    return random.choices(items, weights=weights, k=1)[0]
