"""
Result Validator
"""


import os
import sys
import json
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

# Add project path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from validation.comprehensive_evaluator import ComprehensiveArtworkEvaluator
from validation.vlm_wrapper import create_vlm_wrapper


class ResultValidator:
    """Result Validator - Validate generated images"""
    
    MODEL_FOLDERS = {
        "qwen": "qwen",
        "flux": "black-forest-labs_FLUX.1-dev",
        "sd35": "stabilityai_stable-diffusion-3.5-large",
        "sdxl": "stabilityai_stable-diffusion-xl-base-1.0"
    }
    
    def __init__(self, dataset_dir: str, results_dir: str):
        """
        Initialize validator
        
        Args:
            dataset_dir: Dataset directory
            results_dir: Results directory
        """
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.evaluator = None
    
    def initialize_evaluator(self):
        """Initialize evaluator"""
        print("Initializing VLM Wrapper...")
        try:
            vlm_wrapper = create_vlm_wrapper(use_fallback=False)
        except Exception as e:
            print(f"Warning: Failed to initialize VLM: {e}")
            print("Using Fallback VLM...")
            vlm_wrapper = create_vlm_wrapper(use_fallback=True)
        
        print("Initializing Comprehensive Evaluator...")
        self.evaluator = ComprehensiveArtworkEvaluator(vlm_wrapper=vlm_wrapper)
    
    def validate_model(self, model_key: str, force: bool = False, limit: Optional[int] = None):
        """
        Validate results for specified model
        
        Args:
            model_key: Model key
            force: Force re-validation
            limit: Limit validation count
        """
        if model_key not in self.MODEL_FOLDERS:
            raise ValueError(f"Unknown model: {model_key}")
        
        print(f"Starting validation for model: {model_key}")
        
        # Initialize evaluator
        if self.evaluator is None:
            self.initialize_evaluator()
        
        model_folder = self.MODEL_FOLDERS[model_key]
        model_path = self.results_dir / model_folder
        
        if not model_path.exists():
            print(f"Error: Model folder not found: {model_path}")
            return
        
        # Stats
        total = 0
        processed = 0
        errors = 0
        
        # Iterate generated images
        for root, dirs, files in os.walk(model_path):
            # Skip validation directories
            dirs[:] = [d for d in dirs if not d.endswith('_validation')]
            
            for filename in files:
                if filename.endswith(".png") and filename.startswith("artwork_"):
                    total += 1
                    image_path = Path(root) / filename
                    artwork_id = image_path.stem
                    
                    try:
                        # Build original JSON path
                        rel_path = image_path.parent.relative_to(model_path)
                        
                        # Try multiple potential JSON filenames
                        potential_json_names = [
                            f"{artwork_id}.json",
                            f"{artwork_id}_v2.json",
                            f"{artwork_id.replace('_v2', '')}_v2.json"
                        ]
                        
                        original_json_path = None
                        for json_name in potential_json_names:
                            p = self.dataset_dir / rel_path / json_name
                            if p.exists():
                                original_json_path = p
                                break
                        
                        if not original_json_path:
                            print(f"Warning: Original JSON not found for {artwork_id}")
                            continue
                        
                        # Load original data
                        with open(original_json_path, 'r') as f:
                            original_data = json.load(f)
                        
                        # Define validation output directory
                        val_output_dir = image_path.parent / f"{artwork_id}_validation"
                        
                        # Check if already validated
                        if not force and (val_output_dir / "comprehensive_evaluation_results.json").exists():
                            print(f"Skipping {artwork_id} (already validated)")
                            continue
                        
                        print(f"\nValidating {artwork_id}...")
                        
                        # Run evaluation
                        results = self.evaluator.evaluate_artwork_step_by_step(
                            json_data=original_data,
                            image_path=str(image_path),
                            output_dir=str(val_output_dir)
                        )
                        
                        processed += 1
                        
                    except Exception as e:
                        print(f"Error validating {artwork_id}: {e}")
                        traceback.print_exc()
                        errors += 1
                    
                    # Check limit
                    if limit and processed >= limit:
                        print(f"\nReached limit of {limit} images. Stopping.")
                        break
            
            if limit and processed >= limit:
                break
        
        print(f"\nValidation Complete for {model_key}")
        print(f"Total Images: {total}")
        print(f"Processed: {processed}")
        print(f"Errors: {errors}")
