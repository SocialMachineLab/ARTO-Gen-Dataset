#!/usr/bin/env python3
"""
Validation Script - Step 3
Validate generated image quality
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodel_gen import ResultValidator


def main():
    parser = argparse.ArgumentParser(description="Validate generated images")
    parser.add_argument("--model", required=True,
                       choices=["qwen", "flux", "sd35", "sdxl"],
                       help="Select model to validate")
    parser.add_argument("--force", action="store_true",
                       help="Force re-validation")
    parser.add_argument("--limit", type=int,
                       help="Limit validation count")
    
    args = parser.parse_args()
    
    # Path configuration
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    dataset_dir = repo_root / "experiments/multimodel_generation/experiment_dataset"
    results_dir = repo_root / "experiments/multimodel_generation/experiment_results"
    
    print(f"=" * 80)
    print(f"Image Validation")
    print(f"Model: {args.model}")
    print(f"=" * 80)
    
    # Create validator
    validator = ResultValidator(
        dataset_dir=str(dataset_dir),
        results_dir=str(results_dir)
    )
    
    # Run validation
    validator.validate_model(
        model_key=args.model,
        force=args.force,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
