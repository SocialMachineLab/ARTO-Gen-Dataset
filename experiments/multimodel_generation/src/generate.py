#!/usr/bin/env python3
"""
Multi-model image generation script
Supports Qwen, FLUX, SD3.5, and SDXL models
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodel_gen import MultiModelGenerator


def main():
    parser = argparse.ArgumentParser(description="Multi-model Image Generation")
    parser.add_argument("--model", required=True, 
                       choices=["qwen", "flux", "sd35", "sdxl"],
                       help="Select model")
    parser.add_argument("--dataset-dir", required=True,
                       help="Dataset directory")
    parser.add_argument("--output-dir", required=True,
                       help="Output directory")
    
    args = parser.parse_args()
    
    print(f"=" * 80)
    print(f"Multi-Model Image Generation")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset_dir}")
    print(f"Output: {args.output_dir}")
    print(f"=" * 80)
    
    generator = MultiModelGenerator()
    generator.generate_batch(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        model_key=args.model
    )


if __name__ == "__main__":
    main()
