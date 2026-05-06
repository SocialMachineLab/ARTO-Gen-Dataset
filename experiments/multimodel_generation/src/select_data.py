#!/usr/bin/env python3
"""
Data Selection Script - Step 1
Select balanced dataset from object_detection experiment
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodel_gen import DataSelector


def main():
    # Path configuration
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    source_dir = repo_root / "experiments/object_detection"
    output_dir = repo_root / "experiments/multimodel_generation/experiment_dataset"
    
    # Create selector
    selector = DataSelector(
        source_dir=str(source_dir),
        output_dir=str(output_dir),
        seed=42
    )
    
    # Run selection
    selector.run()


if __name__ == "__main__":
    main()
