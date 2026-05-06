#!/usr/bin/env python3
"""
Report Generation Script - Step 4
Generate multi-model comparison report
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from multimodel_gen import ComparativeReporter


def main():
    # Path configuration
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    results_dir = repo_root / "experiments/multimodel_generation/experiment_results"
    
    print(f"=" * 80)
    print(f"Generating Comparative Report")
    print(f"Results Directory: {results_dir}")
    print(f"=" * 80)
    
    # Create report generator
    reporter = ComparativeReporter(results_dir=str(results_dir))
    
    # Generate report
    reporter.generate_report()


if __name__ == "__main__":
    main()
