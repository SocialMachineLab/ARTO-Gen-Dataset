"""
Comparative Reporter
Generate multi-model comparative report
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any


class ComparativeReporter:
    """Comparative Report Generator"""
    
    MODEL_FOLDERS = {
        "Qwen-Image": "qwen",
        "Flux.1-Dev": "black-forest-labs_FLUX.1-dev",
        "SD3.5-Large": "stabilityai_stable-diffusion-3.5-large",
        "SDXL-Base": "stabilityai_stable-diffusion-xl-base-1.0"
    }
    
    def __init__(self, results_dir: str):
        """
        Initialize report generator
        
        Args:
            results_dir: Results directory
        """
        self.results_dir = Path(results_dir)
    
    def load_json(self, filepath: Path) -> Dict:
        """Load JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def collect_results(self, model_name: str, folder_name: str) -> List[Dict]:
        """Collect validation results for specified model"""
        print(f"Collecting results for {model_name}...")
        model_path = self.results_dir / folder_name
        
        if not model_path.exists():
            print(f"Warning: Model folder not found: {model_path}")
            return []
        
        results = []
        for root, dirs, files in os.walk(model_path):
            for filename in files:
                if filename == "comprehensive_evaluation_results.json":
                    filepath = Path(root) / filename
                    data = self.load_json(filepath)
                    if data:
                        # Flatten metrics
                        flat_data = self.flatten_metrics(data)
                        flat_data['model'] = model_name
                        flat_data['artwork_id'] = data.get('artwork_id', 'unknown')
                        results.append(flat_data)
        
        print(f"Found {len(results)} valid results for {model_name}")
        return results
    
    def flatten_metrics(self, data: Dict) -> Dict:
        """Flatten nested validation metrics"""
        flat = {}
        
        # Overall score
        flat['overall_score'] = data.get('overall_score', 0)
        
        # Dimension scores
        dim_scores = data.get('dimension_scores', {})
        for k, v in dim_scores.items():
            flat[f"dim_{k}"] = v
        
        return flat
    
    def generate_report(self):
        """Generate complete comparative report"""
        print("=" * 70)
        print("Generating Comparative Report")
        print("=" * 70)
        
        # Collect all results
        all_results = []
        for model_name, folder in self.MODEL_FOLDERS.items():
            model_results = self.collect_results(model_name, folder)
            all_results.extend(model_results)
        
        if not all_results:
            print("No results to report.")
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Calculate mean scores
        mean_scores = df.groupby('model').mean()
        
        print("\n=== Comparative Analysis Report ===\n")
        print(mean_scores)
        
        # Generate Markdown report
        report_path = self.results_dir / "comparative_report.md"
        with open(report_path, 'w') as f:
            f.write("# Model Comparison Report\n\n")
            f.write("## Overview\n")
            f.write(mean_scores.to_markdown())
            f.write("\n\n## Detailed Statistics\n")
            
            summary = df.groupby('model').describe().transpose()
            f.write(summary.to_markdown())
            
            # Analysis per dimension
            f.write("\n\n## Dimension Analysis\n")
            metrics = [col for col in df.columns if col.startswith('dim_') or col == 'overall_score']
            
            for metric in metrics:
                f.write(f"\n### {metric}\n")
                metric_stats = df.groupby('model')[metric].describe()
                f.write(metric_stats.to_markdown())
                f.write("\n")
        
        print(f"\nReport generated at {report_path.absolute()}")
        
        # Generate plots
        self.plot_comparisons(df)
    
    def plot_comparisons(self, df: pd.DataFrame):
        """Generate comparison plots"""
        sns.set_theme(style="whitegrid")
        
        # Melt dataframe
        metrics = [col for col in df.columns if col.startswith('dim_') or col == 'overall_score']
        melted = df.melt(id_vars=['model', 'artwork_id'], value_vars=metrics, 
                        var_name='Metric', value_name='Score')
        
        # Box plot
        plt.figure(figsize=(14, 8))
        sns.boxplot(x='Metric', y='Score', hue='model', data=melted)
        plt.xticks(rotation=45, ha='right')
        plt.title("Model Performance Comparison across Metrics")
        plt.tight_layout()
        plt.savefig(self.results_dir / "model_comparison_boxplot.png")
        print("Saved model_comparison_boxplot.png")
        
        # Overall score distribution
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x='overall_score', hue='model', fill=True, 
                   common_norm=False, alpha=0.5)
        plt.title("Distribution of Overall Scores")
        plt.tight_layout()
        plt.savefig(self.results_dir / "overall_score_dist.png")
        print("Saved overall_score_dist.png")
