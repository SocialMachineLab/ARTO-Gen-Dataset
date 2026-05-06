"""Analyze intra-class consistency for each art style"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.metrics import compute_intra_class_metrics
from utils.visualization import plot_intra_class_comparison


def load_features(features_dir: str) -> dict:
    """Load all saved features"""
    features = {}

    print("Loading features")

    for style in config.STYLES:
        features[style] = {}

        for model_name in config.MODELS.keys():
            feature_path = os.path.join(features_dir, model_name, f"{style}.npy")

            if not os.path.exists(feature_path):
                print(f"Warning: Feature file not found: {feature_path}")
                continue

            feat = np.load(feature_path)
            features[style][model_name] = feat
            print(f"Loaded {style} - {model_name}: {feat.shape}")

    return features


def compute_all_intra_metrics(features: dict, metric='cosine') -> dict:
    """Compute intra-class metrics for all styles and models"""
    print("Computing intra-class metrics")

    all_metrics = {}

    for style in config.STYLES:
        all_metrics[style] = {}

        for model_name in config.MODELS.keys():
            if model_name not in features[style]:
                continue

            feat = features[style][model_name]
            metrics = compute_intra_class_metrics(feat, metric=metric)
            all_metrics[style][model_name] = metrics

            print(f"{style:25s} - {model_name:20s}: "
                  f"variance={metrics['variance']:.4f}, "
                  f"avg_dist={metrics['avg_pairwise_distance']:.4f}")

    return all_metrics


def create_summary_table(all_metrics: dict) -> pd.DataFrame:
    """Create a summary table of intra-class metrics"""
    rows = []

    for style in config.STYLES:
        row = {'Style': style}

        variances = []
        avg_dists = []
        std_dists = []

        for model_name in config.MODELS.keys():
            if model_name in all_metrics[style]:
                metrics = all_metrics[style][model_name]
                variances.append(metrics['variance'])
                avg_dists.append(metrics['avg_pairwise_distance'])
                std_dists.append(metrics['std_pairwise_distance'])

                row[f'{model_name}_variance'] = metrics['variance']
                row[f'{model_name}_avg_dist'] = metrics['avg_pairwise_distance']

                if 'n_samples' not in row:
                    row['n_samples'] = metrics['n_samples']

        row['avg_variance'] = np.mean(variances)
        row['avg_pairwise_distance'] = np.mean(avg_dists)
        row['std_variance'] = np.std(variances)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('avg_variance')

    return df


def analyze_consistency_ranking(df: pd.DataFrame):
    """Print analysis of style consistency rankings"""
    print("Consistency ranking analysis")

    print("\nTop 5 most consistent styles (lowest variance):")
    for i, row in df.head(5).iterrows():
        print(f"  {i+1}. {row['Style']:25s} - Avg Variance: {row['avg_variance']:.4f}")

    print("\nBottom 5 least consistent styles (highest variance):")
    for i, row in df.tail(5).iterrows():
        rank = len(df) - i
        print(f"  {rank}. {row['Style']:25s} - Avg Variance: {row['avg_variance']:.4f}")

    print("\nAnalysis of target combination styles:")
    for style in config.TARGET_COMBINATION:
        if style in df['Style'].values:
            row = df[df['Style'] == style].iloc[0]
            rank = df.index[df['Style'] == style].tolist()[0] + 1
            print(f"  {style:25s} - Rank: {rank:2d}/{len(df)} - "
                  f"Variance: {row['avg_variance']:.4f}")


def main():
    """Main execution function"""
    print("Intra-class Analysis")

    if not os.path.exists(config.FEATURES_DIR):
        print(f"ERROR: Features directory not found: {config.FEATURES_DIR}")
        print("Please run wikiart_extract_features.py first")
        return

    features = load_features(config.FEATURES_DIR)
    all_metrics = compute_all_intra_metrics(features, metric=config.DISTANCE_METRIC)

    print("Creating summary table")
    df = create_summary_table(all_metrics)

    output_path = os.path.join(config.RESULTS_DIR, 'intra_class_variance.csv')
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved summary table to {output_path}")

    print("\nSummary table")
    print(df[['Style', 'n_samples', 'avg_variance', 'avg_pairwise_distance']].to_string(index=False))

    analyze_consistency_ranking(df)

    print("Generating visualizations")

    avg_variance_metrics = {
        style: {'variance': df[df['Style'] == style]['avg_variance'].values[0]}
        for style in config.STYLES
        if style in df['Style'].values
    }

    plot_path = os.path.join(config.VISUALIZATION_DIR, 'intra_class_variance_comparison.png')
    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)

    plot_intra_class_comparison(
        avg_variance_metrics,
        metric_name='variance',
        title='Intra-class Variance by Style (Lower = More Consistent)',
        save_path=plot_path,
        dpi=config.FIG_DPI
    )

    print("Done")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print(f"Visualizations saved to: {config.VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
