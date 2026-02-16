"""Analyze inter-class separability between different art styles"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.metrics import (
    compute_centroid_distance_matrix,
    compute_avg_inter_class_distance,
    find_nearest_neighbors,
    compute_outlier_score,
    compute_min_neighbor_distance
)
from utils.visualization import (
    plot_distance_heatmap,
    plot_dendrogram,
    plot_outlier_scores,
    plot_multi_model_consistency
)


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


def analyze_inter_class_distances(features: dict, metric='cosine') -> dict:
    """Compute and analyze inter-class distance matrices for all models"""
    print("Computing inter-class distance matrices")

    distance_matrices = {}

    for model_name in config.MODELS.keys():
        model_features = {}
        for style in config.STYLES:
            if model_name in features[style]:
                model_features[style] = features[style][model_name]

        if not model_features:
            print(f"Skipping {model_name}: no features found")
            continue

        dist_matrix, style_names = compute_centroid_distance_matrix(
            model_features, metric=metric
        )

        distance_matrices[model_name] = (dist_matrix, style_names)

        avg_dist = compute_avg_inter_class_distance(dist_matrix)
        print(f"{model_name}: Average inter-class distance = {avg_dist:.4f}")

    return distance_matrices


def analyze_nearest_neighbors(distance_matrices: dict):
    """Analyze and print nearest neighbors for each style"""
    print("Nearest neighbor analysis")

    model_name = list(distance_matrices.keys())[0]
    dist_matrix, style_names = distance_matrices[model_name]

    neighbors = find_nearest_neighbors(dist_matrix, style_names, top_k=3)

    print(f"\nUsing {model_name} for nearest neighbor analysis:\n")

    for style, neighbor_list in neighbors.items():
        print(f"{style}:")
        for neighbor, dist in neighbor_list:
            print(f"  {neighbor:25s} (distance: {dist:.4f})")
        print()


def analyze_outliers(distance_matrices: dict, features: dict):
    """Analyze which styles are most unique (outliers)"""
    print("Uniqueness/outlier analysis")

    model_name = list(distance_matrices.keys())[0]
    dist_matrix, style_names = distance_matrices[model_name]

    model_features = {
        style: features[style][model_name]
        for style in config.STYLES
        if model_name in features[style]
    }

    outlier_scores = {}
    min_distances = {}

    for style in style_names:
        outlier_score = compute_outlier_score(style, model_features, dist_matrix, style_names)
        min_dist = compute_min_neighbor_distance(style, dist_matrix, style_names)

        outlier_scores[style] = outlier_score
        min_distances[style] = min_dist

    sorted_styles = sorted(outlier_scores.items(), key=lambda x: x[1], reverse=True)

    print("\nMost unique styles (highest average distance to others):\n")
    for i, (style, score) in enumerate(sorted_styles[:5], 1):
        print(f"  {i}. {style:25s} - Avg distance: {score:.4f}, "
              f"Min distance: {min_distances[style]:.4f}")

    print("\nMost similar styles (lowest average distance to others):\n")
    for i, (style, score) in enumerate(sorted_styles[-5:], 1):
        print(f"  {i}. {style:25s} - Avg distance: {score:.4f}, "
              f"Min distance: {min_distances[style]:.4f}")

    print("\nAnalysis of target combination styles:")
    avg_outlier_score = np.mean(list(outlier_scores.values()))

    for style in config.TARGET_COMBINATION:
        if style in outlier_scores:
            score = outlier_scores[style]
            ratio = score / avg_outlier_score
            print(f"  {style:25s} - Uniqueness: {score:.4f} "
                  f"({ratio:.2f}x average)")

    return outlier_scores


def identify_similar_clusters(distance_matrices: dict):
    """Identify clusters of highly similar styles"""
    print("Identifying similar style clusters")

    model_name = list(distance_matrices.keys())[0]
    dist_matrix, style_names = distance_matrices[model_name]

    avg_dist = compute_avg_inter_class_distance(dist_matrix)
    threshold = avg_dist * 0.5

    print(f"\nAverage inter-class distance: {avg_dist:.4f}")
    print(f"Similarity threshold (50% of avg): {threshold:.4f}")
    print("\nPairs of styles below threshold (highly similar):\n")

    similar_pairs = []
    for i, style1 in enumerate(style_names):
        for j, style2 in enumerate(style_names):
            if i < j and dist_matrix[i, j] < threshold:
                similar_pairs.append((style1, style2, dist_matrix[i, j]))

    similar_pairs.sort(key=lambda x: x[2])

    if similar_pairs:
        for style1, style2, dist in similar_pairs:
            print(f"  {style1} - {style2}: {dist:.4f}")
    else:
        print("  No highly similar pairs found")


def generate_visualizations(distance_matrices: dict, outlier_scores: dict):
    """Generate all visualizations"""
    print("Generating visualizations")

    os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)

    for model_name, (dist_matrix, style_names) in distance_matrices.items():
        heatmap_path = os.path.join(
            config.VISUALIZATION_DIR,
            f'distance_matrix_{model_name}.png'
        )
        plot_distance_heatmap(
            dist_matrix,
            style_names,
            title=f"Inter-class Distance Matrix - {model_name}",
            figsize=config.FIGSIZE,
            cmap=config.HEATMAP_CMAP,
            save_path=heatmap_path,
            dpi=config.FIG_DPI
        )

        dendrogram_path = os.path.join(
            config.VISUALIZATION_DIR,
            f'dendrogram_{model_name}.png'
        )
        plot_dendrogram(
            dist_matrix,
            style_names,
            title=f"Hierarchical Clustering - {model_name}",
            save_path=dendrogram_path,
            dpi=config.FIG_DPI
        )

    if len(distance_matrices) > 1:
        multi_model_path = os.path.join(
            config.VISUALIZATION_DIR,
            'multi_model_comparison.png'
        )
        model_distances = {
            model: dist_matrix
            for model, (dist_matrix, _) in distance_matrices.items()
        }
        style_names = distance_matrices[list(distance_matrices.keys())[0]][1]

        plot_multi_model_consistency(
            model_distances,
            style_names,
            save_path=multi_model_path,
            dpi=config.FIG_DPI
        )

    outlier_path = os.path.join(
        config.VISUALIZATION_DIR,
        'style_uniqueness_scores.png'
    )
    plot_outlier_scores(
        outlier_scores,
        title="Style Uniqueness Scores (Higher = More Unique)",
        save_path=outlier_path,
        dpi=config.FIG_DPI
    )


def save_distance_matrices(distance_matrices: dict):
    """Save distance matrices to disk"""
    print("Saving distance matrices")

    for model_name, (dist_matrix, style_names) in distance_matrices.items():
        save_path = os.path.join(
            config.RESULTS_DIR,
            f'distance_matrix_{model_name}.npy'
        )
        np.save(save_path, dist_matrix)
        print(f"Saved {model_name} distance matrix to {save_path}")

        csv_path = os.path.join(
            config.RESULTS_DIR,
            f'distance_matrix_{model_name}.csv'
        )
        df = pd.DataFrame(dist_matrix, index=style_names, columns=style_names)
        df.to_csv(csv_path)
        print(f"Saved {model_name} distance matrix to {csv_path}")

    names_path = os.path.join(config.RESULTS_DIR, 'style_names.txt')
    with open(names_path, 'w') as f:
        f.write('\n'.join(style_names))
    print(f"Saved style names to {names_path}")


def main():
    """Main execution function"""
    print("Inter-class Analysis")

    if not os.path.exists(config.FEATURES_DIR):
        print(f"ERROR: Features directory not found: {config.FEATURES_DIR}")
        print("Please run wikiart_extract_features.py first")
        return

    features = load_features(config.FEATURES_DIR)
    distance_matrices = analyze_inter_class_distances(features, metric=config.DISTANCE_METRIC)
    save_distance_matrices(distance_matrices)
    analyze_nearest_neighbors(distance_matrices)
    outlier_scores = analyze_outliers(distance_matrices, features)
    identify_similar_clusters(distance_matrices)
    generate_visualizations(distance_matrices, outlier_scores)

    print("Done")
    print(f"Results saved to: {config.RESULTS_DIR}")
    print(f"Visualizations saved to: {config.VISUALIZATION_DIR}")


if __name__ == "__main__":
    main()
