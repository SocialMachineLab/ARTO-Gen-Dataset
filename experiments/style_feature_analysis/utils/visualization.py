"""
Visualization utilities for style analysis
Includes heatmaps, dendrograms, and other plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import List, Dict, Optional
import pandas as pd


def plot_distance_heatmap(
    distance_matrix: np.ndarray,
    style_names: List[str],
    title: str = "Inter-class Distance Matrix",
    figsize: tuple = (12, 10),
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Plot a heatmap of the distance matrix

    Args:
        distance_matrix: (N, N) symmetric distance matrix
        style_names: List of style names
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure (optional)
        dpi: DPI for saved figure
    """
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(
        distance_matrix,
        xticklabels=style_names,
        yticklabels=style_names,
        cmap=cmap,
        annot=True,
        fmt='.3f',
        square=True,
        cbar_kws={'label': 'Distance'}
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Style', fontsize=12)
    plt.ylabel('Style', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved heatmap to {save_path}")

    plt.close()


def plot_dendrogram(
    distance_matrix: np.ndarray,
    style_names: List[str],
    title: str = "Hierarchical Clustering of Styles",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Plot a dendrogram showing hierarchical clustering of styles

    Args:
        distance_matrix: (N, N) distance matrix
        style_names: List of style names
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure (optional)
        dpi: DPI for saved figure
    """
    plt.figure(figsize=figsize)

    # Convert distance matrix to condensed form
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(distance_matrix)

    # Perform hierarchical clustering
    Z = linkage(condensed_dist, method='average')

    # Plot dendrogram
    dendrogram(
        Z,
        labels=style_names,
        leaf_font_size=12,
        leaf_rotation=45
    )

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Style', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved dendrogram to {save_path}")

    plt.close()


def plot_intra_class_comparison(
    intra_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'variance',
    title: str = "Intra-class Variance by Style",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Plot bar chart comparing intra-class metrics across styles

    Args:
        intra_metrics: Dictionary mapping style -> metrics dict
        metric_name: Which metric to plot
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    styles = list(intra_metrics.keys())
    values = [intra_metrics[style][metric_name] for style in styles]

    # Sort by value
    sorted_indices = np.argsort(values)
    styles = [styles[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]

    plt.figure(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0, 1, len(styles)))

    plt.barh(styles, values, color=colors)
    plt.xlabel(metric_name.replace('_', ' ').title(), fontsize=12)
    plt.ylabel('Style', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")

    plt.close()


def plot_combination_comparison(
    combinations_scores: Dict[str, float],
    target_combination: str,
    top_k: int = 10,
    title: str = "Top Style Combinations by Total Distance",
    figsize: tuple = (14, 8),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Plot comparison of different style combinations

    Args:
        combinations_scores: Dictionary mapping combination -> score
        target_combination: The combination to highlight
        top_k: Number of top combinations to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    # Sort by score (descending)
    sorted_combos = sorted(combinations_scores.items(),
                          key=lambda x: x[1], reverse=True)

    # Take top k
    top_combos = sorted_combos[:top_k]

    labels = []
    scores = []
    colors = []

    for combo, score in top_combos:
        labels.append(combo)
        scores.append(score)
        # Highlight target combination
        if combo == target_combination:
            colors.append('red')
        else:
            colors.append('steelblue')

    plt.figure(figsize=figsize)
    plt.barh(range(len(labels)), scores, color=colors)
    plt.yticks(range(len(labels)), labels, fontsize=9)
    plt.xlabel('Total Inter-class Distance', fontsize=12)
    plt.ylabel('Style Combination', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest score on top

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Target Combination'),
        Patch(facecolor='steelblue', label='Other Combinations')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved combination comparison to {save_path}")

    plt.close()


def plot_multi_model_consistency(
    model_distances: Dict[str, np.ndarray],
    style_names: List[str],
    figsize: tuple = (16, 4),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Plot distance matrices from multiple models side-by-side

    Args:
        model_distances: Dictionary mapping model_name -> distance_matrix
        style_names: List of style names
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    n_models = len(model_distances)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, dist_matrix) in zip(axes, model_distances.items()):
        sns.heatmap(
            dist_matrix,
            xticklabels=style_names,
            yticklabels=style_names,
            cmap='viridis',
            ax=ax,
            cbar=True,
            square=True
        )
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.suptitle('Distance Matrices Across Models', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved multi-model comparison to {save_path}")

    plt.close()


def plot_outlier_scores(
    outlier_scores: Dict[str, float],
    title: str = "Style Uniqueness Scores",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
    dpi: int = 300
):
    """
    Plot outlier/uniqueness scores for each style

    Args:
        outlier_scores: Dictionary mapping style -> outlier score
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        dpi: DPI for saved figure
    """
    styles = list(outlier_scores.keys())
    scores = [outlier_scores[style] for style in styles]

    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]  # Descending
    styles = [styles[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]

    plt.figure(figsize=figsize)
    colors = plt.cm.plasma(np.linspace(0, 1, len(styles)))

    plt.barh(styles, scores, color=colors)
    plt.xlabel('Average Distance to Other Styles', fontsize=12)
    plt.ylabel('Style', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved outlier scores plot to {save_path}")

    plt.close()


def create_summary_table(
    intra_metrics: Dict[str, Dict[str, Dict[str, float]]],
    style_names: List[str],
    model_names: List[str],
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a summary table of intra-class metrics across models

    Args:
        intra_metrics: Nested dict: style -> model -> metrics
        style_names: List of style names
        model_names: List of model names
        save_path: Path to save CSV (optional)

    Returns:
        pandas DataFrame
    """
    rows = []

    for style in style_names:
        row = {'Style': style}
        for model in model_names:
            metrics = intra_metrics[style][model]
            row[f'{model}_variance'] = metrics['variance']
            row[f'{model}_avg_dist'] = metrics['avg_pairwise_distance']
        rows.append(row)

    df = pd.DataFrame(rows)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved summary table to {save_path}")

    return df
