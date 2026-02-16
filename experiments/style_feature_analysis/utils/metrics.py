"""
Metrics for analyzing feature distributions
Includes intra-class and inter-class distance calculations
"""

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from typing import Dict, List, Tuple


def compute_intra_class_variance(features: np.ndarray) -> float:
    """
    Compute intra-class variance (lower is more consistent)

    Args:
        features: Array of shape (N, D) where N is number of samples

    Returns:
        Variance value (scalar)
    """
    if len(features) < 2:
        return 0.0

    centroid = np.mean(features, axis=0)
    variance = np.mean(np.sum((features - centroid) ** 2, axis=1))
    return float(variance)


def compute_avg_pairwise_distance(features: np.ndarray, metric='cosine') -> float:
    """
    Compute average pairwise distance within a class (lower is more consistent)

    Args:
        features: Array of shape (N, D)
        metric: 'cosine' or 'euclidean'

    Returns:
        Average pairwise distance
    """
    if len(features) < 2:
        return 0.0

    distances = pdist(features, metric=metric)
    return float(np.mean(distances))


def compute_std_pairwise_distance(features: np.ndarray, metric='cosine') -> float:
    """
    Compute standard deviation of pairwise distances (lower is more consistent)

    Args:
        features: Array of shape (N, D)
        metric: 'cosine' or 'euclidean'

    Returns:
        Standard deviation of pairwise distances
    """
    if len(features) < 2:
        return 0.0

    distances = pdist(features, metric=metric)
    return float(np.std(distances))


def compute_centroid_distance_matrix(
    features_dict: Dict[str, np.ndarray],
    metric='cosine'
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute distance matrix between centroids of different styles

    Args:
        features_dict: Dictionary mapping style_name -> feature array (N, D)
        metric: 'cosine' or 'euclidean'

    Returns:
        distance_matrix: (n_styles, n_styles) array
        style_names: List of style names in the same order
    """
    style_names = list(features_dict.keys())
    n_styles = len(style_names)

    # Compute centroids
    centroids = []
    for style in style_names:
        centroid = np.mean(features_dict[style], axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)  # Shape: (n_styles, D)

    # Compute distance matrix
    if metric == 'cosine':
        distance_matrix = cosine_distances(centroids)
    elif metric == 'euclidean':
        distance_matrix = euclidean_distances(centroids)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return distance_matrix, style_names


def compute_avg_inter_class_distance(distance_matrix: np.ndarray) -> float:
    """
    Compute average inter-class distance (excluding diagonal)

    Args:
        distance_matrix: (N, N) symmetric distance matrix

    Returns:
        Average distance
    """
    n = distance_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)  # Exclude diagonal
    return float(np.mean(distance_matrix[mask]))


def find_nearest_neighbors(
    distance_matrix: np.ndarray,
    style_names: List[str],
    top_k: int = 3
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find k nearest neighbors for each style

    Args:
        distance_matrix: (N, N) distance matrix
        style_names: List of style names
        top_k: Number of neighbors to return

    Returns:
        Dictionary mapping style -> [(neighbor, distance), ...]
    """
    neighbors = {}

    for i, style in enumerate(style_names):
        # Get distances to all other styles
        distances = distance_matrix[i].copy()
        distances[i] = np.inf  # Exclude self

        # Find top k nearest
        nearest_indices = np.argsort(distances)[:top_k]

        neighbors[style] = [
            (style_names[idx], distances[idx])
            for idx in nearest_indices
        ]

    return neighbors


def compute_outlier_score(
    target_style: str,
    features_dict: Dict[str, np.ndarray],
    distance_matrix: np.ndarray,
    style_names: List[str]
) -> float:
    """
    Compute how much a style is an outlier (higher = more unique)

    Args:
        target_style: The style to evaluate
        features_dict: Dictionary of all features
        distance_matrix: Pre-computed distance matrix
        style_names: List of style names

    Returns:
        Average distance to all other styles
    """
    target_idx = style_names.index(target_style)
    distances = distance_matrix[target_idx].copy()
    distances[target_idx] = 0  # Exclude self

    avg_distance = np.mean(distances[distances > 0])
    return float(avg_distance)


def compute_combination_score(
    selected_styles: List[str],
    distance_matrix: np.ndarray,
    style_names: List[str]
) -> float:
    """
    Compute total inter-class distance for a combination of styles
    (Higher is better - more diverse)

    Args:
        selected_styles: List of selected style names
        distance_matrix: Full distance matrix
        style_names: All style names

    Returns:
        Sum of pairwise distances
    """
    indices = [style_names.index(s) for s in selected_styles]

    total_distance = 0.0
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            total_distance += distance_matrix[indices[i], indices[j]]

    return float(total_distance)


def compute_min_neighbor_distance(
    style: str,
    distance_matrix: np.ndarray,
    style_names: List[str]
) -> float:
    """
    Compute distance to the nearest neighbor (higher = more unique)

    Args:
        style: Style name
        distance_matrix: Distance matrix
        style_names: Style names

    Returns:
        Distance to nearest neighbor
    """
    idx = style_names.index(style)
    distances = distance_matrix[idx].copy()
    distances[idx] = np.inf
    return float(np.min(distances))


def compute_silhouette_coefficient(
    features_dict: Dict[str, np.ndarray],
    metric='cosine'
) -> Dict[str, float]:
    """
    Compute silhouette coefficient for each style
    (Measures how similar a style is to its own cluster vs other clusters)

    Args:
        features_dict: Dictionary of features per style
        metric: Distance metric

    Returns:
        Dictionary mapping style -> silhouette score
    """
    from sklearn.metrics import silhouette_samples

    # Concatenate all features
    all_features = []
    labels = []
    style_names = list(features_dict.keys())

    for i, style in enumerate(style_names):
        features = features_dict[style]
        all_features.append(features)
        labels.extend([i] * len(features))

    all_features = np.vstack(all_features)
    labels = np.array(labels)

    # Compute silhouette scores
    silhouette_vals = silhouette_samples(all_features, labels, metric=metric)

    # Average by style
    scores = {}
    start_idx = 0
    for style in style_names:
        n_samples = len(features_dict[style])
        scores[style] = float(np.mean(silhouette_vals[start_idx:start_idx + n_samples]))
        start_idx += n_samples

    return scores


def compute_intra_class_metrics(features: np.ndarray, metric='cosine') -> Dict[str, float]:
    """
    Compute all intra-class metrics for a single style

    Args:
        features: Feature array (N, D)
        metric: Distance metric

    Returns:
        Dictionary with multiple metrics
    """
    return {
        'variance': compute_intra_class_variance(features),
        'avg_pairwise_distance': compute_avg_pairwise_distance(features, metric),
        'std_pairwise_distance': compute_std_pairwise_distance(features, metric),
        'n_samples': len(features)
    }
