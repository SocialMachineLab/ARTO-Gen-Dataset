"""Compare generated images vs real WikiArt images to evaluate style reproduction quality"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.spatial.distance import cosine

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from utils.feature_extractors import MultiModelFeatureExtractor


STYLE_NAME_MAPPING = {
    'Chinese_painting': 'Ink and wash painting',
    'Ink and wash painting': 'Chinese_painting',
}


def extract_validation_features():
    """Extract features from all validation images using existing extractor"""
    print("Extracting features from validation images")

    validation_dir = 'outputs/validation_images'
    output_dir = 'outputs/validation_features'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(validation_dir):
        print(f"Validation directory not found: {validation_dir}")
        return {}

    style_dirs = [d for d in Path(validation_dir).iterdir() if d.is_dir()]
    if not style_dirs:
        print(f"No style directories found in {validation_dir}")
        return {}

    print(f"Found {len(style_dirs)} validation style directories")

    print("Initializing feature extraction models")
    extractor = MultiModelFeatureExtractor(config.MODELS, device=config.DEVICE)

    validation_features = {}

    for style_dir in style_dirs:
        style = style_dir.name
        print(f"Processing {style}")

        image_files = list(style_dir.glob('*.png'))
        if not image_files:
            print("  No images found")
            continue

        image_paths = [str(p) for p in image_files]
        print(f"  Found {len(image_paths)} images")

        features_dict = extractor.extract_all_features(image_paths, config.BATCH_SIZE)

        for model_name, features in features_dict.items():
            model_output_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)
            save_path = os.path.join(model_output_dir, f"{style}.npy")
            np.save(save_path, features)
            print(f"  Saved {model_name}: {features.shape}")

        validation_features[style] = features_dict

    print("Feature extraction complete")
    return validation_features


def load_wikiart_features():
    """Load pre-extracted WikiArt features by scanning features directory"""
    print("Loading WikiArt features")

    features_dir = config.FEATURES_DIR
    wikiart_features = {}

    if not os.path.exists(features_dir):
        print(f"Features directory not found: {features_dir}")
        return {}

    all_styles = set()
    for model_name in config.MODELS.keys():
        model_dir = os.path.join(features_dir, model_name)
        if os.path.exists(model_dir):
            feature_files = list(Path(model_dir).glob('*.npy'))
            for f in feature_files:
                style_name = f.stem
                all_styles.add(style_name)

    print(f"Found {len(all_styles)} styles in features directory")

    for style in all_styles:
        wikiart_features[style] = {}

        for model_name in config.MODELS.keys():
            feature_path = os.path.join(features_dir, model_name, f"{style}.npy")

            if os.path.exists(feature_path):
                feat = np.load(feature_path)
                wikiart_features[style][model_name] = feat
                print(f"  Loaded {style:30s} - {model_name:20s}: {feat.shape}")
            else:
                print(f"  Not found: {style:30s} - {model_name}")

    return wikiart_features


def compute_distribution_similarity(real_features, gen_features):
    """Compute similarity between real and generated feature distributions"""
    metrics = {}

    real_centroid = real_features.mean(axis=0)
    gen_centroid = gen_features.mean(axis=0)
    centroid_distance = cosine(real_centroid, gen_centroid)
    metrics['centroid_cosine_distance'] = float(centroid_distance)
    metrics['centroid_cosine_similarity'] = float(1 - centroid_distance)

    l2_distance = np.linalg.norm(real_centroid - gen_centroid)
    metrics['centroid_l2_distance'] = float(l2_distance)

    pairwise_distances = []
    for gen_feat in gen_features:
        distances = [cosine(gen_feat, real_feat) for real_feat in real_features]
        pairwise_distances.append(min(distances))
    metrics['mean_nearest_neighbor_distance'] = float(np.mean(pairwise_distances))

    threshold = 0.3
    coverage_count = 0
    for real_feat in real_features:
        distances = [cosine(real_feat, gen_feat) for gen_feat in gen_features]
        if min(distances) < threshold:
            coverage_count += 1
    metrics['coverage_rate'] = float(coverage_count / len(real_features))

    real_variance = np.mean(np.var(real_features, axis=0))
    gen_variance = np.mean(np.var(gen_features, axis=0))
    metrics['real_variance'] = float(real_variance)
    metrics['generated_variance'] = float(gen_variance)
    metrics['variance_ratio'] = float(gen_variance / real_variance) if real_variance > 0 else 0.0

    return metrics


def compare_features(wikiart_features, validation_features):
    """Compare WikiArt and validation features"""
    print("Comparing real vs generated features")

    print("Style Mapping:")
    for val_style in sorted(validation_features.keys()):
        wiki_style = STYLE_NAME_MAPPING.get(val_style, val_style)
        status = "Match" if wiki_style in wikiart_features else "No Match"
        print(f"  {val_style:<30s} -> {wiki_style:<30s} ({status})")

    comparison_results = {}

    for val_style in validation_features.keys():
        wiki_style = STYLE_NAME_MAPPING.get(val_style, val_style)

        if wiki_style not in wikiart_features:
            print(f"  {val_style} (validation) has no matching WikiArt features")
            continue

        print(f"Analyzing {val_style}")

        style_results = {}

        for model_name in config.MODELS.keys():
            if model_name not in wikiart_features[wiki_style]:
                continue
            if model_name not in validation_features[val_style]:
                continue

            real_feat = wikiart_features[wiki_style][model_name]
            gen_feat = validation_features[val_style][model_name]

            if len(real_feat) == 0 or len(gen_feat) == 0:
                continue

            metrics = compute_distribution_similarity(real_feat, gen_feat)
            style_results[model_name] = metrics

            print(f"  {model_name:20s}: Sim={metrics['centroid_cosine_similarity']:.4f}, "
                  f"Dist={metrics['mean_nearest_neighbor_distance']:.4f}, "
                  f"Cov={metrics['coverage_rate']:.4f}")

        if style_results:
            avg_centroid_sim = np.mean([m['centroid_cosine_similarity'] for m in style_results.values()])
            avg_nn_dist = np.mean([m['mean_nearest_neighbor_distance'] for m in style_results.values()])
            avg_coverage = np.mean([m['coverage_rate'] for m in style_results.values()])

            style_results['aggregate'] = {
                'avg_centroid_similarity': float(avg_centroid_sim),
                'avg_nn_distance': float(avg_nn_dist),
                'avg_coverage': float(avg_coverage)
            }

            print(f"  Aggregate: Sim={avg_centroid_sim:.4f}, Dist={avg_nn_dist:.4f}, Cov={avg_coverage:.4f}")

        comparison_results[val_style] = style_results

    return comparison_results


def generate_visualizations(comparison_results):
    """Generate comparison visualizations"""
    print("Generating visualizations")

    viz_dir = os.path.join(config.OUTPUT_DIR, 'generation_quality')
    os.makedirs(viz_dir, exist_ok=True)

    styles = []
    centroid_sims = []
    nn_distances = []
    coverage_rates = []

    for style, results in comparison_results.items():
        if 'aggregate' in results:
            styles.append(style)
            centroid_sims.append(results['aggregate']['avg_centroid_similarity'])
            nn_distances.append(results['aggregate']['avg_nn_distance'])
            coverage_rates.append(results['aggregate']['avg_coverage'])

    if not styles:
        print("No results to visualize")
        return

    sorted_indices = np.argsort(centroid_sims)[::-1]
    styles = [styles[i] for i in sorted_indices]
    centroid_sims = [centroid_sims[i] for i in sorted_indices]
    nn_distances = [nn_distances[i] for i in sorted_indices]
    coverage_rates = [coverage_rates[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['green' if s > 0.85 else 'orange' if s > 0.75 else 'red' for s in centroid_sims]
    bars = ax.barh(range(len(styles)), centroid_sims, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
    ax.set_yticks(range(len(styles)))
    ax.set_yticklabels(styles, fontsize=10)
    ax.set_xlabel('Centroid Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Generation Quality: Feature Space Centroid Similarity\n(Generated vs Real WikiArt)',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    for i, (bar, sim) in enumerate(zip(bars, centroid_sims)):
        ax.text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'centroid_similarity_by_style.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].barh(range(len(styles)), centroid_sims, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_yticks(range(len(styles)))
    axes[0].set_yticklabels(styles, fontsize=9)
    axes[0].set_xlabel('Similarity', fontsize=11, fontweight='bold')
    axes[0].set_title('Centroid Similarity\n(Higher = Better)', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, 1.0)
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(range(len(styles)), nn_distances, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_yticks(range(len(styles)))
    axes[1].set_yticklabels(styles, fontsize=9)
    axes[1].set_xlabel('Distance', fontsize=11, fontweight='bold')
    axes[1].set_title('Nearest Neighbor Distance\n(Lower = Better)', fontsize=12, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)

    axes[2].barh(range(len(styles)), coverage_rates, color='lightgreen', alpha=0.7, edgecolor='black')
    axes[2].set_yticks(range(len(styles)))
    axes[2].set_yticklabels(styles, fontsize=9)
    axes[2].set_xlabel('Coverage', fontsize=11, fontweight='bold')
    axes[2].set_title('Coverage Rate\n(Higher = Better)', fontsize=12, fontweight='bold')
    axes[2].set_xlim(0, 1.0)
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'multi_metric_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    models = list(config.MODELS.keys())
    heatmap_data = []

    for style in styles:
        row = []
        for model in models:
            if model in comparison_results[style]:
                sim = comparison_results[style][model]['centroid_cosine_similarity']
                row.append(sim)
            else:
                row.append(np.nan)
        heatmap_data.append(row)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.6, vmax=1.0,
                xticklabels=models, yticklabels=styles, cbar_kws={'label': 'Centroid Similarity'},
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_title('Per-Model Centroid Similarity Heatmap\n(Generated vs Real)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Art Style', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'per_model_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved visualizations")


def save_results(comparison_results):
    """Save comparison results"""
    print("Saving results")

    output_dir = os.path.join(config.OUTPUT_DIR, 'generation_quality')
    os.makedirs(output_dir, exist_ok=True)

    json_file = os.path.join(output_dir, 'generation_quality_report.json')
    with open(json_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    print(f"Saved JSON: {json_file}")

    rows = []
    for style, results in comparison_results.items():
        if 'aggregate' not in results:
            continue

        row = {
            'style': style,
            'centroid_similarity': results['aggregate']['avg_centroid_similarity'],
            'nn_distance': results['aggregate']['avg_nn_distance'],
            'coverage_rate': results['aggregate']['avg_coverage']
        }

        for model_name in config.MODELS.keys():
            if model_name in results:
                row[f'{model_name}_centroid_sim'] = results[model_name]['centroid_cosine_similarity']
                row[f'{model_name}_nn_dist'] = results[model_name]['mean_nearest_neighbor_distance']

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values('centroid_similarity', ascending=False)

    csv_file = os.path.join(output_dir, 'generation_quality_summary.csv')
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV: {csv_file}")

    print("\nGeneration Quality Summary:")
    print("Top 3 Best Reproduced Styles:")
    for i, row in df.head(3).iterrows():
        print(f"  {i+1}. {row['style']:25s}: {row['centroid_similarity']:.4f}")

    print("\nBottom 3 Worst Reproduced Styles:")
    for i, row in df.tail(3).iterrows():
        print(f"  {len(df)-2+i-len(df)+1}. {row['style']:25s}: {row['centroid_similarity']:.4f}")

    print(f"\nOverall Average Centroid Similarity: {df['centroid_similarity'].mean():.4f}")
    print(f"Overall Average Coverage Rate: {df['coverage_rate'].mean():.4f}")


def main():
    """Main execution"""
    print("Generation Model Evaluation")

    validation_features = extract_validation_features()
    wikiart_features = load_wikiart_features()
    comparison_results = compare_features(wikiart_features, validation_features)

    if comparison_results:
        generate_visualizations(comparison_results)

    save_results(comparison_results)

    print("Evaluation Complete")
    print(f"Results saved to: {os.path.join(config.OUTPUT_DIR, 'generation_quality')}/")


if __name__ == '__main__':
    main()
