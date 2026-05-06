"""Analyze and summarize object detection results from WikiArt and Generated images"""

import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config


def load_detection_results(base_dir):
    """Load detection results from JSON files"""
    results = {}

    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return results

    style_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for style in style_dirs:
        style_path = os.path.join(base_dir, style)

        # Look for result files
        wikiart_file = os.path.join(style_path, f"{style}_wikiart_results.json")
        generated_file = os.path.join(style_path, f"{style}_generated_results.json")

        if os.path.exists(wikiart_file):
            with open(wikiart_file, 'r') as f:
                results[f"wikiart_{style}"] = json.load(f)

        if os.path.exists(generated_file):
            with open(generated_file, 'r') as f:
                results[f"generated_{style}"] = json.load(f)

    return results


def compute_detection_stats(results):
    """Compute statistics from detection results"""
    stats = []

    for key, data in results.items():
        source, style = key.split('_', 1)

        total_images = data['total_images']
        if total_images == 0:
            continue

        per_image = data['per_image_results']

        # Detections per model
        yolo_counts = [d['counts']['yolo'] for d in per_image]
        owl_counts = [d['counts']['owl'] for d in per_image]
        dino_counts = [d['counts']['dino'] for d in per_image]

        # Detection rates (at least one object detected)
        yolo_rate = sum(1 for c in yolo_counts if c > 0) / total_images
        owl_rate = sum(1 for c in owl_counts if c > 0) / total_images
        dino_rate = sum(1 for c in dino_counts if c > 0) / total_images

        # VLM usage stats
        vlm_used = sum(1 for d in per_image if d.get('used_vlm', False))
        vlm_rate = vlm_used / total_images

        # Consistency (only for generated images where we compute it)
        avg_iou = np.nan
        if source == 'generated':
            ious = [d['bbox_consistency']['avg_iou'] for d in per_image
                    if d.get('bbox_consistency')]
            if ious:
                avg_iou = np.mean(ious)

        stats.append({
            'source': source,
            'style': style,
            'images': total_images,
            'yolo_rate': yolo_rate,
            'owl_rate': owl_rate,
            'dino_rate': dino_rate,
            'vlm_rate': vlm_rate,
            'avg_objects_yolo': np.mean(yolo_counts),
            'avg_objects_owl': np.mean(owl_counts),
            'avg_objects_dino': np.mean(dino_counts),
            'consistency_iou': avg_iou
        })

    return pd.DataFrame(stats)


def visualize_stats(df, output_dir):
    """Generate visualizations for detection statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Detection Rate Comparison
    long_df = pd.melt(df, id_vars=['source', 'style'],
                      value_vars=['yolo_rate', 'owl_rate', 'dino_rate'],
                      var_name='model', value_name='detection_rate')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=long_df, x='style', y='detection_rate', hue='model')
    plt.title('Detection Rate by Model and Style')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_rates.png'))
    plt.close()

    # 2. Object Count Comparison
    long_count_df = pd.melt(df, id_vars=['source', 'style'],
                            value_vars=['avg_objects_yolo', 'avg_objects_owl', 'avg_objects_dino'],
                            var_name='model', value_name='avg_count')

    plt.figure(figsize=(12, 6))
    sns.barplot(data=long_count_df, x='style', y='avg_count', hue='model')
    plt.title('Average Object Count per Image')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'object_counts.png'))
    plt.close()

    # 3. VLM Usage
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='style', y='vlm_rate', hue='source')
    plt.title('VLM Fallback Usage Rate')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vlm_usage.png'))
    plt.close()


def main():
    print("Analyzing Detection Results")

    wikiart_dir = os.path.join('outputs', 'three_model_consensus', 'wikiart', 'detections')
    generated_dir = os.path.join('outputs', 'three_model_consensus', 'generated', 'detections')

    wikiart_results = load_detection_results(wikiart_dir)
    generated_results = load_detection_results(generated_dir)

    all_results = {**wikiart_results, **generated_results}

    if not all_results:
        print("No detection results found")
        return

    df = compute_detection_stats(all_results)
    print("\nDetection Statistics:")
    print(df.round(3).to_string())

    output_dir = os.path.join('outputs', 'three_model_consensus', 'analysis')
    visualize_stats(df, output_dir)
    df.to_csv(os.path.join(output_dir, 'detection_summary.csv'), index=False)
    print(f"\nAnalysis saved to {output_dir}")


if __name__ == '__main__':
    main()
