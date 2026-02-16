"""Extract features from WikiArt images using multiple vision models"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.feature_extractors import MultiModelFeatureExtractor


def get_image_paths_from_folder(folder_path: str) -> List[str]:
    """Get all image paths from a folder"""
    image_paths = []
    folder = Path(folder_path)

    if not folder.exists():
        print(f"Warning: Folder {folder_path} does not exist")
        return []

    for ext in config.IMAGE_EXTENSIONS:
        image_paths.extend(list(folder.glob(f"*{ext}")))
        image_paths.extend(list(folder.glob(f"*{ext.upper()}")))

    return [str(p) for p in image_paths]


def sample_images_per_style(
    image_dir: str,
    styles: List[str],
    sample_size: int,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Sample images for each style

    Args:
        image_dir: Root directory containing style subfolders
        styles: List of style names
        sample_size: Number of images to sample per style
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping style -> list of image paths
    """
    random.seed(random_seed)
    sampled_images = {}

    print("Sampling images")

    for style in styles:
        style_folder = os.path.join(image_dir, style)
        all_images = get_image_paths_from_folder(style_folder)

        if not all_images:
            print(f"Warning: No images found for style '{style}' in {style_folder}")
            sampled_images[style] = []
            continue

        if len(all_images) <= sample_size:
            sampled = all_images
            print(f"{style}: Using all {len(all_images)} images")
        else:
            sampled = random.sample(all_images, sample_size)
            print(f"{style}: Sampled {len(sampled)} of {len(all_images)} images")

        sampled_images[style] = sampled

    return sampled_images


def save_sampling_metadata(sampled_images: Dict[str, List[str]], save_path: str):
    """Save the record of which images were sampled"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    metadata = {
        'sample_size': config.SAMPLE_SIZE,
        'random_seed': config.RANDOM_SEED,
        'styles': list(sampled_images.keys()),
        'sampled_images': sampled_images,
        'counts': {style: len(paths) for style, paths in sampled_images.items()}
    }

    with open(save_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved sampling metadata to {save_path}")


def extract_and_save_features(
    sampled_images: Dict[str, List[str]],
    models_config: Dict,
    output_dir: str,
    batch_size: int
):
    """
    Extract features using all models and save to disk

    Args:
        sampled_images: Dictionary mapping style -> image paths
        models_config: Model configuration
        output_dir: Directory to save features
        batch_size: Batch size for processing
    """
    print("Initializing models")
    extractor = MultiModelFeatureExtractor(models_config, device=config.DEVICE)

    print("Extracting features")

    for style, image_paths in sampled_images.items():
        if not image_paths:
            print(f"Skipping {style} (no images)")
            continue

        print(f"Processing {style} ({len(image_paths)} images)")

        try:
            features_dict = extractor.extract_all_features(image_paths, batch_size)

            for model_name, features in features_dict.items():
                model_output_dir = os.path.join(output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)

                output_path = os.path.join(model_output_dir, f"{style}.npy")
                np.save(output_path, features)
                print(f"  Saved {model_name} features to {output_path}")

        except Exception as e:
            print(f"Error extracting features for {style}: {e}")
            continue

    print("Feature extraction complete")


def main():
    """Main execution function"""
    print("Style Feature Extraction")

    if config.IMAGE_DIR == '/path/to/your/images':
        print("ERROR: Please update IMAGE_DIR in config.py with your actual image folder path")
        print("Expected structure:")
        print("  IMAGE_DIR/")
        print("    Impressionism/")
        print("    Realism/")
        print("    ...")
        print("    Chinese_painting/")
        return

    if not os.path.exists(config.IMAGE_DIR):
        print(f"ERROR: Image directory does not exist: {config.IMAGE_DIR}")
        return

    os.makedirs(config.FEATURES_DIR, exist_ok=True)
    os.makedirs(config.METADATA_DIR, exist_ok=True)

    print(f"Image directory: {config.IMAGE_DIR}")
    print(f"Styles: {len(config.STYLES)}")
    print(f"Sample size per style: {config.SAMPLE_SIZE}")
    print(f"Models: {list(config.MODELS.keys())}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Device: {config.DEVICE}")

    sampled_images = sample_images_per_style(
        config.IMAGE_DIR,
        config.STYLES,
        config.SAMPLE_SIZE,
        config.RANDOM_SEED
    )

    metadata_path = os.path.join(config.METADATA_DIR, 'sampled_images.json')
    save_sampling_metadata(sampled_images, metadata_path)

    extract_and_save_features(
        sampled_images,
        config.MODELS,
        config.FEATURES_DIR,
        config.BATCH_SIZE
    )

    print("Done")
    print(f"Features saved to: {config.FEATURES_DIR}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()
