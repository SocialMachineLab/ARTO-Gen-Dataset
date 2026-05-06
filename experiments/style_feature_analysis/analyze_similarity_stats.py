"""Analyze similarity statistics for WikiArt images using OpenCLIP"""

import os
import json
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import sys
import numpy as np


def load_metadata(metadata_path: str) -> Dict:
    """Load metadata from JSON file"""
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def setup_openclip_model(model_name: str, pretrained: str, device: str = 'cuda'):
    """Initialize OpenCLIP model"""
    print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def calculate_similarity_stats(
    sampled_images: Dict[str, List[str]], 
    styles: List[str], 
    model,
    preprocess,
    tokenizer,
    device: str
):
    """Calculate similarity statistics for each style"""
    style_scores = {style: [] for style in styles}
    
    STYLE_PROMPTS = {
        "Baroque": "a Baroque painting",
        "Impressionism": "an Impressionist painting",
        "Chinese Ink Painting": "a traditional Chinese ink and wash painting",
        "Chinese_painting": "a traditional Chinese ink and wash painting",
        "Realism": "a Realist painting",
        "Neoclassicism": "a Neoclassical painting"
    }

    text_descriptions = []
    for style in styles:
        prompt = STYLE_PROMPTS.get(style, f"a painting in the style of {style}")
        text_descriptions.append(prompt)
    
    print(f"Prompts used: {text_descriptions}")

    text_tokens = tokenizer(text_descriptions).to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print(f"Starting similarity analysis for {len(styles)} styles")
    
    for true_style, image_paths in sampled_images.items():
        if true_style not in styles:
            continue
            
        true_style_idx = styles.index(true_style)
        
        print(f"Analyzing {true_style} ({len(image_paths)} images)")
        
        for img_path in tqdm(image_paths, leave=False):
            if not os.path.exists(img_path):
                continue
                
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    logits = (100.0 * image_features @ text_features.T)
                    probs = logits.softmax(dim=-1)
                    
                true_class_score = probs[0][true_style_idx].item()
                style_scores[true_style].append(true_class_score)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return style_scores


def generate_statistics(style_scores: Dict[str, List[float]], output_dir: str):
    """Generate and save statistical summary"""
    stats_summary = []
    
    print("Statistical Analysis (Confidence Scores for True Class)")
    print(f"{'Style':<25} | {'Mean':<6} | {'Std':<6} | {'Min':<6} | {'25%':<6} | {'50%':<6} | {'75%':<6} | {'Max':<6}")
    print("-" * 95)

    for style, scores in style_scores.items():
        if not scores:
            continue
            
        scores_np = np.array(scores)
        
        mean_val = np.mean(scores_np)
        std_val = np.std(scores_np)
        min_val = np.min(scores_np)
        max_val = np.max(scores_np)
        quartiles = np.percentile(scores_np, [25, 50, 75])
        
        print(f"{style:<25} | {mean_val:.4f} | {std_val:.4f} | {min_val:.4f} | {quartiles[0]:.4f} | {quartiles[1]:.4f} | {quartiles[2]:.4f} | {max_val:.4f}")
        
        stats_summary.append({
            "style": style,
            "count": len(scores),
            "mean": float(mean_val),
            "std": float(std_val),
            "min": float(min_val),
            "q1_25": float(quartiles[0]),
            "median_50": float(quartiles[1]),
            "q3_75": float(quartiles[2]),
            "max": float(max_val),
            "raw_scores": scores
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'similarity_distribution_stats.json')
    
    with open(output_path, 'w') as f:
        json.dump(stats_summary, f, indent=2)
    
    print(f"Detailed statistics saved to: {output_path}")


def main():
    print("OpenCLIP Similarity Distribution Analysis")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    METADATA_FILE = os.path.join(SCRIPT_DIR, 'outputs', 'metadata', 'sampled_images.json')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'clip_similar_results')
    
    try:
        metadata = load_metadata(METADATA_FILE)
        raw_sampled_images = metadata.get('sampled_images', {})
        
        TARGET_STYLES = ["Baroque", "Neoclassicism", "Realism", "Impressionism", "Chinese Ink Painting"]
        
        style_map = {
            "Baroque": "Baroque",
            "Neoclassicism": "Neoclassicism",
            "Realism": "Realism",
            "Impressionism": "Impressionism",
            "Chinese_painting": "Chinese Ink Painting",
            "Chinese Ink Painting": "Chinese Ink Painting",
            "Ink and wash painting": "Chinese Ink Painting"
        }

        filtered_images = {}
        for k, v in raw_sampled_images.items():
            if k in style_map:
                clean_name = style_map[k]
                if clean_name in TARGET_STYLES:
                    filtered_images[clean_name] = [p for p in v if os.path.exists(p)]

        styles = sorted(list(filtered_images.keys()))
        print(f"Analyzing styles: {styles}")
        
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return

    model, preprocess, tokenizer = setup_openclip_model(
        model_name='ViT-SO400M-14-SigLIP-384', 
        pretrained='webli',
        device=device
    )

    style_scores = calculate_similarity_stats(
        filtered_images, styles, model, preprocess, tokenizer, device
    )
    
    generate_statistics(style_scores, OUTPUT_DIR)


if __name__ == "__main__":
    main()
