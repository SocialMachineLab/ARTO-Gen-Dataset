"""Classify WikiArt images using OpenCLIP"""

import os
import json
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from typing import List, Dict
import sys
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def load_metadata(metadata_path: str) -> Dict:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


def setup_openclip_model(model_name: str = 'ViT-B-32', pretrained: str = 'laion2b_s34b_b79k', device: str = 'cuda'):
    print(f"Loading OpenCLIP model: {model_name} ({pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer


def classify_images(
    sampled_images: Dict[str, List[str]], 
    styles: List[str], 
    model,
    preprocess,
    tokenizer,
    device: str
):
    results = []
    
    STYLE_PROMPTS = {
        "Baroque": "Baroque painting with dramatic lighting and emotional intensity",
        "Chinese Ink Painting": "Chinese ink painting",
        "Impressionism": "Impressionist painting with visible brushstrokes and natural outdoor light",
        "Neoclassicism": "Neoclassical painting with smooth surfaces, clear contours, and restrained colors",
        "Post-Impressionism": "Post-Impressionist painting with prominent black outlines, flat decorative patterns, and intense non-realistic colors",
    }
    
    text_descriptions = []
    for style in styles:
        prompt = STYLE_PROMPTS.get(style, f"a painting in the style of {style}")
        text_descriptions.append(prompt)
        print(f"  Style '{style}' -> Prompt: '{prompt}'")

    text_tokens = tokenizer(text_descriptions).to(device)
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    print(f"\nStarting classification for {len(styles)} styles")
    
    overall_correct = 0
    total_images = 0
    
    for true_style, image_paths in sampled_images.items():
        if true_style not in styles:
            continue
            
        print(f"Processing {true_style} ({len(image_paths)} images)")
        style_correct = 0
        
        for img_path in tqdm(image_paths, leave=False):
            if not os.path.exists(img_path):
                continue
                
            try:
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                
                with torch.no_grad(), torch.cuda.amp.autocast():
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    
                    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    
                pred_idx = text_probs.argmax(dim=-1).item()
                pred_style = styles[pred_idx]
                confidence = text_probs[0, pred_idx].item()
                
                is_correct = (pred_style == true_style)
                if is_correct:
                    style_correct += 1
                    overall_correct += 1
                
                total_images += 1
                
                results.append({
                    'image_path': img_path,
                    'true_style': true_style,
                    'predicted_style': pred_style,
                    'confidence': confidence,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"  Accuracy for {true_style}: {style_correct}/{len(image_paths)} ({style_correct/len(image_paths)*100:.1f}%)")

    accuracy = overall_correct / total_images if total_images > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.4f} ({overall_correct}/{total_images})")
    
    return results


def save_results(results: List[Dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'openclip_classification_results.json')
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {output_path}")
    
    df = pd.DataFrame(results)
    if not df.empty:
        report = classification_report(df['true_style'], df['predicted_style'], output_dict=True)
        report_path = os.path.join(output_dir, 'openclip_classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Classification report saved to {report_path}")
        
        labels = sorted(df['true_style'].unique())
        cm = confusion_matrix(df['true_style'], df['predicted_style'], labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_path = os.path.join(output_dir, 'openclip_confusion_matrix.csv')
        cm_df.to_csv(cm_path)
        print(f"Confusion matrix saved to {cm_path}")


def main():
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    METADATA_FILE = os.path.join(SCRIPT_DIR, 'outputs', 'metadata', 'sampled_images.json')
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs', 'openclip_results')
    
    print("OpenCLIP Style Classification")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    try:
        print(f"Loading metadata from {METADATA_FILE}")
        metadata = load_metadata(METADATA_FILE)
        raw_sampled_images = metadata.get('sampled_images', {})
        
        if not raw_sampled_images:
            print("No sampled images found in metadata")
            return

        TARGET_CLASSIFICATION_STYLES = ["Baroque", "Neoclassicism", "Post-Impressionism", "Impressionism", "Chinese Ink Painting"]
        
        style_key_to_classification_name_map = {
            "Baroque": "Baroque",
            "Neoclassicism": "Neoclassicism",
            "Post-Impressionism": "Post-Impressionism",
            "Impressionism": "Impressionism",
            "Chinese_painting": "Chinese Ink Painting",
            "Chinese Ink Painting": "Chinese Ink Painting",
            "Ink and wash painting": "Chinese Ink Painting"
        }

        filtered_sampled_images = {}
        for sampled_style_key, image_paths in raw_sampled_images.items():
            if sampled_style_key in style_key_to_classification_name_map:
                classification_name = style_key_to_classification_name_map[sampled_style_key]
                if classification_name in TARGET_CLASSIFICATION_STYLES:
                    existing_image_paths = [p for p in image_paths if os.path.exists(p)]
                    if existing_image_paths:
                        filtered_sampled_images[classification_name] = existing_image_paths
                    else:
                        print(f"Warning: No valid image paths found for style '{sampled_style_key}' mapped to '{classification_name}'")
            else:
                print(f"Skipping style '{sampled_style_key}' as it's not one of the 5 target styles for classification")

        styles = sorted([s for s in TARGET_CLASSIFICATION_STYLES if s in filtered_sampled_images])

        if not styles:
            print("No images found for the specified 5 styles after filtering")
            return
            
        print(f"Will classify into {len(styles)} styles: {styles}")
        
    except Exception as e:
        print(f"Failed to load or filter metadata: {e}")
        return

    model, preprocess, tokenizer = setup_openclip_model(
        model_name='ViT-SO400M-14-SigLIP-384', 
        pretrained='webli',
        device=device
    )

    results = classify_images(filtered_sampled_images, styles, model, preprocess, tokenizer, device)
    save_results(results, OUTPUT_DIR)
    
    print("Done")


if __name__ == "__main__":
    main()
