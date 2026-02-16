import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np
from typing import List, Union, Dict
from tqdm import tqdm


def get_device(device='auto'):
    """Auto-detect the best available device"""
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device)


class FeatureExtractor:
    """Base class for feature extraction"""

    def __init__(self, model_name: str, model_config: Dict, device='auto'):
        """
        Args:
            model_name: Name identifier of the model (e.g., 'dinov3_vit')
            model_config: Dictionary with 'name', 'feature_dim', 'input_size'
            device: Device to run the model on
        """
        self.model_name = model_name
        self.model_config = model_config
        self.device = get_device(device)

        print(f"Loading {model_name} on {self.device}")
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()

        # Load image processor
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_config['name'])
        except:
            # Fallback to manual preprocessing
            self.processor = None
            self.transform = self._get_default_transform(model_config['input_size'])

        print(f"Loaded {model_name}")

    def _load_model(self):
        """Load the pretrained model"""
        model = AutoModel.from_pretrained(self.model_config['name'])

        # For SigLIP, extract only the vision encoder
        if 'siglip' in self.model_name.lower():
            if hasattr(model, 'vision_model'):
                print(f"Using vision encoder only for {self.model_name}")
                return model.vision_model

        return model

    def _get_default_transform(self, input_size):
        """Default image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a single PIL image"""
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs['pixel_values']
        else:
            return self.transform(image).unsqueeze(0)

    def extract_features(self, images: Union[List[Image.Image], Image.Image]) -> np.ndarray:
        """
        Extract features from one or multiple images

        Args:
            images: Single PIL Image or list of PIL Images

        Returns:
            numpy array of shape (N, feature_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]

        features = []
        with torch.no_grad():
            for img in images:
                # Preprocess
                if self.processor is not None:
                    # Use processor (returns dict with 'pixel_values')
                    inputs = self.processor(images=img, return_tensors="pt")

                    # For vision-only models (like SigLIP vision_model), only pass pixel_values
                    if 'siglip' in self.model_name.lower():
                        pixel_values = inputs['pixel_values'].to(self.device)
                        outputs = self.model(pixel_values=pixel_values)
                    else:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs)
                else:
                    # Use manual transform
                    img_tensor = self.transform(img).unsqueeze(0).to(self.device)
                    outputs = self.model(img_tensor)

                # Get global feature vector
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    feat = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use CLS token or average pooling
                    feat = outputs.last_hidden_state.mean(dim=1)
                else:
                    # Fallback: use the first output
                    feat = outputs[0].mean(dim=1) if len(outputs[0].shape) > 2 else outputs[0]

                features.append(feat.cpu().numpy())

        return np.vstack(features)

    def extract_features_batch(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extract features from a list of image paths in batches

        Args:
            image_paths: List of paths to images
            batch_size: Number of images to process at once

        Returns:
            numpy array of shape (N, feature_dim)
        """
        all_features = []

        for i in tqdm(range(0, len(image_paths), batch_size),
                     desc=f"Extracting features ({self.model_name})"):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load {path}: {e}")
                    continue

            if batch_images:
                batch_features = self.extract_features(batch_images)
                all_features.append(batch_features)

        return np.vstack(all_features) if all_features else np.array([])


class MultiModelFeatureExtractor:
    """Extract features using multiple models simultaneously"""

    def __init__(self, models_config: Dict, device='auto'):
        """
        Args:
            models_config: Dictionary of model configurations from config.py
            device: Device to run models on
        """
        self.extractors = {}

        for model_name, model_config in models_config.items():
            self.extractors[model_name] = FeatureExtractor(
                model_name, model_config, device
            )

    def extract_all_features(self, image_paths: List[str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        """
        Extract features using all models

        Args:
            image_paths: List of paths to images
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping model_name -> feature array of shape (N, feature_dim)
        """
        features = {}

        for model_name, extractor in self.extractors.items():
            print(f"Extracting features with {model_name}")
            features[model_name] = extractor.extract_features_batch(
                image_paths, batch_size
            )
            print(f"{model_name}: Extracted {features[model_name].shape[0]} features "
                  f"of dimension {features[model_name].shape[1]}")

        return features


# Convenience function
def create_feature_extractor(model_name: str, model_config: Dict, device='auto') -> FeatureExtractor:
    """Create a feature extractor for a specific model"""
    return FeatureExtractor(model_name, model_config, device)
