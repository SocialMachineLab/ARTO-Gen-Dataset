
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
import cv2


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


class SAMAnalyzer:
    """SAM 2.1 Analyzer for Visual Structure Parsability"""

    def __init__(self, model_config: Dict, device='auto'):
        """
        Args:
            model_config: SAM configuration dict
            device: Device to run on
        """
        self.config = model_config
        self.device = get_device(device)

        print(f"Loading SAM 2.1 from Hugging Face on {self.device}")

        try:
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Load SAM 2.1 directly from Hugging Face
            model_name = "facebook/sam2.1-hiera-large"
            print(f"Model: {model_name}")

            self.predictor = SAM2ImagePredictor.from_pretrained(model_name)

            # Move to device
            self.predictor.model.to(self.device)

            # Store parameters for automatic mask generation
            # Reduce points_per_side to save memory (16x16 = 256 points instead of 32x32 = 1024)
            self.points_per_side = model_config.get('points_per_side', 16)
            self.pred_iou_thresh = model_config.get('pred_iou_thresh', 0.88)
            self.stability_score_thresh = model_config.get('stability_score_thresh', 0.95)
            self.min_mask_region_area = model_config.get('min_mask_region_area', 100)

            print("SAM 2.1 loaded successfully from Hugging Face")

        except ImportError:
            print("Error: sam2 package not installed")
            print("Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
            raise

        except Exception as e:
            print(f"Error loading SAM 2.1: {e}")
            raise

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze an image using SAM 2.1 to generate masks

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with mask statistics
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            # Set image for predictor (SAM 2.1 requires this)
            self.predictor.set_image(image_np)

            # Generate masks automatically (Grid prompt simulation)
            # Create a grid of points
            h, w = image_np.shape[:2]
            n_points = self.points_per_side

            # Generate grid points
            xs = np.linspace(0, w, n_points + 2)[1:-1]
            ys = np.linspace(0, h, n_points + 2)[1:-1]
            grid_xs, grid_ys = np.meshgrid(xs, ys)
            points = np.stack([grid_xs.flatten(), grid_ys.flatten()], axis=1)

            # Prepare labels (1 = positive point)
            labels = np.ones(len(points))

            # Batch prediction allows processing many points
            # Depending on VRAM, we might need to batch this further, but 256 points is usually fine
            all_masks = []
            all_scores = []
            all_logits = []

            # Predict masks for grid points
            masks, scores, logits = self.predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True # We want the best mask per point
            )
            # masks shape: [N_points, 3, H, W] - 3 masks per point usually

            # Filter masks
            valid_masks = []
            mask_areas = []
            stability_scores = []

            # Process each point's prediction
            for i in range(len(points)):
                # Get best mask for this point (highest score)
                best_idx = np.argmax(scores[i])

                mask = masks[i][best_idx]
                score = scores[i][best_idx]

                # Check thresholds
                if score < self.pred_iou_thresh:
                    continue

                # Compute stability (simple proxy using score here, full stability calculation is complex)
                if score < self.stability_score_thresh:
                    continue

                # Compute area
                area = np.sum(mask)
                if area < self.min_mask_region_area:
                    continue

                valid_masks.append(mask)
                mask_areas.append(area)
                stability_scores.append(score)

            # Non-Maximum Suppression (NMS) equivalent to remove duplicate masks
            # Since we sampled dense grid, many points land on same object
            unique_masks = self._nms_masks(valid_masks, iou_thresh=0.7)

            return {
                'mask_count': len(unique_masks),
                'avg_mask_area': float(np.mean([np.sum(m) for m in unique_masks])) if unique_masks else 0,
                'avg_stability_score': float(np.mean(stability_scores)) if stability_scores else 0,
                'masks': unique_masks # Optional, might consume memory
            }

        except Exception as e:
            print(f"Error analyzing image with SAM: {e}")
            return {'mask_count': 0, 'avg_mask_area': 0, 'avg_stability_score': 0}

    def _nms_masks(self, masks, iou_thresh=0.7):
        """Simple NMS for binary masks"""
        if not masks:
            return []

        # Sort by size (area) - or score if available
        # Here we just iterate and keep if not covered
        kept_masks = []

        for mask in masks:
            is_duplicate = False
            mask_area = np.sum(mask)

            for kept in kept_masks:
                kept_area = np.sum(kept)
                intersection = np.sum(mask & kept)
                union = np.sum(mask | kept)
                iou = intersection / union if union > 0 else 0

                if iou > iou_thresh:
                    is_duplicate = True
                    break

            if not is_duplicate:
                kept_masks.append(mask)

        return kept_masks


class OWLEncoder:
    """OWL-ViT v2 Detector for Semantic Recognition"""

    def __init__(self, model_config: Dict, queries: Dict[str, List[str]], device='auto'):
        """
        Args:
            model_config: OWL configuration dict
            queries: Dictionary of queries (generic, basic, art-specific)
            device: Device to run on
        """
        self.config = model_config
        self.queries = queries
        self.device = get_device(device)

        print(f"Loading OWL-ViT v2 on {self.device}")

        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection

            model_name = model_config.get('model_name', 'google/owlv2-base-patch16-ensemble')
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()

            # Prepare text queries
            self.flattened_queries = (
                queries['generic'] +
                queries['basic'] +
                queries['art_specific']
            )

            # Pre-encode text queries (Optimization)
            # OWL-ViT text encoding is independent of image
            # However, HF API usually takes text+image together.
            # We will batch process text+image during inference.

            print("OWL-ViT v2 loaded successfully")

        except Exception as e:
            print(f"Error loading OWL-ViT: {e}")
            raise

    def detect(self, image_path: str, threshold: float = 0.1) -> Dict:
        """
        Detect objects in image using predefined queries

        Args:
            image_path: Path to image
            threshold: Confidence threshold

        Returns:
            Dictionary with detection statistics
        """
        try:
            image = Image.open(image_path).convert('RGB')
            W, H = image.size

            # Process inputs
            # Split queries into batches if too many (max length usually 77 tokens, usually fine for ~40 short queries)
            # HF Owlv2 handles list of text queries
            inputs = self.processor(
                text=self.flattened_queries,
                images=image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process
            # Target size (H, W)
            target_sizes = torch.tensor([[H, W]], device=self.device)
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]

            # Analyze results
            detections = []
            scores = results["scores"].tolist()
            labels = results["labels"].tolist()
            boxes = results["boxes"].tolist()

            # Count per category type
            query_counts = {
                'generic': 0,
                'basic': 0,
                'art_specific': 0
            }

            # Map label index back to query category
            n_generic = len(self.queries['generic'])
            n_basic = len(self.queries['basic'])

            for score, label, box in zip(scores, labels, boxes):
                query_idx = label
                query_text = self.flattened_queries[query_idx]

                # Determine category
                if query_idx < n_generic:
                    cat = 'generic'
                elif query_idx < n_generic + n_basic:
                    cat = 'basic'
                else:
                    cat = 'art_specific'

                query_counts[cat] += 1

                detections.append({
                    'label': query_text,
                    'score': score,
                    'box': box,
                    'category': cat
                })

            return {
                'total_detections': len(detections),
                'max_confidence': max(scores) if scores else 0.0,
                'avg_confidence': float(np.mean(scores)) if scores else 0.0,
                'query_counts': query_counts,
                'query_coverage': len(set(d['label'] for d in detections)) / len(self.flattened_queries)
            }

        except Exception as e:
            print(f"Error running OWL detection: {e}")
            return {'total_detections': 0, 'max_confidence': 0.0, 'query_counts': {}}


# ==================== Convenience Functions ====================

def create_sam_analyzer(config, device='auto'):
    return SAMAnalyzer(config['SAM_CONFIG'], device)

def create_owl_detector(config, device='auto'):
    return OWLEncoder(config['OWL_CONFIG'], config['OWL_QUERIES'], device)

def calculate_detectability_scores(sam_stats, owl_stats, weights):
    """
    Compute composite detectability score from SAM and OWL statistics

    Formula:
    Score = w1 * SAM_Stability + w2 * OWL_Confidence + w3 * OWL_Count_Normalized
    """
    sam_w = weights['final']['sam_weight']
    owl_w = weights['final']['owl_weight']

    # SAM Score (0-1)
    # Stability is already 0-1 (usually high, e.g., 0.95+)
    # Mask count: normalized (e.g., 3-20 is good)
    sam_score = sam_stats.get('avg_stability_score', 0)

    # OWL Score (0-1)
    # Confidence is 0-1
    owl_conf = owl_stats.get('max_confidence', 0)
    # Coverage: portion of queries that found something
    owl_cov = owl_stats.get('query_coverage', 0)

    owl_score = (owl_conf * 0.7) + (owl_cov * 0.3)

    return {
        'total_score': (sam_score * sam_w) + (owl_score * owl_w),
        'sam_component': sam_score,
        'owl_component': owl_score
    }
