"""Object detection validation: detect the specified objects and score the match."""

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False
    # Minimal stand-in so imports succeed without OpenCV
    class MockCV2:
        @staticmethod
        def imread(*args, **kwargs):
            return None
    cv2 = MockCV2()
import numpy as np
import json
import re
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

from .. import instance_assignment as ia
from ...conceptualization.object_instance import (
    PRIMARY, SECONDARY, colours_of, materials_of)
import torch
from PIL import Image

import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from rfdetr import RFDETRLarge  
    from rfdetr import RFDETRBase  
    from rfdetr.util.coco_classes import COCO_CLASSES
    RFDETR_AVAILABLE = True
except Exception as e:
    logger.warning(f"RF-DETR not available: {e}")
    RFDETR_AVAILABLE = False

# OWLv2 and GroundingDINO via transformers (preferred)
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    TRANSFORMERS_GROUNDING_AVAILABLE = True
    logger.info("Transformers GroundingDINO available")
except ImportError:
    TRANSFORMERS_GROUNDING_AVAILABLE = False
    logger.warning("Transformers GroundingDINO not available")

# Fallback: standalone groundingdino package
try:
    from groundingdino.util.inference import load_model, predict
    GROUNDINGDINO_AVAILABLE = True
    logger.info("GroundingDINO package imported successfully")
except ImportError as e:
    logger.warning(f"GroundingDINO not available: {e}")
    GROUNDINGDINO_AVAILABLE = False


class ObjectDetectionValidator:
    """Object detection with OWLv2 and GroundingDINO, falling back to RF-DETR."""

    def __init__(self, model_variant: str = "base", confidence_threshold: float = 0.25):
        """Load the detectors; `confidence_threshold` is the detection cutoff."""
        
        self.model_variant = model_variant.lower().strip()
        self.confidence_threshold = confidence_threshold
        self.grounding_dino_model = None
        self.rf_detr_model = None
        self.class_names = {}
        
        self.initialize_model()
    
    def initialize_model(self):
        """Load OWLv2 and GroundingDINO, else the standalone GroundingDINO, else RF-DETR."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models_loaded = []

        # Load OWLv2 (high recall)
        if TRANSFORMERS_GROUNDING_AVAILABLE:
            try:
                logger.info("Loading OWLv2 model (high recall)...")
                model_id = "google/owlv2-base-patch16-ensemble"
                self.owlv2_processor = AutoProcessor.from_pretrained(model_id)
                self.owlv2_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.owlv2_model = self.owlv2_model.to(device)
                self.owlv2_device = device
                models_loaded.append('OWLv2')
                logger.info(f"OWLv2 model loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load OWLv2 model: {e}")
                self.owlv2_processor = None
                self.owlv2_model = None

        # Load GroundingDINO (high precision)
        if TRANSFORMERS_GROUNDING_AVAILABLE:
            try:
                logger.info("Loading GroundingDINO model (high precision)...")
                model_id = "IDEA-Research/grounding-dino-base"
                self.grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
                self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.grounding_dino_model = self.grounding_dino_model.to(device)
                self.grounding_dino_device = device
                models_loaded.append('GroundingDINO')
                logger.info(f"GroundingDINO loaded via transformers on {device}")
            except Exception as e:
                logger.warning(f"Failed to load GroundingDINO via transformers: {e}")
                self.grounding_dino_processor = None
                self.grounding_dino_model = None

        # Check if at least one model is loaded
        if models_loaded:
            logger.info(f"Dual-model detection initialized with: {', '.join(models_loaded)}")
            return True
        
        # Fallback: standalone groundingdino package
        if GROUNDINGDINO_AVAILABLE:
            try:
                logger.info("Checking GroundingDINO installation...")

                import groundingdino
                groundingdino_path = os.path.dirname(groundingdino.__file__)
                logger.info(f"   GroundingDINO package path: {groundingdino_path}")
                
                # Config and weights shipped inside the package
                package_model_paths = [
                    (os.path.join(groundingdino_path, "config", "GroundingDINO_SwinT_OGC.py"),
                     os.path.join(groundingdino_path, "checkpoints", "groundingdino_swint_ogc.pth")),
                    (os.path.join(groundingdino_path, "GroundingDINO_SwinT_OGC.py"),
                     os.path.join(groundingdino_path, "groundingdino_swint_ogc.pth")),
                ]
                
                # Local and cache locations
                user_paths = [
                    ("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth"),
                    ("models/GroundingDINO_SwinT_OGC.py", "models/groundingdino_swint_ogc.pth"),
                    ("~/.cache/groundingdino/GroundingDINO_SwinT_OGC.py", "~/.cache/groundingdino/groundingdino_swint_ogc.pth"),
                ]
                
                all_paths = package_model_paths + user_paths
                
                model_loaded = False
                for config_path, weights_path in all_paths:
                    config_path = os.path.expanduser(config_path)
                    weights_path = os.path.expanduser(weights_path)
                    
                    logger.info(f"   Checking: {config_path}")
                    logger.info(f"   Checking: {weights_path}")
                    
                    if os.path.exists(config_path) and os.path.exists(weights_path):
                        logger.info(f"   Found model files")
                        try:
                            self.grounding_dino_model = load_model(config_path, weights_path)
                            logger.info(f"GroundingDINO model loaded from {config_path}")
                            model_loaded = True
                            break
                        except Exception as load_error:
                            logger.warning(f"   Failed to load model: {load_error}")
                            continue
                    else:
                        logger.info(f"   Files not found")
                
                if model_loaded:
                    return True
                else:
                    logger.warning("GroundingDINO model files not found in any expected location")
                    logger.info("Solutions:")
                    logger.info("   1. Download from: https://github.com/IDEA-Research/GroundingDINO/releases/")
                    logger.info("   2. Place files in project directory")
                    logger.info("   3. Or use: huggingface-cli download --repo-type model IDEA-Research/grounding-dino-base")
                    
                    # Try downloading the files, then load again
                    if self._try_auto_download():
                        for config_path, weights_path in [("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth")]:
                            if os.path.exists(config_path) and os.path.exists(weights_path):
                                try:
                                    self.grounding_dino_model = load_model(config_path, weights_path)
                                    logger.info(f"GroundingDINO model loaded after auto-download")
                                    return True
                                except Exception as e:
                                    logger.warning(f"Failed to load after download: {e}")
                    
                    logger.info("Falling back to RF-DETR...")
            except Exception as e:
                logger.warning(f"Failed to load GroundingDINO: {e}, falling back to RF-DETR")
        
        # Fallback to RF-DETR
        if RFDETR_AVAILABLE:
            try:
                self.rf_detr_model = RFDETRLarge()
                logger.info(f"RF-DETR model loaded as fallback")
                try:
                    self.rf_detr_model.optimize_for_inference()
                except Exception:
                    pass

                if COCO_CLASSES:
                    self.class_names = {i: name for i, name in enumerate(COCO_CLASSES)}
                else:
                    self.class_names = self._get_default_coco_names()
                
                logger.info(f"RF-DETR initialized ({self.model_variant}); classes={len(self.class_names)}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load RF-DETR model: {e}")
                return False
        else:
            logger.error("No detection model available")
            return False
    
    def _get_default_coco_names(self):
        """Return the 80 COCO class names by index."""
        return {
            0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
            6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
            11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
            16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
            22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag',
            27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard',
            32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
            36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
            46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
            51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
            56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
            61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
            67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
            72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
        }
    
    def _try_auto_download(self) -> bool:
        """Download the GroundingDINO config and weights into the working directory."""
        try:
            import urllib.request
            logger.info("Attempting to auto-download GroundingDINO model files...")

            config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            
            if not os.path.exists("GroundingDINO_SwinT_OGC.py"):
                logger.info("   Downloading config file...")
                urllib.request.urlretrieve(config_url, "GroundingDINO_SwinT_OGC.py")
                logger.info("   Config file downloaded")
            
            if not os.path.exists("groundingdino_swint_ogc.pth"):
                logger.info("   Downloading weights file (this may take a while)...")
                urllib.request.urlretrieve(weights_url, "groundingdino_swint_ogc.pth")
                logger.info("   Weights file downloaded")
            
            return True
            
        except Exception as e:
            logger.warning(f"   Auto-download failed: {e}")
            logger.info("   Please download manually and place in project directory")
            return False
    
    def extract_object_names(self, json_data: Dict[str, Any]) -> List[str]:
        """Build ordered detector queries: variety names first, then COCO categories.

        Querying the COCO category lets instance assignment fall back to it
        (`parent_of_gt`) when the variety itself is not found.
        """
        instances = ia.instances_from_spec(json_data)
        terms = ia.build_queries(instances)

        if not terms:
            # No instances: fall back to the flat name list.
            objects = json_data.get('objects') or {}
            for name in objects.get('object_names') or []:
                if str(name) not in terms:
                    terms.append(str(name))

        logger.info(f"Detection queries ({len(terms)}): {terms}")
        return terms
    
    def detect_objects_guided(self, image_path: str, expected_objects: List[str],
                              per_query: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Detect the queried objects with the best available model setup.

        `per_query` (query term -> number of instances it can stand for, from
        instance_assignment.query_counts) sets how many distinct boxes are kept
        per term; a term not listed keeps one.
        """
        has_owlv2 = hasattr(self, 'owlv2_processor') and self.owlv2_processor is not None
        has_gdino = hasattr(self, 'grounding_dino_processor') and self.grounding_dino_processor is not None

        # Dual-model fusion detection
        if has_owlv2 and has_gdino:
            logger.info("Using dual-model fusion (OWLv2 + GroundingDINO)")
            return self._dual_model_fusion_detection(image_path, expected_objects, per_query)

        # Single model detection (fallback)
        elif has_owlv2:
            logger.info("Using OWLv2 only")
            return self._owlv2_detection(image_path, expected_objects)

        elif has_gdino:
            logger.info("Using GroundingDINO only")
            return self._transformers_grounding_detection(image_path, expected_objects, per_query)

        elif hasattr(self, 'grounding_dino_model') and self.grounding_dino_model is not None:
            return self._grounding_dino_detection(image_path, expected_objects)

        elif self.rf_detr_model is not None:
            logger.info("Using RF-DETR fallback for detection")
            return self._rf_detr_detection(image_path)

        else:
            return {'error': 'No detection model available'}

    def _dual_model_fusion_detection(self, image_path: str, expected_objects: List[str],
                                     per_query: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Fuse OWLv2 (high recall) and GroundingDINO (high precision) detections per query.

        Each query keeps up to per_query[term] distinct objects (one by default),
        so two cats in the specification can be found as two boxes.
        """
        try:
            logger.info(f"Dual-model detection for {len(expected_objects)} objects: {expected_objects}")

            # 1. OWLv2 detection (high recall)
            logger.info("   -> Running OWLv2 (high recall)...")
            owlv2_result = self._owlv2_detection(image_path, expected_objects)

            # 2. GroundingDINO detection (high precision)
            logger.info("   -> Running GroundingDINO (high precision)...")
            gdino_result = self._transformers_grounding_detection(image_path, expected_objects, per_query)

            # A model that failed contributes no detections; if both failed
            # there is no detection result at all.
            model_errors = {name: result['error']
                            for name, result in (('OWLv2', owlv2_result), ('GroundingDINO', gdino_result))
                            if 'error' in result}
            if len(model_errors) == 2:
                return {'error': f"both detectors failed: {model_errors}"}
            for name, error in model_errors.items():
                logger.warning(f"   {name} failed ({error}); fusing the other model's detections only")

            # 3. Fuse detection results
            logger.info("   -> Fusing detection results...")

            owlv2_objects = owlv2_result.get('detected_objects', [])
            gdino_objects = gdino_result.get('detected_objects', [])

            # Up to `wanted` distinct detections for each expected object
            fused_objects = []
            fused_counts = {}
            detection_sources: Dict[str, List[str]] = {}

            for obj_name, wanted in self._query_rounds(expected_objects, per_query):
                owlv2_left = [obj for obj in owlv2_objects
                              if self._is_detection_match(obj['class_name'].lower(), obj_name.lower())]
                gdino_left = [obj for obj in gdino_objects
                              if self._is_detection_match(obj['class_name'].lower(), obj_name.lower())]

                for _ in range(wanted):
                    best_detection, source = self._fuse_best(obj_name, owlv2_left, gdino_left)
                    if best_detection is None:
                        if not fused_counts.get(obj_name):
                            logger.info(f"   '{obj_name}' not detected by either model")
                        break

                    # Recompute center and area from the final box
                    x1, y1, x2, y2 = best_detection['bbox']
                    best_detection['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
                    best_detection['area'] = (x2 - x1) * (y2 - y1)

                    fused_objects.append(best_detection)
                    fused_counts[obj_name] = fused_counts.get(obj_name, 0) + 1
                    detection_sources.setdefault(obj_name, []).append(source)

                    # Whatever shows the same object is used up.
                    owlv2_left = [o for o in owlv2_left
                                  if self._calculate_iou(o['bbox'], best_detection['bbox']) < 0.5]
                    gdino_left = [o for o in gdino_left
                                  if self._calculate_iou(o['bbox'], best_detection['bbox']) < 0.5]

            sources = [s for per_name in detection_sources.values() for s in per_name]
            avg_confidence = np.mean([obj['confidence'] for obj in fused_objects]) if fused_objects else 0.0

            both_count = sum(1 for s in sources if s == 'both')
            gdino_count = sum(1 for s in sources if 'gdino' in s)
            owlv2_count = sum(1 for s in sources if s == 'owlv2_only')

            logger.info(f"   Fusion summary: {len(fused_objects)} boxes for {len(expected_objects)} queries")
            logger.info(f"      - Both models: {both_count}")
            logger.info(f"      - GDINO only: {gdino_count}")
            logger.info(f"      - OWLv2 only: {owlv2_count}")

            return {
                'detected_objects': fused_objects,
                'object_counts': fused_counts,
                'total_detections': len(fused_objects),
                'unique_classes': len(fused_counts),
                'average_confidence': float(avg_confidence),
                'detection_method': 'Dual-OWLv2-GroundingDINO',
                'model_errors': model_errors,
                'detection_sources': detection_sources,
                'fusion_stats': {
                    'both_models': both_count,
                    'gdino_only': gdino_count,
                    'owlv2_only': owlv2_count,
                    'dual_confirmation_rate': both_count / len(sources) if sources else 0.0
                }
            }

        except Exception as e:
            logger.error(f"Dual-model fusion detection failed: {e}")
            return {'error': str(e)}

    @staticmethod
    def _query_rounds(expected_objects: List[str],
                      per_query: Optional[Dict[str, int]]) -> List[tuple]:
        """(query, how many distinct boxes to keep) for each distinct query."""
        rounds, seen = [], set()
        for name in expected_objects:
            key = str(name).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            rounds.append((name, max(1, int((per_query or {}).get(key, 1)))))
        return rounds

    def _fuse_best(self, obj_name: str, owlv2_matches: List[Dict[str, Any]],
                   gdino_matches: List[Dict[str, Any]]) -> tuple:
        """The best remaining detection of one query, and which model(s) found it.

        Both models: their best boxes are merged when they overlap (IoU > 0.3),
        otherwise GroundingDINO's is taken. One model: its best box. Returns
        (None, 'none') when neither has a detection left.
        """
        owlv2_best = max(owlv2_matches, key=lambda x: x['confidence']) if owlv2_matches else None
        gdino_best = max(gdino_matches, key=lambda x: x['confidence']) if gdino_matches else None

        if owlv2_best and gdino_best:
            # IoU > 0.3 is treated as the same object
            iou = self._calculate_iou(owlv2_best['bbox'], gdino_best['bbox'])
            if iou > 0.3:
                # Same object, fuse bbox
                fused = {
                    'class_name': obj_name,
                    'bbox': self._merge_bbox(owlv2_best['bbox'], gdino_best['bbox']),
                    'confidence': (owlv2_best['confidence'] + gdino_best['confidence']) / 2,
                    'center': None,
                    'area': None
                }
                logger.info(f"   '{obj_name}' detected by both (IoU={iou:.2f}, conf={fused['confidence']:.3f})")
                return fused, 'both'
            # Different location, use high-precision GDINO
            detection = dict(gdino_best, class_name=obj_name)
            logger.info(f"   '{obj_name}' using GDINO (different locations, IoU={iou:.2f})")
            return detection, 'gdino_preferred'

        if gdino_best:
            detection = dict(gdino_best, class_name=obj_name)
            logger.info(f"   '{obj_name}' detected by GDINO only (conf={detection['confidence']:.3f})")
            return detection, 'gdino_only'

        if owlv2_best:
            detection = dict(owlv2_best, class_name=obj_name)
            logger.info(f"   '{obj_name}' detected by OWLv2 only (conf={detection['confidence']:.3f})")
            return detection, 'owlv2_only'

        return None, 'none'

    def _owlv2_detection(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """Detect the queried objects with OWLv2, retrying at a lower threshold if few are found."""
        try:
            if not expected_objects:
                logger.warning("No expected objects provided for OWLv2 detection")
                return {
                    'detected_objects': [],
                    'object_counts': {},
                    'total_detections': 0,
                    'unique_classes': 0,
                    'average_confidence': 0.0,
                    'detection_method': 'OWLv2'
                }

            logger.info(f"OWLv2 detection with objects: {expected_objects}")

            image = Image.open(image_path).convert("RGB")

            # Upscale images whose short side is under 512 px
            if min(image.size) < 512:
                scale_factor = 512 / min(image.size)
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Upscaled image to {new_size} for better detection")

            # OWLv2 requires text as list of lists
            texts = [expected_objects]

            inputs = self.owlv2_processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(self.owlv2_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.owlv2_model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]])  # [height, width]
            results = self.owlv2_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]

            # Fewer boxes than half the queries: retry at half the threshold (min 0.1)
            if len(results["boxes"]) < len(expected_objects) * 0.5:
                logger.info(f"Low detection count ({len(results['boxes'])}), retrying with lower threshold...")
                lower_threshold = max(0.1, self.confidence_threshold * 0.5)
                results = self.owlv2_processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=lower_threshold
                )[0]
                logger.info(f"Retry with threshold {lower_threshold}: found {len(results['boxes'])} objects")

            raw_detections = []

            if len(results["boxes"]) > 0:
                for i in range(len(results["boxes"])):
                    box = results["boxes"][i].cpu().numpy()  # [x1, y1, x2, y2]
                    score = results["scores"][i].cpu().item()
                    label_idx = results["labels"][i].cpu().item()

                    class_name = expected_objects[label_idx] if label_idx < len(expected_objects) else f"unknown_{label_idx}"

                    x1, y1, x2, y2 = box
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    area = (x2 - x1) * (y2 - y1)

                    detected_obj = {
                        'class_name': class_name,
                        'confidence': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center[0]), float(center[1])],
                        'area': float(area)
                    }

                    raw_detections.append(detected_obj)

            # Apply NMS to remove duplicate detections
            if raw_detections:
                nms_detections = self._apply_nms(raw_detections, iou_threshold=0.5)
                logger.info(f"NMS: {len(raw_detections)} raw detections -> {len(nms_detections)} filtered detections")

                detected_objects = nms_detections

                object_counts = {}
                for obj in detected_objects:
                    class_name = obj['class_name']
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
            else:
                detected_objects = []
                object_counts = {}

            avg_confidence = np.mean([obj['confidence'] for obj in detected_objects]) if detected_objects else 0.0

            return {
                'detected_objects': detected_objects,
                'object_counts': object_counts,
                'total_detections': len(detected_objects),
                'unique_classes': len(object_counts),
                'average_confidence': avg_confidence,
                'detection_method': 'OWLv2'
            }

        except Exception as e:
            logger.error(f"OWLv2 detection failed: {e}")
            return {'error': str(e)}

    def _transformers_grounding_detection(self, image_path: str, expected_objects: List[str],
                                          per_query: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """Detect the queried objects with GroundingDINO via transformers."""
        try:
            if not expected_objects:
                text_prompt = "artwork objects . painting elements . artistic subjects"
            else:
                # Each object is queried by its own name, and a few also by synonyms
                enhanced_objects = []
                for obj in expected_objects:
                    obj_lower = obj.lower()
                    enhanced_objects.append(obj)
                    if 'microwave' in obj_lower:
                        enhanced_objects.extend(['microwave', 'oven', 'kitchen appliance'])
                    elif 'horse' in obj_lower:
                        enhanced_objects.extend(['horse', 'steed', 'equine'])
                    elif 'zebra' in obj_lower:
                        enhanced_objects.extend(['zebra', 'striped horse'])
                    elif 'knife' in obj_lower:
                        enhanced_objects.extend(['knife', 'blade', 'dagger'])

                unique_objects = list(dict.fromkeys(enhanced_objects))  # Order-preserving dedup
                text_prompt = " . ".join(unique_objects)

            logger.info(f"Transformers GroundingDINO detection with prompt: '{text_prompt}'")
            
            image = Image.open(image_path).convert("RGB")

            # Upscale images whose short side is under 512 px
            if min(image.size) < 512:
                scale_factor = 512 / min(image.size)
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Upscaled image to {new_size} for better detection")

            inputs = self.grounding_dino_processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.grounding_dino_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.grounding_dino_model(**inputs)

            results = self.grounding_dino_processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=self.confidence_threshold, target_sizes=[image.size[::-1]]
            )[0]

            # Fewer boxes than half the queries: retry at half the threshold (min 0.1)
            if len(results["boxes"]) < len(expected_objects) * 0.5:
                logger.info(f"Low detection count ({len(results['boxes'])}), retrying with lower threshold...")
                lower_threshold = max(0.1, self.confidence_threshold * 0.5)
                results = self.grounding_dino_processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], threshold=lower_threshold, target_sizes=[image.size[::-1]]
                )[0]
                logger.info(f"Retry with threshold {lower_threshold}: found {len(results['boxes'])} objects")
            
            raw_detections = []

            if len(results["boxes"]) > 0:
                for i in range(len(results["boxes"])):
                    box = results["boxes"][i].cpu().numpy()  # [x1, y1, x2, y2]
                    score = results["scores"][i].cpu().item()
                    label = results["labels"][i]

                    x1, y1, x2, y2 = box
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    area = (x2 - x1) * (y2 - y1)

                    class_name = label.strip().lower()

                    detected_obj = {
                        'class_name': class_name,
                        'confidence': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float(center[0]), float(center[1])],
                        'area': float(area)
                    }

                    raw_detections.append(detected_obj)

            # Apply NMS to remove duplicate detections
            if raw_detections:
                nms_detections = self._apply_nms(raw_detections, iou_threshold=0.5)
                logger.info(f"NMS: {len(raw_detections)} raw detections -> {len(nms_detections)} filtered detections")

                # Map synonym labels back to the queried names
                detected_objects = self._map_to_original_objects(nms_detections, expected_objects, per_query)
                logger.info(f"Object mapping: {len(nms_detections)} filtered -> {len(detected_objects)} final objects")

                object_counts = {}
                for obj in detected_objects:
                    class_name = obj['class_name']
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
            else:
                detected_objects = []
                object_counts = {}

            avg_confidence = np.mean([obj['confidence'] for obj in detected_objects]) if detected_objects else 0.0

            return {
                'detected_objects': detected_objects,
                'object_counts': object_counts,
                'total_detections': len(detected_objects),
                'unique_classes': len(object_counts),
                'average_confidence': avg_confidence,
                'detection_method': 'GroundingDINO-Transformers'
            }
            
        except Exception as e:
            logger.error(f"Transformers GroundingDINO detection failed: {e}")
            return {'error': str(e)}

    def _apply_nms(self, detections: List[Dict[str, Any]], iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Non-maximum suppression within each class or synonym group."""
        if not detections:
            return []

        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections_sorted:
            current = detections_sorted.pop(0)
            keep.append(current)

            remaining = []
            for det in detections_sorted:
                iou = self._calculate_iou(current['bbox'], det['bbox'])

                # Suppress only overlapping boxes of semantically similar classes
                if iou < iou_threshold or not self._are_semantic_similar(current['class_name'], det['class_name']):
                    remaining.append(det)
                else:
                    logger.debug(f"NMS removed: {det['class_name']} (IoU={iou:.3f} with {current['class_name']})")

            detections_sorted = remaining

        return keep

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Return the intersection-over-union of two [x1, y1, x2, y2] boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _are_semantic_similar(self, class1: str, class2: str) -> bool:
        """Return True for identical classes or classes in the same synonym group."""
        class1_lower = class1.lower()
        class2_lower = class2.lower()

        microwave_terms = {'microwave', 'microwave oven', 'oven'}
        if class1_lower in microwave_terms and class2_lower in microwave_terms:
            return True

        horse_terms = {'horse', 'zebra', 'steed', 'equine'}
        if class1_lower in horse_terms and class2_lower in horse_terms:
            return True

        knife_terms = {'knife', 'blade', 'dagger'}
        if class1_lower in knife_terms and class2_lower in knife_terms:
            return True

        if class1_lower == class2_lower:
            return True

        return False

    def _map_to_original_objects(self, detections: List[Dict[str, Any]],
                                original_objects: List[str],
                                per_query: Optional[Dict[str, int]] = None) -> List[Dict[str, Any]]:
        """Keep the most confident matching detections per query, renamed to the query.

        Up to per_query[term] distinct boxes are kept for a term (one by default).
        """
        mapped_results = []

        for original_obj, wanted in self._query_rounds(original_objects, per_query):
            original_lower = original_obj.lower()

            matching_detections = sorted(
                (d for d in detections
                 if self._is_detection_match(d['class_name'].lower(), original_lower)),
                key=lambda x: x['confidence'], reverse=True)

            kept: List[Dict[str, Any]] = []
            for detection in matching_detections:
                if len(kept) == wanted:
                    break
                if any(self._calculate_iou(detection['bbox'], k['bbox']) >= 0.5 for k in kept):
                    continue
                kept.append(detection)

            for detection in kept:
                mapped_detection = detection.copy()
                mapped_detection['class_name'] = original_obj
                mapped_detection['original_detection'] = detection['class_name']  # Label the detector returned

                mapped_results.append(mapped_detection)
                logger.debug(f"Mapped '{detection['class_name']}' -> '{original_obj}' (confidence: {detection['confidence']:.3f})")

        return mapped_results

    def _is_detection_match(self, detected_class: str, original_obj: str) -> bool:
        """Return True if a detected label matches a queried object name.

        An empty label matches nothing, since "" is a substring of every query.
        """
        detected_class = (detected_class or "").strip().lower()
        original_obj = (original_obj or "").strip().lower()
        if not detected_class or not original_obj:
            return False

        if detected_class == original_obj:
            return True

        # Synonym groups
        if original_obj == 'microwave':
            microwave_matches = {'microwave', 'microwave oven', 'oven'}
            return detected_class in microwave_matches

        elif original_obj == 'horse':
            horse_matches = {'horse', 'steed', 'equine'}
            return detected_class in horse_matches

        elif original_obj == 'zebra':
            zebra_matches = {'zebra', 'striped horse'}
            return detected_class in zebra_matches

        elif original_obj == 'knife':
            knife_matches = {'knife', 'blade', 'dagger'}
            return detected_class in knife_matches

        # Compound names: the shorter term must be a separate word or the end
        # of the longer one, so "car" does not match inside "scarf":
        #
        #   sailboat / boat   suffix          -> match
        #   cow herd / cow    separate word   -> match
        #   scarf    / car    neither         -> no match
        #   cart     / car    prefix only     -> no match
        for shorter, longer in ((detected_class, original_obj),
                                (original_obj, detected_class)):
            if len(shorter) < 3 or shorter == longer or shorter not in longer:
                continue
            if longer.endswith(shorter):
                return True
            if any(word == shorter for word in re.split(r"[^a-z0-9]+", longer)):
                return True

        return False

    def _merge_bbox(self, bbox1: List[float], bbox2: List[float], weights: List[float] = [0.5, 0.5]) -> List[float]:
        """Fuse two boxes by weighted average of their coordinates."""
        x1 = bbox1[0] * weights[0] + bbox2[0] * weights[1]
        y1 = bbox1[1] * weights[0] + bbox2[1] * weights[1]
        x2 = bbox1[2] * weights[0] + bbox2[2] * weights[1]
        y2 = bbox1[3] * weights[0] + bbox2[3] * weights[1]
        return [float(x1), float(y1), float(x2), float(y2)]

    def _grounding_dino_detection(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """Detect the queried objects with the standalone GroundingDINO package."""
        try:
            if not expected_objects:
                text_prompt = "objects"
            else:
                text_prompt = " . ".join(expected_objects)
            
            logger.info(f"GroundingDINO detection with prompt: '{text_prompt}'")
            
            image = Image.open(image_path).convert("RGB")

            boxes, logits, phrases = predict(
                model=self.grounding_dino_model,
                image=image,
                caption=text_prompt,
                box_threshold=self.confidence_threshold,
                text_threshold=0.25
            )
            
            detected_objects = []
            object_counts = {}

            for i, (box, confidence, phrase) in enumerate(zip(boxes, logits, phrases)):
                # Convert normalised coordinates to pixels
                h, w = image.size[1], image.size[0]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h

                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                area = (x2 - x1) * (y2 - y1)

                class_name = phrase.strip().lower()
                
                detected_obj = {
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center[0]), float(center[1])],
                    'area': float(area)
                }
                
                detected_objects.append(detected_obj)

                if class_name in object_counts:
                    object_counts[class_name] += 1
                else:
                    object_counts[class_name] = 1
            
            return {
                'detected_objects': detected_objects,
                'object_counts': object_counts,
                'total_detections': len(detected_objects),
                'unique_classes': len(object_counts),
                'average_confidence': float(np.mean(logits)) if len(logits) > 0 else 0.0,
                'detection_method': 'GroundingDINO'
            }
            
        except Exception as e:
            logger.error(f"GroundingDINO detection failed: {e}")
            return {'error': str(e)}
        
    def _rf_detr_detection(self, image_path: str) -> Dict[str, Any]:
        """Detect all COCO classes with RF-DETR (used when no open-vocabulary model loads)."""
        if self.rf_detr_model is None:
            return {'error': 'RF-DETR model not loaded'}

        try:
            image = Image.open(image_path).convert("RGB")

            detections = self.rf_detr_model.predict(image, threshold=self.confidence_threshold)

            # Detections carry xyxy (N x 4), class_id (N,) and confidence (N,)
            if detections is None or getattr(detections, "xyxy", None) is None:
                return {
                    'detected_objects': [],
                    'object_counts': {},
                    'total_detections': 0,
                    'unique_classes': 0,
                    'average_confidence': 0.0,
                    'detection_method': 'RF-DETR'
                }

            xyxy = np.asarray(detections.xyxy)            # (N,4)
            class_ids = np.asarray(detections.class_id)    # (N,)
            confidences = np.asarray(detections.confidence)# (N,)
            
            class_names_from_detection = getattr(detections, 'class_name', None)
            if class_names_from_detection is not None:
                logger.info(f"RF-DETR also returned class_names: {class_names_from_detection}")
            
            logger.info(f"Raw class_ids from RF-DETR: {class_ids}")
            logger.info(f"Available class mappings: {len(self.class_names)} classes")

            H, W = image.size[1], image.size[0]  # PIL: size=(W,H)
            image_area = W * H

            detected_objects = []
            object_counts = defaultdict(int)

            for box, cid, conf in zip(xyxy, class_ids, confidences):
                x1, y1, x2, y2 = [float(v) for v in box]
                class_id = int(cid)
                
                logger.info(f"Processing detection: class_id={class_id}")
                
                # Name classes from the local COCO table, not the detector's labels
                default_names = self._get_default_coco_names()
                class_name = default_names.get(class_id, f"unknown_class_{class_id}")

                class_name = str(class_name)

                logger.info(f"Mapped: class_id={class_id} -> class_name='{class_name}'")

                area = max(0.0, (x2 - x1) * (y2 - y1))
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                detected_objects.append({
                    'class_name': class_name,
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2],
                    'center': [cx, cy],
                    'area': area
                })
                object_counts[class_name] += 1

            avg_conf = float(confidences.mean()) if len(confidences) else 0.0

            return {
                'detected_objects': sorted(detected_objects, key=lambda d: d['confidence'], reverse=True),
                'object_counts': dict(object_counts),
                'total_detections': len(detected_objects),
                'unique_classes': len(object_counts),
                'average_confidence': avg_conf,
                'detection_method': 'RF-DETR'
            }

        except Exception as e:
            logger.error(f"RF-DETR detection failed: {e}")
            return {'error': f'Detection failed: {str(e)}'}


    def extract_expected_objects(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Collect expected object names, ids, details and roles from the specification."""
        expected_info = {
            'object_names': [],
            'object_details': [],
            'primary_objects': [],
            'secondary_objects': [],
            'object_ids': [],
            'object_metadata': {}
        }
        
        try:
            objects_data = json_data.get('objects', {})

            object_names = objects_data.get('object_names', [])
            if isinstance(object_names, list):
                expected_info['object_names'] = object_names
            
            object_ids = objects_data.get('object_ids', [])
            if isinstance(object_ids, list):
                expected_info['object_ids'] = object_ids

            # enhanced_objects is the main source; it also fills missing names and ids
            enhanced_objects = objects_data.get('enhanced_objects', [])
            if isinstance(enhanced_objects, list):
                if not expected_info['object_names']:
                    expected_info['object_names'] = [obj.get('name', '') for obj in enhanced_objects if obj.get('name')]

                if not expected_info['object_ids']:
                    expected_info['object_ids'] = [obj.get('object_id', -1) for obj in enhanced_objects if obj.get('object_id') is not None]
                
                for obj in enhanced_objects:
                    if isinstance(obj, dict):
                        obj_name = obj.get('name', '')
                        obj_id = obj.get('object_id', -1)
                        
                        if obj_name:
                            obj_info = {
                                'name': obj_name,
                                'object_id': obj_id,
                                'instance_id': obj.get('instance_id'),
                                'role': obj.get('role', SECONDARY),
                                'size': obj.get('size', 'medium'),
                                'state': obj.get('state', ''),
                                'main_colours': colours_of(obj),
                                'materials': materials_of(obj),
                                'artistic_description': obj.get('artistic_description', '')
                            }
                            
                            expected_info['object_details'].append(obj_info)
                            
                            # Per-name object metadata
                            expected_info['object_metadata'][obj_name] = {
                                'id': obj_id,
                                'size': obj_info['size'],
                                'colors': obj_info['main_colours'],
                                'materials': obj_info['materials'],
                                'state': obj_info['state']
                            }
                            
                            if obj_info['role'] == PRIMARY:
                                expected_info['primary_objects'].append(obj_name)
                            else:
                                expected_info['secondary_objects'].append(obj_name)
            
            # Without enhanced_objects, build minimal details from object_names
            if not expected_info['object_details'] and expected_info['object_names']:
                for i, obj_name in enumerate(expected_info['object_names']):
                    obj_id = expected_info['object_ids'][i] if i < len(expected_info['object_ids']) else -1
                    expected_info['object_details'].append({
                        'name': obj_name,
                        'object_id': obj_id,
                        'role': SECONDARY
                    })
                    expected_info['secondary_objects'].append(obj_name)

        except Exception as e:
            logger.error(f"Error extracting expected objects: {e}")
        
        return expected_info
    
    def calculate_object_matching_score(self, detected_counts: Dict[str, int], 
                                      expected_names: List[str]) -> Dict[str, Any]:
        """Match detected classes to expected names and score precision and recall.

        Matching tries exact names, synonym groups, known COCO confusions, then
        substring containment.
        """
        if not expected_names:
            return {
                'precision': 1.0 if not detected_counts else 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'accuracy_score': 50.0,
                'matched_objects': [],
                'missing_objects': [],
                'extra_objects': list(detected_counts.keys())
            }
        
        def normalize_name(name):
            return str(name).lower().strip()

        # Synonym groups for objects common in artworks
        semantic_mapping = {
            # Kitchen appliances
            'microwave': ['microwave', 'oven', 'toaster', 'hair drier', 'appliance', 'toothbrush', 'kitchen appliance'],
            'oven': ['oven', 'microwave', 'toaster', 'hair drier', 'appliance'],
            'refrigerator': ['refrigerator', 'fridge'],

            # People
            'person': ['person', 'people', 'human', 'man', 'woman', 'figure', 'character'],

            # Animals
            'zebra': ['zebra', 'striped horse'],
            'horse': ['horse', 'cow', 'steed', 'equine'],
            'cow': ['cow', 'horse', 'cattle'],

            # Tools and utensils
            'knife': ['knife', 'fork', 'spoon', 'blade', 'dagger', 'cutlery'],
            'fork': ['fork', 'knife', 'spoon', 'cutlery'],
            'spoon': ['spoon', 'knife', 'fork', 'cutlery'],

            # Furniture
            'chair': ['chair', 'seat'],
            'table': ['table', 'desk', 'dining table'],

            # Vehicles
            'car': ['car', 'automobile', 'vehicle'],

            # COCO classes commonly confused with expected objects
            'hair drier': ['hair drier', 'microwave', 'appliance'],
            'toothbrush': ['toothbrush', 'microwave', 'knife'],

            # Artwork-specific objects
            'angel': ['person', 'human', 'figure', 'winged figure'],
            'deity': ['person', 'human', 'figure', 'divine figure'],
            'warrior': ['person', 'human', 'figure', 'armored figure'],
        }
        
        # COCO class ids accepted as a detection of an expected object
        coco_id_to_expected = {
            78: 'microwave',   # hair drier
            79: 'microwave',   # toothbrush
            22: 'zebra',       # zebra
            17: 'horse',       # horse
            19: 'horse',       # cow
            43: 'knife',       # knife
            44: 'knife',       # spoon
            42: 'knife',       # fork
        }

        def find_semantic_match(expected, detected_list):
            """Return the detected label that matches `expected`, or None."""
            expected_lower = expected.lower()

            if expected_lower in [d.lower() for d in detected_list]:
                return expected_lower

            # A synonym of the expected object was detected
            if expected_lower in semantic_mapping:
                for synonym in semantic_mapping[expected_lower]:
                    if synonym in [d.lower() for d in detected_list]:
                        logger.info(f"Semantic match: '{expected}' matched with detected '{synonym}'")
                        return synonym
            
            # The expected object is a synonym of a detected label
            for detected in detected_list:
                detected_lower = detected.lower()
                if detected_lower in semantic_mapping:
                    if expected_lower in semantic_mapping[detected_lower]:
                        logger.info(f"Reverse semantic match: '{expected}' matched with detected '{detected}'")
                        return detected_lower
            
            # A detected COCO class known to stand in for the expected object
            for detected in detected_list:
                detected_lower = detected.lower()

                for class_id, expected_obj in coco_id_to_expected.items():
                    default_names = {
                        78: 'hair drier', 79: 'toothbrush', 22: 'zebra', 
                        17: 'horse', 19: 'cow',
                        43: 'knife', 44: 'spoon', 42: 'fork'
                    }
                    coco_class_name = default_names.get(class_id, '').lower()
                    
                    if detected_lower == coco_class_name and expected_lower == expected_obj.lower():
                        logger.info(f"COCO class_id match: '{expected}' matched with detected '{detected}' (class_id {class_id})")
                        return detected
            
            return None
        
        expected_normalized = [normalize_name(name) for name in expected_names]
        detected_normalized = {normalize_name(k): v for k, v in detected_counts.items()}
        detected_names_list = list(detected_normalized.keys())
        
        matched_objects = []
        missing_objects = []
        
        for expected in expected_names:
            expected_norm = normalize_name(expected)
            
            # 1. Direct match
            if expected_norm in detected_normalized:
                matched_objects.append(expected)
                continue
            
            # 2. Semantic match
            semantic_match = find_semantic_match(expected, detected_names_list)
            if semantic_match:
                matched_objects.append(expected)
                logger.info(f"Semantic match found: '{expected}' matched with '{semantic_match}'")
                continue
            
            # 3. Substring match
            found = False
            for detected_norm in detected_normalized:
                if expected_norm in detected_norm or detected_norm in expected_norm:
                    matched_objects.append(expected)
                    logger.info(f"Fuzzy match found: '{expected}' matched with '{detected_norm}'")
                    found = True
                    break
            
            if not found:
                missing_objects.append(expected)
        
        # Detections that match no expected object
        matched_normalized = [normalize_name(obj) for obj in matched_objects]
        extra_objects = []
        for detected_norm, count in detected_normalized.items():
            if detected_norm not in expected_normalized and detected_norm not in matched_normalized:
                found = False
                for exp_norm in expected_normalized:
                    if exp_norm in detected_norm or detected_norm in exp_norm:
                        found = True
                        break
                if not found:
                    extra_objects.append(detected_norm)
        
        num_matched = len(matched_objects)
        num_expected = len(expected_names)
        num_detected = sum(detected_counts.values())
        
        precision = num_matched / num_detected if num_detected > 0 else 0.0
        recall = num_matched / num_expected if num_expected > 0 else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy score (0-100)
        if num_expected == 0:
            accuracy_score = 100.0 if num_detected == 0 else 50.0
        else:
            accuracy_score = (recall * 70 + precision * 30)  # Recall weighted higher
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy_score': accuracy_score,
            'matched_objects': matched_objects,
            'missing_objects': missing_objects,
            'extra_objects': extra_objects,
            'num_matched': num_matched,
            'num_expected': num_expected,
            'num_detected': num_detected
        }
    
    def analyze_object_distribution(self, detected_objects: List[Dict], 
                                  image_path: str) -> Dict[str, Any]:
        """Summarise coverage, spatial balance and density of the detections."""
        if not detected_objects:
            return {
                'coverage_ratio': 0.0,
                'balance_score': 0.0,
                'spatial_distribution': 'empty',
                'object_density': 0.0
            }

        try:
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image for distribution analysis'}

            height, width = image.shape[:2]
            image_area = width * height

            total_bbox_area = sum(obj['area'] for obj in detected_objects)
            coverage_ratio = total_bbox_area / image_area

            centers = [obj['center'] for obj in detected_objects]

            centroid_x = np.mean([c[0] for c in centers])
            centroid_y = np.mean([c[1] for c in centers])
            
            # Balance from the spread of distances to the centroid
            distances = [np.sqrt((c[0] - centroid_x)**2 + (c[1] - centroid_y)**2) for c in centers]
            balance_score = 100.0 - min(np.std(distances) / max(width, height) * 100, 100.0)
            
            object_density = len(detected_objects) / (image_area / 1000000)  # Objects per million pixels
            
            return {
                'coverage_ratio': coverage_ratio,
                'balance_score': balance_score,
                'spatial_distribution': 'balanced' if balance_score > 70 else 'unbalanced',
                'object_density': object_density,
                'centroid': [centroid_x, centroid_y],
                'image_dimensions': [width, height]
            }
            
        except Exception as e:
            logger.error(f"Distribution analysis failed: {e}")
            return {'error': f'Distribution analysis failed: {str(e)}'}
    
    def create_detection_visualization(self, detection_result: Dict[str, Any], 
                                     image_path: str, output_path: str):
        """Draw detected boxes and labels on the image and save it."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            ax.imshow(image_rgb)
            
            if 'detected_objects' in detection_result:
                for obj in detection_result['detected_objects']:
                    bbox = obj['bbox']  # [x1, y1, x2, y2]
                    class_name = obj['class_name']
                    confidence = obj['confidence']
                    
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), 
                        bbox[2] - bbox[0], 
                        bbox[3] - bbox[1],
                        linewidth=2, 
                        edgecolor='red', 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    ax.text(bbox[0], bbox[1] - 5, 
                           f'{class_name}: {confidence:.2f}',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=10, color='black')
            
            ax.set_title(f'{detection_result.get("detection_method", "Object")} Detection Results\n'
                        f'Total: {detection_result.get("total_detections", 0)} objects')
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Visualization creation failed: {e}")
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str, 
                       bboxes_dict: Optional[Dict[str, List[int]]] = None) -> Dict[str, Any]:
        """Run the full object detection evaluation."""
        return self.comprehensive_object_detection_evaluation(json_data, image_path)
    
    def comprehensive_object_detection_evaluation(self, json_data: Dict[str, Any], 
                                                 image_path: str) -> Dict[str, Any]:
        """Detect the expected objects and score them.

        The overall score is 70% matching accuracy and 30% mean detection confidence.
        """
        print(f"Starting comprehensive object detection evaluation for: {os.path.basename(image_path)}")

        try:
            # 1. Extract expected object information
            print("   Step 1: Extracting expected objects from JSON...")
            expected_info = self.extract_expected_objects(json_data)
            expected_object_names = self.extract_object_names(json_data)

            total_expected = len(expected_info['object_names'])
            print(f"   Found {total_expected} expected objects: {expected_object_names}")

            # 2. Perform guided object detection
            print("   Step 2: Performing guided object detection...")
            detection_result = self.detect_objects_guided(
                image_path, expected_object_names,
                ia.query_counts(ia.instances_from_spec(json_data)))

            if 'error' in detection_result:
                return {'error': f"Object detection failed: {detection_result['error']}"}

            print(f"   Detected {detection_result['total_detections']} objects using {detection_result['detection_method']}")
            print(f"     - Detected objects: {list(detection_result['object_counts'].keys())}")

            # 3. Calculate object matching scores
            print("   Step 3: Calculating object matching scores...")
            matching_result = self.calculate_object_matching_score(
                detection_result['object_counts'],
                expected_info['object_names']
            )

            # 4. Calculate quality metrics
            print("   Step 4: Calculating core quality metrics...")

            confidence_score = detection_result['average_confidence'] * 100

            # 5. Weighted score
            weights = {
                'accuracy': 0.7,
                'confidence': 0.3,
            }

            overall_score = (
                matching_result['accuracy_score'] * weights['accuracy'] +
                confidence_score * weights['confidence']
            )

            print(f"   Object detection evaluation completed. Overall score: {overall_score:.1f}/100")

            # 6. Build the result
            evaluation_result = {
                'overall_score': overall_score,
                'dimension_scores': {
                    'detection_accuracy': matching_result['accuracy_score'],
                    'detection_confidence': confidence_score
                },
                'detailed_results': {
                    'detection': detection_result,
                    'expected_info': expected_info,
                    'matching_result': matching_result
                },
                'evaluation_summary': {
                    'total_detected': detection_result['total_detections'],
                    'total_expected': total_expected,
                    'detection_method': detection_result.get('detection_method', 'GroundingDINO'),
                    'detection_precision': matching_result['precision'],
                    'detection_recall': matching_result['recall'],
                    'f1_score': matching_result['f1_score'],
                    'evaluation_weights': weights
                }
            }
            
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Object detection evaluation failed: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}

            
                        
            
            
            