
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError as e:
    print(f"OpenCV not available: {e}")
    CV2_AVAILABLE = False
    # Create a mock cv2 object
    class MockCV2:
        @staticmethod
        def imread(*args, **kwargs):
            return None
    cv2 = MockCV2()
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
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

# Try using transformers library GroundingDINO (simpler)
try:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    TRANSFORMERS_GROUNDING_AVAILABLE = True
    logger.info("✅ Transformers GroundingDINO available")
except ImportError:
    TRANSFORMERS_GROUNDING_AVAILABLE = False
    logger.warning("Transformers GroundingDINO not available")

# Fallback: original groundingdino package
try:
    from groundingdino.util.inference import load_model, predict
    GROUNDINGDINO_AVAILABLE = True
    logger.info("✅ GroundingDINO package imported successfully")
except ImportError as e:
    logger.warning(f"GroundingDINO not available: {e}")
    GROUNDINGDINO_AVAILABLE = False


class ObjectDetectionValidator:
    """Object Detection Validator based on GroundingDINO"""

    def __init__(self, model_variant: str = "base", confidence_threshold: float = 0.25):
        """
        Initialize Object Detection Validator

        Args:
            model_variant: Model variant
            confidence_threshold: Detection confidence threshold
        """
        
        self.model_variant = model_variant.lower().strip()
        self.confidence_threshold = confidence_threshold
        self.grounding_dino_model = None
        self.rf_detr_model = None
        self.class_names = {}
        
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize detection model - Load both OWLv2 and GroundingDINO"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        models_loaded = []

        # Load OWLv2 (high recall)
        if TRANSFORMERS_GROUNDING_AVAILABLE:
            try:
                logger.info("🔄 Loading OWLv2 model (high recall)...")
                model_id = "google/owlv2-base-patch16-ensemble"
                self.owlv2_processor = AutoProcessor.from_pretrained(model_id)
                self.owlv2_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.owlv2_model = self.owlv2_model.to(device)
                self.owlv2_device = device
                models_loaded.append('OWLv2')
                logger.info(f"✅ OWLv2 model loaded on {device}")
            except Exception as e:
                logger.warning(f"Failed to load OWLv2 model: {e}")
                self.owlv2_processor = None
                self.owlv2_model = None

        # Load GroundingDINO (high precision)
        if TRANSFORMERS_GROUNDING_AVAILABLE:
            try:
                logger.info("🔄 Loading GroundingDINO model (high precision)...")
                model_id = "IDEA-Research/grounding-dino-base"
                self.grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
                self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
                self.grounding_dino_model = self.grounding_dino_model.to(device)
                self.grounding_dino_device = device
                models_loaded.append('GroundingDINO')
                logger.info(f"✅ GroundingDINO loaded via transformers on {device}")
            except Exception as e:
                logger.warning(f"Failed to load GroundingDINO via transformers: {e}")
                self.grounding_dino_processor = None
                self.grounding_dino_model = None

        # Check if at least one model is loaded
        if models_loaded:
            logger.info(f"✅ Dual-model detection initialized with: {', '.join(models_loaded)}")
            return True
        
        # Fallback: Use original groundingdino package
        if GROUNDINGDINO_AVAILABLE:
            try:
                # First check if pip installed groundingdino contains pretrained model
                logger.info("🔍 Checking GroundingDINO installation...")
                
                # Try importing and checking groundingdino package structure
                import groundingdino
                groundingdino_path = os.path.dirname(groundingdino.__file__)
                logger.info(f"   GroundingDINO package path: {groundingdino_path}")
                
                # Check for pretrained model in package
                package_model_paths = [
                    (os.path.join(groundingdino_path, "config", "GroundingDINO_SwinT_OGC.py"),
                     os.path.join(groundingdino_path, "checkpoints", "groundingdino_swint_ogc.pth")),
                    (os.path.join(groundingdino_path, "GroundingDINO_SwinT_OGC.py"),
                     os.path.join(groundingdino_path, "groundingdino_swint_ogc.pth")),
                ]
                
                # User custom paths
                user_paths = [
                    # Relative paths (in project directory)
                    ("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth"),
                    ("models/GroundingDINO_SwinT_OGC.py", "models/groundingdino_swint_ogc.pth"),
                    # Absolute paths (common installation locations)
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
                        logger.info(f"   ✅ Found model files!")
                        try:
                            self.grounding_dino_model = load_model(config_path, weights_path)
                            logger.info(f"✅ GroundingDINO model loaded from {config_path}")
                            model_loaded = True
                            break
                        except Exception as load_error:
                            logger.warning(f"   ❌ Failed to load model: {load_error}")
                            continue
                    else:
                        logger.info(f"   ❌ Files not found")
                
                if model_loaded:
                    return True
                else:
                    logger.warning("GroundingDINO model files not found in any expected location")
                    logger.info("💡 Solutions:")
                    logger.info("   1. Download from: https://github.com/IDEA-Research/GroundingDINO/releases/")
                    logger.info("   2. Place files in project directory")
                    logger.info("   3. Or use: huggingface-cli download --repo-type model IDEA-Research/grounding-dino-base")
                    
                    # Try auto-download (if network available)
                    if self._try_auto_download():
                        # Retry loading
                        for config_path, weights_path in [("GroundingDINO_SwinT_OGC.py", "groundingdino_swint_ogc.pth")]:
                            if os.path.exists(config_path) and os.path.exists(weights_path):
                                try:
                                    self.grounding_dino_model = load_model(config_path, weights_path)
                                    logger.info(f"✅ GroundingDINO model loaded after auto-download")
                                    return True
                                except Exception as e:
                                    logger.warning(f"Failed to load after download: {e}")
                    
                    logger.info("🔄 Falling back to RF-DETR...")
            except Exception as e:
                logger.warning(f"Failed to load GroundingDINO: {e}, falling back to RF-DETR")
        
        # Fallback to RF-DETR
        if RFDETR_AVAILABLE:
            try:
                self.rf_detr_model = RFDETRLarge()
                logger.info(f"✅ RF-DETR model loaded as fallback")
                try:
                    self.rf_detr_model.optimize_for_inference()
                except Exception:
                    pass

                # Load COCO class names
                if COCO_CLASSES:
                    self.class_names = {i: name for i, name in enumerate(COCO_CLASSES)}
                else:
                    self.class_names = self._get_default_coco_names()
                
                logger.info(f"✅ RF-DETR initialized ({self.model_variant}); classes={len(self.class_names)}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load RF-DETR model: {e}")
                return False
        else:
            logger.error("No detection model available")
            return False
    
    def _get_default_coco_names(self):
        """Get default COCO dataset class names"""
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
        """Try auto-download GroundingDINO model files"""
        try:
            import urllib.request
            logger.info("🔄 Attempting to auto-download GroundingDINO model files...")
            
            # Model file URLs
            config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
            weights_url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            
            # Download config file
            if not os.path.exists("GroundingDINO_SwinT_OGC.py"):
                logger.info("   📥 Downloading config file...")
                urllib.request.urlretrieve(config_url, "GroundingDINO_SwinT_OGC.py")
                logger.info("   ✅ Config file downloaded")
            
            # Download weights file (large, may take time)
            if not os.path.exists("groundingdino_swint_ogc.pth"):
                logger.info("   📥 Downloading weights file (this may take a while)...")
                urllib.request.urlretrieve(weights_url, "groundingdino_swint_ogc.pth")
                logger.info("   ✅ Weights file downloaded")
            
            return True
            
        except Exception as e:
            logger.warning(f"   ❌ Auto-download failed: {e}")
            logger.info("   Please download manually and place in project directory")
            return False
    
    def extract_object_names(self, json_data: Dict[str, Any]) -> List[str]:
        """Extract object name list from JSON data"""
        names = []
        
        # Extract from objects.object_names
        if 'objects' in json_data and 'object_names' in json_data['objects']:
            object_names = json_data['objects']['object_names']
            if isinstance(object_names, list):
                names.extend([str(name) for name in object_names])
        
        # Extract from enhanced_objects
        if 'objects' in json_data and 'enhanced_objects' in json_data['objects']:
            enhanced_objects = json_data['objects']['enhanced_objects']
            if isinstance(enhanced_objects, list):
                for obj in enhanced_objects:
                    if isinstance(obj, dict) and 'name' in obj:
                        names.append(str(obj['name']))
        
        # Deduplicate and return
        unique_names = list(set(names))
        logger.info(f"🎯 Extracted object names from JSON: {unique_names}")
        return unique_names
    
    def detect_objects_guided(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """Use dual-model fusion for guided detection"""
        # Check if dual models exist
        has_owlv2 = hasattr(self, 'owlv2_processor') and self.owlv2_processor is not None
        logger.info(hasattr(self, 'grounding_dino_processor'))
        has_gdino = hasattr(self, 'grounding_dino_processor') and self.grounding_dino_processor is not None

        # Dual-model fusion detection
        if has_owlv2 and has_gdino:
            logger.info("🔄 Using dual-model fusion (OWLv2 + GroundingDINO)")
            return self._dual_model_fusion_detection(image_path, expected_objects)

        # Single model detection (fallback)
        elif has_owlv2:
            logger.info("🔄 Using OWLv2 only")
            return self._owlv2_detection(image_path, expected_objects)

        elif has_gdino:
            logger.info("🔄 Using GroundingDINO only")
            return self._transformers_grounding_detection(image_path, expected_objects)

        elif hasattr(self, 'grounding_dino_model') and self.grounding_dino_model is not None:
            return self._grounding_dino_detection(image_path, expected_objects)

        elif self.rf_detr_model is not None:
            logger.info("Using RF-DETR fallback for detection")
            return self._rf_detr_detection(image_path)

        else:
            return {'error': 'No detection model available'}

    def _dual_model_fusion_detection(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """Dual-model fusion detection - OWLv2 (high recall) + GroundingDINO (high precision)"""
        try:
            logger.info(f"🔍 Dual-model detection for {len(expected_objects)} objects: {expected_objects}")

            # 1. OWLv2 detection (high recall, slightly lower threshold)
            logger.info("   → Running OWLv2 (high recall)...")
            owlv2_result = self._owlv2_detection(image_path, expected_objects)

            # 2. GroundingDINO detection (high precision)
            logger.info("   → Running GroundingDINO (high precision)...")
            gdino_result = self._transformers_grounding_detection(image_path, expected_objects)

            # 3. Fuse detection results
            logger.info("   → Fusing detection results...")

            owlv2_objects = owlv2_result.get('detected_objects', [])
            gdino_objects = gdino_result.get('detected_objects', [])

            # Find best detection for each expected object
            fused_objects = []
            fused_counts = {}
            detection_sources = {}

            for obj_name in expected_objects:
                # Find matches from both model results
                owlv2_matches = [obj for obj in owlv2_objects
                                if self._is_detection_match(obj['class_name'].lower(), obj_name.lower())]
                gdino_matches = [obj for obj in gdino_objects
                               if self._is_detection_match(obj['class_name'].lower(), obj_name.lower())]

                best_detection = None
                source = 'none'

                # Case 1: Both models detected
                if owlv2_matches and gdino_matches:
                    owlv2_best = max(owlv2_matches, key=lambda x: x['confidence'])
                    gdino_best = max(gdino_matches, key=lambda x: x['confidence'])

                    # Check if same object (IoU > 0.3)
                    iou = self._calculate_iou(owlv2_best['bbox'], gdino_best['bbox'])

                    if iou > 0.3:
                        # Same object, fuse bbox
                        best_detection = {
                            'class_name': obj_name, 
                            'bbox': self._merge_bbox(owlv2_best['bbox'], gdino_best['bbox']),
                            'confidence': (owlv2_best['confidence'] + gdino_best['confidence']) / 2,
                            'center': None, 
                            'area': None
                        }
                        source = 'both'
                        logger.info(f"   ✅✅ '{obj_name}' detected by both (IoU={iou:.2f}, conf={best_detection['confidence']:.3f})")
                    else:
                        # Different location, use high-precision GDINO
                        best_detection = gdino_best.copy()
                        best_detection['class_name'] = obj_name
                        source = 'gdino_preferred'
                        logger.info(f"   ✅ '{obj_name}' using GDINO (different locations, IoU={iou:.2f})")

                # Case 2: Only GDINO detected (high confidence)
                elif gdino_matches:
                    gdino_best = max(gdino_matches, key=lambda x: x['confidence'])
                    best_detection = gdino_best.copy()
                    best_detection['class_name'] = obj_name
                    source = 'gdino_only'
                    logger.info(f"   ✅ '{obj_name}' detected by GDINO only (conf={best_detection['confidence']:.3f})")

                # Case 3: Only OWLv2 detected
                elif owlv2_matches:
                    owlv2_best = max(owlv2_matches, key=lambda x: x['confidence'])
                    best_detection = owlv2_best.copy()
                    best_detection['class_name'] = obj_name
                    source = 'owlv2_only'
                    logger.info(f"   ⚠️ '{obj_name}' detected by OWLv2 only (conf={best_detection['confidence']:.3f})")

                # Case 4: Neither detected
                else:
                    logger.info(f"   ❌ '{obj_name}' not detected by either model")
                    continue

                # Add to results
                if best_detection:
                    # Recalculate center and area
                    x1, y1, x2, y2 = best_detection['bbox']
                    best_detection['center'] = [(x1 + x2) / 2, (y1 + y2) / 2]
                    best_detection['area'] = (x2 - x1) * (y2 - y1)

                    fused_objects.append(best_detection)
                    fused_counts[obj_name] = fused_counts.get(obj_name, 0) + 1
                    detection_sources[obj_name] = source

            # Calculate average confidence
            avg_confidence = np.mean([obj['confidence'] for obj in fused_objects]) if fused_objects else 0.0

            # Count detection sources
            both_count = sum(1 for s in detection_sources.values() if s == 'both')
            gdino_count = sum(1 for s in detection_sources.values() if 'gdino' in s)
            owlv2_count = sum(1 for s in detection_sources.values() if s == 'owlv2_only')

            logger.info(f"   📊 Fusion summary: {len(fused_objects)}/{len(expected_objects)} detected")
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
                'detection_sources': detection_sources,
                'fusion_stats': {
                    'both_models': both_count,
                    'gdino_only': gdino_count,
                    'owlv2_only': owlv2_count,
                    'dual_confirmation_rate': both_count / len(expected_objects) if expected_objects else 0.0
                }
            }

        except Exception as e:
            logger.error(f"Dual-model fusion detection failed: {e}")
            return {'error': str(e)}

    def _merge_detection_results(self, primary_result: Dict[str, Any],
                               fallback_result: Dict[str, Any],
                               expected_objects: List[str]) -> Dict[str, Any]:
        """Merge two detection results, prioritize high match with expected objects"""
        try:
            merged_objects = []
            merged_counts = {}

            # Add high quality detections from primary results
            primary_objects = primary_result.get('detected_objects', [])
            for obj in primary_objects:
                merged_objects.append(obj)
                class_name = obj.get('class_name', '')
                merged_counts[class_name] = merged_counts.get(class_name, 0) + 1

            # Add missing objects from fallback results
            fallback_objects = fallback_result.get('detected_objects', [])
            expected_set = set(obj.lower() for obj in expected_objects)

            for obj in fallback_objects:
                class_name = obj.get('class_name', '').lower()
                # If this category is not yet detected and is in expected list, add it
                if class_name in expected_set and class_name not in [o.get('class_name', '').lower() for o in merged_objects]:
                    merged_objects.append(obj)
                    merged_counts[obj.get('class_name', '')] = merged_counts.get(obj.get('class_name', ''), 0) + 1
                    logger.info(f"   Added missing object '{class_name}' from fallback detection")

            # Calculate average confidence
            confidences = [obj.get('confidence', 0) for obj in merged_objects]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            return {
                'detected_objects': merged_objects,
                'object_counts': merged_counts,
                'total_detections': len(merged_objects),
                'unique_classes': len(merged_counts),
                'average_confidence': avg_confidence,
                'detection_method': 'Merged-GroundingDINO-RFDETR'
            }

        except Exception as e:
            logger.error(f"Failed to merge detection results: {e}")
            return primary_result  # Return primary result as fallback

    def _owlv2_detection(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """Detect using OWLv2 model"""
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

            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")

            # For low resolution images, upscale to improve detection
            if min(image.size) < 512:
                scale_factor = 512 / min(image.size)
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Upscaled image to {new_size} for better detection")

            # OWLv2 requires text as list of lists
            texts = [expected_objects]

            # Preprocess
            inputs = self.owlv2_processor(text=texts, images=image, return_tensors="pt")
            inputs = {k: v.to(self.owlv2_device) for k, v in inputs.items()}

            # Inference
            with torch.no_grad():
                outputs = self.owlv2_model(**inputs)

            # Post-processing - OWLv2 uses different post-processing method
            target_sizes = torch.Tensor([image.size[::-1]])  # [height, width]
            results = self.owlv2_processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.confidence_threshold
            )[0]

            # If too few detections, retry with lower threshold
            if len(results["boxes"]) < len(expected_objects) * 0.5:
                logger.info(f"Low detection count ({len(results['boxes'])}), retrying with lower threshold...")
                lower_threshold = max(0.1, self.confidence_threshold * 0.5)
                results = self.owlv2_processor.post_process_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=lower_threshold
                )[0]
                logger.info(f"Retry with threshold {lower_threshold}: found {len(results['boxes'])} objects")

            # Parse results
            raw_detections = []

            if len(results["boxes"]) > 0:
                for i in range(len(results["boxes"])):
                    box = results["boxes"][i].cpu().numpy()  # [x1, y1, x2, y2]
                    score = results["scores"][i].cpu().item()
                    label_idx = results["labels"][i].cpu().item()

                    # Get class name from label index
                    class_name = expected_objects[label_idx] if label_idx < len(expected_objects) else f"unknown_{label_idx}"

                    # Calculate center and area
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

                # Count objects
                object_counts = {}
                for obj in detected_objects:
                    class_name = obj['class_name']
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
            else:
                detected_objects = []
                object_counts = {}

            # Calculate average confidence
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

    def _transformers_grounding_detection(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """GroundingDINO detection using transformers library"""
        try:
            # Build enhanced text prompt for artwork objects
            if not expected_objects:
                text_prompt = "artwork objects . painting elements . artistic subjects"
            else:
                # Add context and synonyms for artwork objects
                enhanced_objects = []
                for obj in expected_objects:
                    obj_lower = obj.lower()
                    if 'microwave' in obj_lower:
                        enhanced_objects.extend(['microwave', 'oven', 'kitchen appliance'])
                    elif 'horse' in obj_lower:
                        enhanced_objects.extend(['horse', 'steed', 'equine'])
                    elif 'zebra' in obj_lower:
                        enhanced_objects.extend(['zebra', 'striped horse'])
                    elif 'knife' in obj_lower:
                        enhanced_objects.extend(['knife', 'blade', 'dagger'])
                    else:
                        enhanced_objects.append(obj)

                # Deduplicate and build prompt
                unique_objects = list(dict.fromkeys(enhanced_objects))  # Keep order while deduplicating
                text_prompt = " . ".join(unique_objects)

            logger.info(f"Transformers GroundingDINO detection with prompt: '{text_prompt}'")
            
            # Load and preprocess image, optimized for artwork detection
            image = Image.open(image_path).convert("RGB")

            # Upscale low resolution images to improve detection
            if min(image.size) < 512:
                scale_factor = 512 / min(image.size)
                new_size = (int(image.size[0] * scale_factor), int(image.size[1] * scale_factor))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.info(f"Upscaled image to {new_size} for better detection")
            
            # Preprocess
            inputs = self.grounding_dino_processor(images=image, text=text_prompt, return_tensors="pt")
            inputs = {k: v.to(self.grounding_dino_device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = self.grounding_dino_model(**inputs)
            
            # Post-processing - Use multi-threshold strategy to improve detection
            # First try normal threshold
            results = self.grounding_dino_processor.post_process_grounded_object_detection(
                outputs, inputs["input_ids"], threshold=self.confidence_threshold, target_sizes=[image.size[::-1]]
            )[0]

            # If too few detections, retry with lower threshold
            if len(results["boxes"]) < len(expected_objects) * 0.5:  # If fewer than half of expected objects detected
                logger.info(f"Low detection count ({len(results['boxes'])}), retrying with lower threshold...")
                lower_threshold = max(0.1, self.confidence_threshold * 0.5)
                results = self.grounding_dino_processor.post_process_grounded_object_detection(
                    outputs, inputs["input_ids"], threshold=lower_threshold, target_sizes=[image.size[::-1]]
                )[0]
                logger.info(f"Retry with threshold {lower_threshold}: found {len(results['boxes'])} objects")
            
            # Parse results
            raw_detections = []

            if len(results["boxes"]) > 0:
                for i in range(len(results["boxes"])):
                    box = results["boxes"][i].cpu().numpy()  # [x1, y1, x2, y2]
                    score = results["scores"][i].cpu().item()
                    label = results["labels"][i]

                    # Calculate center and area
                    x1, y1, x2, y2 = box
                    center = [(x1 + x2) / 2, (y1 + y2) / 2]
                    area = (x2 - x1) * (y2 - y1)

                    # Extract class name from label
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

                # Map detection results back to original expected object names
                detected_objects = self._map_to_original_objects(nms_detections, expected_objects)
                logger.info(f"Object mapping: {len(nms_detections)} filtered -> {len(detected_objects)} final objects")

                # Re-count objects (based on original names)
                object_counts = {}
                for obj in detected_objects:
                    class_name = obj['class_name']
                    object_counts[class_name] = object_counts.get(class_name, 0) + 1
            else:
                detected_objects = []
                object_counts = {}

            # Calculate average confidence
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
        """Apply Non-Maximum Suppression (NMS) to remove duplicate detections"""
        if not detections:
            return []

        # Sort by confidence
        detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        # NMS algorithm
        keep = []
        while detections_sorted:
            # Take highest confidence
            current = detections_sorted.pop(0)
            keep.append(current)

            # Remove detections with high overlap
            remaining = []
            for det in detections_sorted:
                iou = self._calculate_iou(current['bbox'], det['bbox'])

                # Keep if IoU is below threshold or if semantic categories are different
                if iou < iou_threshold or not self._are_semantic_similar(current['class_name'], det['class_name']):
                    remaining.append(det)
                else:
                    logger.debug(f"NMS removed: {det['class_name']} (IoU={iou:.3f} with {current['class_name']})")

            detections_sorted = remaining

        return keep

    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU of two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def _are_semantic_similar(self, class1: str, class2: str) -> bool:
        """Determine if two classes are semantically similar"""
        class1_lower = class1.lower()
        class2_lower = class2.lower()

        # Microwave related
        microwave_terms = {'microwave', 'microwave oven', 'oven'}
        if class1_lower in microwave_terms and class2_lower in microwave_terms:
            return True

        # Horse related
        horse_terms = {'horse', 'zebra', 'steed', 'equine'}
        if class1_lower in horse_terms and class2_lower in horse_terms:
            return True

        # Knife related
        knife_terms = {'knife', 'blade', 'dagger'}
        if class1_lower in knife_terms and class2_lower in knife_terms:
            return True

        # Exact match
        if class1_lower == class2_lower:
            return True

        return False

    def _map_to_original_objects(self, detections: List[Dict[str, Any]],
                                original_objects: List[str]) -> List[Dict[str, Any]]:
        """Map detection results back to original expected object names"""
        mapped_results = []

        # Find best matching detection for each original object
        for original_obj in original_objects:
            original_lower = original_obj.lower()

            # Find all detections matching the original object
            matching_detections = []
            for detection in detections:
                detected_class = detection['class_name'].lower()

                # Check for match
                if self._is_detection_match(detected_class, original_lower):
                    matching_detections.append(detection)

            # If matching detections found, select the one with highest confidence
            if matching_detections:
                best_detection = max(matching_detections, key=lambda x: x['confidence'])

                # Create mapped result, keeping original object name
                mapped_detection = best_detection.copy()
                mapped_detection['class_name'] = original_obj  # Use original name
                mapped_detection['original_detection'] = best_detection['class_name']  # Save actual detected class

                mapped_results.append(mapped_detection)
                logger.debug(f"Mapped '{best_detection['class_name']}' -> '{original_obj}' (confidence: {best_detection['confidence']:.3f})")

        return mapped_results

    def _is_detection_match(self, detected_class: str, original_obj: str) -> bool:
        """Determine if detection result matches original object"""
        # Direct match
        if detected_class == original_obj:
            return True

        # Semantic matching rules
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

        # Substring match (as fallback)
        if original_obj in detected_class or detected_class in original_obj:
            return True

        return False

    def _merge_bbox(self, bbox1: List[float], bbox2: List[float], weights: List[float] = [0.5, 0.5]) -> List[float]:
        """Fuse two bboxes - weighted average"""
        x1 = bbox1[0] * weights[0] + bbox2[0] * weights[1]
        y1 = bbox1[1] * weights[0] + bbox2[1] * weights[1]
        x2 = bbox1[2] * weights[0] + bbox2[2] * weights[1]
        y2 = bbox1[3] * weights[0] + bbox2[3] * weights[1]
        return [float(x1), float(y1), float(x2), float(y2)]

    def _grounding_dino_detection(self, image_path: str, expected_objects: List[str]) -> Dict[str, Any]:
        """GroundingDINO detection implementation"""
        try:
            # Build text prompt
            if not expected_objects:
                text_prompt = "objects"
            else:
                text_prompt = " . ".join(expected_objects)
            
            logger.info(f"GroundingDINO detection with prompt: '{text_prompt}'")
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            
            # GroundingDINO prediction
            boxes, logits, phrases = predict(
                model=self.grounding_dino_model,
                image=image,
                caption=text_prompt,
                box_threshold=self.confidence_threshold,
                text_threshold=0.25
            )
            
            # Parse results
            detected_objects = []
            object_counts = {}
            
            for i, (box, confidence, phrase) in enumerate(zip(boxes, logits, phrases)):
                # box is normalized coordinate, need to convert to pixel coordinate
                h, w = image.size[1], image.size[0]
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
                
                # Calculate center and area
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                area = (x2 - x1) * (y2 - y1)
                
                # Clean class name
                class_name = phrase.strip().lower()
                
                detected_obj = {
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center[0]), float(center[1])],
                    'area': float(area)
                }
                
                detected_objects.append(detected_obj)
                
                # Count statistics
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
        """RF-DETR detection implementation (fallback method)"""
        if self.rf_detr_model is None:
            return {'error': 'RF-DETR model not loaded'}

        try:
            # Read with PIL and ensure RGB
            image = Image.open(image_path).convert("RGB")

            # Inference (threshold corresponds to set confidence)
            detections = self.rf_detr_model.predict(image, threshold=self.confidence_threshold)

            # Debug: Check actual structure returned by RF-DETR
            logger.info(f"RF-DETR detection result type: {type(detections)}")
            logger.info(f"RF-DETR detection attributes: {dir(detections) if detections else 'None'}")
            
            # rfdetr returned detections generally contain: xyxy (np.ndarray Nx4), class_id (N,), confidence (N,)
            # Add robustness check
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
            
            # Check if class_name field exists
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
                
                # Force use of our COCO mapping, ignore potential erroneous class_name from RF-DETR
                default_names = self._get_default_coco_names()
                class_name = default_names.get(class_id, f"unknown_class_{class_id}")
                
                # Ensure class_name is string type
                class_name = str(class_name)
                
                logger.info(f"Mapped: class_id={class_id} -> class_name='{class_name}'")
                
                # Special case logging
                if class_id == 78:  # hair drier
                    logger.info(f"Class 78 (hair drier) detected - might be microwave misclassification")
                elif class_id == 79:  # toothbrush
                    logger.info(f"Class 79 (toothbrush) detected - might be microwave misclassification")
                elif class_id in [19, 21]:  # cow category
                    logger.info(f"Class {class_id} (cow) detected - might be horse-like animal")
                elif class_id == 22:  # zebra
                    logger.info(f"Class 22 (zebra) detected - correct match")
                elif class_id in [43, 44]:  # knife, spoon
                    logger.info(f"Class {class_id} (kitchen utensil) detected")
                elif class_id == 49:  # orange
                    logger.info(f"Class 49 (orange) detected - unexpected for artwork")
                elif class_id == 24:  # backpack
                    logger.info(f"Class 24 (backpack) detected")
                elif class_id == 55:  # cake
                    logger.info(f"Class 55 (cake) detected")
                
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

            # Keep consistent with original structure
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
        """Extract expected object information from JSON - Supports new JSON structure"""
        expected_info = {
            'object_names': [],
            'object_details': [],
            'primary_objects': [],
            'secondary_objects': [],
            'object_ids': [],
            'object_metadata': {}
        }
        
        try:
            # Extract from objects field
            objects_data = json_data.get('objects', {})
            
            # Basic object name list (if exists)
            object_names = objects_data.get('object_names', [])
            if isinstance(object_names, list):
                expected_info['object_names'] = object_names
            
            # Object ID list (if exists)
            object_ids = objects_data.get('object_ids', [])
            if isinstance(object_ids, list):
                expected_info['object_ids'] = object_ids
            
            # Enhanced object information (Main source for new structure)
            enhanced_objects = objects_data.get('enhanced_objects', [])
            if isinstance(enhanced_objects, list):
                # If object_names is empty, extract from enhanced_objects
                if not expected_info['object_names']:
                    expected_info['object_names'] = [obj.get('name', '') for obj in enhanced_objects if obj.get('name')]
                
                # If object_ids is empty, extract from enhanced_objects
                if not expected_info['object_ids']:
                    expected_info['object_ids'] = [obj.get('object_id', -1) for obj in enhanced_objects if obj.get('object_id') is not None]
                
                for obj in enhanced_objects:
                    if isinstance(obj, dict):
                        obj_name = obj.get('name', '')
                        obj_id = obj.get('object_id', -1)
                        
                        if obj_name:
                            # Detailed object information
                            obj_info = {
                                'name': obj_name,
                                'object_id': obj_id,
                                'quantity': obj.get('quantity', 1),
                                'importance': obj.get('importance', 'secondary'),
                                'size': obj.get('size', 'medium'),
                                'state': obj.get('state', ''),
                                'primary_colors': obj.get('primary_colors', []),
                                'material': obj.get('material', ''),
                                'artistic_description': obj.get('artistic_description', ''),
                                'position_preference': obj.get('position_preference', 'midground')
                            }
                            
                            expected_info['object_details'].append(obj_info)
                            
                            # Store object metadata for cross-validation
                            expected_info['object_metadata'][obj_name] = {
                                'id': obj_id,
                                'size': obj_info['size'],
                                'colors': obj_info['primary_colors'],
                                'material': obj_info['material'],
                                'state': obj_info['state']
                            }
                            
                            # Classify by importance
                            if obj_info['importance'] in ['primary', 'main', 'dominant']:
                                expected_info['primary_objects'].append(obj_name)
                            else:
                                expected_info['secondary_objects'].append(obj_name)
            
            # If no enhanced_objects but basic object_names exist, create basic info
            if not expected_info['object_details'] and expected_info['object_names']:
                for i, obj_name in enumerate(expected_info['object_names']):
                    obj_id = expected_info['object_ids'][i] if i < len(expected_info['object_ids']) else -1
                    expected_info['object_details'].append({
                        'name': obj_name,
                        'object_id': obj_id,
                        'quantity': 1,
                        'importance': 'secondary'
                    })
                    expected_info['secondary_objects'].append(obj_name)
                    
            # Get additional object info from composition
            composition = json_data.get('composition', {})
            if isinstance(composition, dict):
                spatial_rels = composition.get('spatial_relationships', {})
                if isinstance(spatial_rels, dict):
                    object_positions = spatial_rels.get('object_positions', {})
                    for obj_name in object_positions.keys():
                        if obj_name not in expected_info['object_names']:
                            expected_info['object_names'].append(obj_name)
                            expected_info['secondary_objects'].append(obj_name)
                    
        except Exception as e:
            logger.error(f"Error extracting expected objects: {e}")
        
        return expected_info
    
    def validate_object_consistency(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency of object information in JSON"""
        consistency_issues = []
        consistency_score = 100.0
        
        try:
            expected_info = self.extract_expected_objects(json_data)
            
            # Check consistency between object_names and enhanced_objects
            object_names_set = set(expected_info['object_names'])
            enhanced_names_set = set([obj['name'] for obj in expected_info['object_details']])
            
            if object_names_set != enhanced_names_set:
                missing_in_enhanced = object_names_set - enhanced_names_set
                extra_in_enhanced = enhanced_names_set - object_names_set
                
                if missing_in_enhanced:
                    consistency_issues.append(f"Objects in object_names but missing in enhanced_objects: {list(missing_in_enhanced)}")
                    consistency_score -= 15
                
                if extra_in_enhanced:
                    consistency_issues.append(f"Objects in enhanced_objects but missing in object_names: {list(extra_in_enhanced)}")
                    consistency_score -= 10
            
            # Check count consistency between object_ids and enhanced_objects
            if expected_info['object_ids'] and len(expected_info['object_ids']) != len(expected_info['object_details']):
                consistency_issues.append(f"Mismatch between object_ids count ({len(expected_info['object_ids'])}) and enhanced_objects count ({len(expected_info['object_details'])})")
                consistency_score -= 20
            
            # Check uniqueness of object_ids
            if expected_info['object_ids']:
                valid_ids = [id for id in expected_info['object_ids'] if id != -1]
                if len(valid_ids) != len(set(valid_ids)):
                    consistency_issues.append("Duplicate object_ids found")
                    consistency_score -= 25
            
            # Check if objects in composition are all in object_names
            composition = json_data.get('composition', {})
            if isinstance(composition, dict):
                spatial_rels = composition.get('spatial_relationships', {})
                if isinstance(spatial_rels, dict):
                    composition_objects = set(spatial_rels.get('object_positions', {}).keys())
                    missing_in_objects = composition_objects - object_names_set
                    if missing_in_objects:
                        consistency_issues.append(f"Objects in composition but missing in object_names: {list(missing_in_objects)}")
                        consistency_score -= 10
            
            return {
                'consistency_score': max(0, consistency_score),
                'issues': consistency_issues,
                'total_issues': len(consistency_issues),
                'is_consistent': len(consistency_issues) == 0
            }
            
        except Exception as e:
            return {
                'consistency_score': 0.0,
                'issues': [f"Consistency validation failed: {str(e)}"],
                'total_issues': 1,
                'is_consistent': False
            }
    
    def calculate_object_matching_score(self, detected_counts: Dict[str, int], 
                                      expected_names: List[str]) -> Dict[str, Any]:
        """Calculate object matching score"""
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
        
        # Normalize object names for matching
        def normalize_name(name):
            return str(name).lower().strip()
        
        # Enhanced semantic synonym mapping (optimized for artwork scenes)
        semantic_mapping = {
            # Kitchen appliances
            'microwave': ['microwave', 'oven', 'toaster', 'hair drier', 'appliance', 'toothbrush', 'kitchen appliance'],
            'oven': ['oven', 'microwave', 'toaster', 'hair drier', 'appliance'],
            'refrigerator': ['refrigerator', 'fridge'],

            # People
            'person': ['person', 'people', 'human', 'man', 'woman', 'figure', 'character'],

            # Animals (common in artworks)
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

            # Reverse mapping (COCO misclassification correction)
            'hair drier': ['hair drier', 'microwave', 'appliance'],
            'toothbrush': ['toothbrush', 'microwave', 'knife'],

            # Artwork-specific objects
            'angel': ['person', 'human', 'figure', 'winged figure'],
            'deity': ['person', 'human', 'figure', 'divine figure'],
            'warrior': ['person', 'human', 'figure', 'armored figure'],
        }
        
        # Direct mapping from COCO class ID to expected objects (based on your detection results)
        coco_id_to_expected = {
            78: 'microwave',   # hair drier -> microwave
            79: 'microwave',   # toothbrush -> microwave (possible misclassification)
            22: 'zebra',       # zebra -> zebra (correct)
            17: 'horse',       # horse -> horse (correct)
            19: 'horse',       # cow -> horse (possible misclassification)
            21: 'horse',       # cow -> horse (possible misclassification) 
            43: 'knife',       # knife -> knife (correct)
            44: 'knife',       # spoon -> knife (similar object)
            42: 'knife',       # fork -> knife (similar object)
        }
        
        def find_semantic_match(expected, detected_list):
            """Find semantic match"""
            expected_lower = expected.lower()
            
            # Direct match
            if expected_lower in [d.lower() for d in detected_list]:
                return expected_lower
            
            # Semantic match - Check if synonyms of expected object are in detection list
            if expected_lower in semantic_mapping:
                for synonym in semantic_mapping[expected_lower]:
                    if synonym in [d.lower() for d in detected_list]:
                        logger.info(f"Semantic match: '{expected}' matched with detected '{synonym}'")
                        return synonym
            
            # Reverse semantic match - Check if detected object is synonym of expected object
            for detected in detected_list:
                detected_lower = detected.lower()
                if detected_lower in semantic_mapping:
                    if expected_lower in semantic_mapping[detected_lower]:
                        logger.info(f"Reverse semantic match: '{expected}' matched with detected '{detected}'")
                        return detected_lower
            
            # Special handling: Smart matching based on COCO class_id
            for detected in detected_list:
                detected_lower = detected.lower()
                
                # If specific COCO class name is detected, check if it matches expected object
                for class_id, expected_obj in coco_id_to_expected.items():
                    # Get COCO class name corresponding to this class_id
                    default_names = {
                        78: 'hair drier', 79: 'toothbrush', 22: 'zebra', 
                        17: 'horse', 19: 'cow', 21: 'cow', 
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
        
        # Calculate matching
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
            
            # 3. Fuzzy match (inclusion relation)
            found = False
            for detected_norm in detected_normalized:
                if expected_norm in detected_norm or detected_norm in expected_norm:
                    matched_objects.append(expected)
                    logger.info(f"Fuzzy match found: '{expected}' matched with '{detected_norm}'")
                    found = True
                    break
            
            if not found:
                missing_objects.append(expected)
        
        # Extra objects
        matched_normalized = [normalize_name(obj) for obj in matched_objects]
        extra_objects = []
        for detected_norm, count in detected_normalized.items():
            if detected_norm not in expected_normalized and detected_norm not in matched_normalized:
                # Reverse check for fuzzy match
                found = False
                for exp_norm in expected_normalized:
                    if exp_norm in detected_norm or detected_norm in exp_norm:
                        found = True
                        break
                if not found:
                    extra_objects.append(detected_norm)
        
        # Calculate metrics
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
            accuracy_score = (recall * 70 + precision * 30)  # Recall has higher weight
        
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
        """Analyze object distribution"""
        if not detected_objects:
            return {
                'coverage_ratio': 0.0,
                'balance_score': 0.0,
                'spatial_distribution': 'empty',
                'object_density': 0.0
            }
        
        try:
            # Read image to get dimensions
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Could not load image for distribution analysis'}
            
            height, width = image.shape[:2]
            image_area = width * height
            
            # Calculate coverage ratio
            total_bbox_area = sum(obj['area'] for obj in detected_objects)
            coverage_ratio = total_bbox_area / image_area
            
            # Analyze spatial distribution
            centers = [obj['center'] for obj in detected_objects]
            
            # Calculate centroid
            centroid_x = np.mean([c[0] for c in centers])
            centroid_y = np.mean([c[1] for c in centers])
            
            # Calculate distribution uniformity (based on distance to centroid)
            distances = [np.sqrt((c[0] - centroid_x)**2 + (c[1] - centroid_y)**2) for c in centers]
            balance_score = 100.0 - min(np.std(distances) / max(width, height) * 100, 100.0)
            
            # Object density
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
        """Create detection result visualization"""
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
                    
                    # Draw bounding box
                    rect = patches.Rectangle(
                        (bbox[0], bbox[1]), 
                        bbox[2] - bbox[0], 
                        bbox[3] - bbox[1],
                        linewidth=2, 
                        edgecolor='red', 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # Add label
                    ax.text(bbox[0], bbox[1] - 5, 
                           f'{class_name}: {confidence:.2f}',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           fontsize=10, color='black')
            
            ax.set_title(f'RF-DETR Object Detection Results\n'
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
        """Main object detection evaluation method"""
        return self.comprehensive_object_detection_evaluation(json_data, image_path)
    
    def comprehensive_object_detection_evaluation(self, json_data: Dict[str, Any], 
                                                 image_path: str) -> Dict[str, Any]:
        """
        Comprehensive Object Detection Evaluation - Enhanced Version
        
        Args:
            json_data: Artwork JSON data
            image_path: Path to image file
            
        Returns:
            Complete object detection evaluation results
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
            detection_result = self.detect_objects_guided(image_path, expected_object_names)

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

            # Average confidence score
            confidence_score = detection_result['average_confidence'] * 100

            # 5. Simplified scoring - Keep only two core dimensions
            weights = {
                'accuracy': 0.7,       # 70% - Detection accuracy (core metric)
                'confidence': 0.3,     # 30% - Confidence (quality metric)
            }

            overall_score = (
                matching_result['accuracy_score'] * weights['accuracy'] +
                confidence_score * weights['confidence']
            )

            print(f"   Object detection evaluation completed! Overall score: {overall_score:.1f}/100")

            # 6. Build simplified results
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
            error_msg = f"Enhanced object detection evaluation failed: {str(e)}"
            logger.error(error_msg)
            return {'error': error_msg}

            
                        
            
            
            