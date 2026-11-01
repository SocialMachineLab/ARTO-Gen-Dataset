
import os
import json
import glob
import argparse
import logging
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

# Library imports for models
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    print("Warning: 'ultralytics' not installed. YOLO model will not run.")

try:
    from rfdetr import RFDETRLarge
    # RF-DETR class ids are COCO category ids (1-90, with gaps).
    from rfdetr.util.coco_classes import COCO_CLASSES as RFDETR_CLASSES
except ImportError:
    RFDETRLarge = None
    RFDETR_CLASSES = {}
    print("Warning: 'rfdetr' not installed. RF-DETR model will not run.")

try:
    from transformers import AutoProcessor, GroundingDinoForObjectDetection, AutoModelForZeroShotObjectDetection
except ImportError:
    AutoProcessor = None
    GroundingDinoForObjectDetection = None
    AutoModelForZeroShotObjectDetection = None
    print("Warning: 'transformers' not installed. HuggingFace models will not run.")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration constants
IOU_THRESHOLD = 0.5
# Below the RF-DETR default to raise recall
CONFIDENCE_THRESHOLD_RFDETR = 0.05
# OWLv2 threshold
CONFIDENCE_THRESHOLD_OWL = 0.1
DEFAULT_IMAGE_EXT = "*.png"
GROUND_TRUTH_SUFFIX = "_groundtruth.json"

# Saved prediction files next to each image: <image>_<suffix>_result.json
SAVED_RESULT_SUFFIX = {
    "GroundingDINO": "GroundingDINO",
    "OWLv2": "OWLv2",
    "YOLOv11-large": "YOLOv11",
}

# Headless OpenCV stubs
# Prevents GUI errors on headless servers
if not hasattr(cv2, 'imshow'):
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: -1
    cv2.destroyAllWindows = lambda *args, **kwargs: None
    cv2.namedWindow = lambda *args, **kwargs: None
    cv2.setMouseCallback = lambda *args, **kwargs: None
    # Constants required if cv2 is not strictly fully imported or mocked
    if not hasattr(cv2, 'IMREAD_COLOR'): cv2.IMREAD_COLOR = 1
    if not hasattr(cv2, 'IMREAD_GRAYSCALE'): cv2.IMREAD_GRAYSCALE = 0
    if not hasattr(cv2, 'IMREAD_UNCHANGED'): cv2.IMREAD_UNCHANGED = -1
    if not hasattr(cv2, 'WINDOW_NORMAL'): cv2.WINDOW_NORMAL = 0
    if not hasattr(cv2, 'WINDOW_AUTOSIZE'): cv2.WINDOW_AUTOSIZE = 1


def get_ground_truth(json_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Extracts ground truth boxes and labels from a JSON file.

    Args:
        json_path: Path to the JSON ground truth file.

    Returns:
        A tuple containing:
        - List of ground truth dictionaries with "box" (xyxy) and "phrase" (label).
        - List of all object names (labels) present in the image.
    """
    if not os.path.exists(json_path):
        logger.warning(f"JSON file not found: {json_path}")
        return [], []

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        ground_truth = []
        if "detections" in data:
            for det in data["detections"]:
                # Normalize label to lowercase and strip whitespace
                ground_truth.append({
                    "box": det["box"],
                    "phrase": det["phrase"].strip().lower()
                })
        
        # Object names for open-vocabulary queries
        object_names = [name.strip().lower() for name in data.get("object_names", [])]
        return ground_truth, object_names
        
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON file: {json_path}")
        return [], []
    except Exception as e:
        logger.error(f"Unexpected error reading {json_path}: {e}")
        return [], []


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU score (float).
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def calculate_ap(ground_truth: List[Dict], predictions: List[Dict], iou_threshold: float = 0.5) -> Tuple[float, int, int, int]:
    """
    Calculates Average Precision (AP) for one class (VOC all-point interpolation).

    A prediction matches only a ground-truth box of the same image, and each
    ground-truth box at most once. Recall is TP over all ground-truth boxes, so
    missed objects lower the AP.

    Args:
        ground_truth: List of GT items, each with "box" and "image_id".
        predictions: List of prediction items, each with "box", "confidence" and "image_id".
        iou_threshold: IoU threshold for a match.

    Returns:
        Tuple of (AP, TP, FP, FN).
    """
    if not ground_truth:
        return np.nan, 0, len(predictions), 0

    if not predictions:
        return 0.0, 0, 0, len(ground_truth)

    predictions = sorted(predictions, key=lambda x: x.get("confidence", 0.0) or 0.0, reverse=True)

    gt_by_image: Dict[Any, List[int]] = {}
    for idx, gt in enumerate(ground_truth):
        gt_by_image.setdefault(gt.get("image_id"), []).append(idx)
    matched_gt = [False] * len(ground_truth)

    hits = []
    for pred in predictions:
        best_iou = 0.0
        best_gt_idx = -1
        for idx in gt_by_image.get(pred.get("image_id"), []):
            iou = calculate_iou(pred["box"], ground_truth[idx]["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        # A second detection of an already matched box is a false positive.
        if best_iou > iou_threshold and not matched_gt[best_gt_idx]:
            matched_gt[best_gt_idx] = True
            hits.append(1)
        else:
            hits.append(0)

    hits = np.array(hits)
    tp_cum = np.cumsum(hits)
    fp_cum = np.cumsum(1 - hits)
    recall = tp_cum / len(ground_truth)
    precision = tp_cum / (tp_cum + fp_cum)

    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    steps = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[steps + 1] - mrec[steps]) * mpre[steps + 1]))

    tp = int(hits.sum())
    fp = len(hits) - tp
    fn = len(ground_truth) - tp
    return ap, tp, fp, fn


# Model inference wrappers

def run_yolo(model, image_path: str) -> List[Dict]:
    """Runs YOLO inference."""
    if model is None: return []
    try:
        results = model(image_path, verbose=False)
        detections = []
        for i, box in enumerate(results[0].boxes):
            detections.append({
                "box": box.xyxy[0].tolist(),
                "confidence": box.conf[0].item(),
                "phrase": results[0].names[int(box.cls[0])].lower(),
                "index": i
            })
        return detections
    except Exception as e:
        logger.error(f"YOLO inference failed: {e}")
        return []

def run_rfdetr(model, image_path: str, debug: bool = False) -> List[Dict]:
    """Runs RF-DETR inference."""
    if model is None: return []
    
    if debug:
        logger.info(f"Debugging RF-DETR for image: {os.path.basename(image_path)}")

    try:
        image = Image.open(image_path)
        # Using a slightly lower threshold for better recall in complex scenes
        results = model.predict(image, threshold=CONFIDENCE_THRESHOLD_RFDETR)
    except Exception as e:
        logger.error(f"RF-DETR prediction failed for {image_path}: {e}")
        return []

    detections = []
    if len(results) > 0:
        try:
            boxes = results.xyxy if hasattr(results, 'xyxy') else None
            confidences = results.confidence if hasattr(results, 'confidence') else None
            class_ids = results.class_id if hasattr(results, 'class_id') else None

            num_detections = len(results)
            for i in range(num_detections):
                try:
                    # Safe Box Extraction
                    if boxes is not None:
                        box = boxes[i]
                        if hasattr(box, 'tolist'):
                            box_list = box.tolist()
                        elif isinstance(box, (list, tuple)):
                            box_list = list(box)
                        else:
                            # Fallback for other indexable box types
                            box_list = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
                    else:
                        continue

                    # Safe Confidence Extraction
                    if confidences is not None:
                        conf = confidences[i]
                        conf_float = conf.item() if hasattr(conf, 'item') else float(conf)
                    else:
                        conf_float = 0.5

                    # Safe Class Extraction
                    if class_ids is not None:
                        class_id = class_ids[i]
                        try:
                            label = RFDETR_CLASSES.get(int(class_id), "unknown")
                        except (ValueError, TypeError):
                            label = "unknown"
                    else:
                        label = "unknown"

                    detections.append({
                        "box": box_list,
                        "confidence": conf_float,
                        "phrase": label.lower(),
                        "index": i
                    })

                except Exception as e:
                    if debug: logger.warning(f"Error parsing detection {i}: {e}")
                    continue
        except Exception as e:
             if debug: logger.error(f"Error processing RF-DETR results object: {e}")

    if debug:
        logger.info(f"RF-DETR returned {len(detections)} valid detections")
    
    return detections

def run_grounding_dino(processor, model, image_path: str, text_query: str) -> List[Dict]:
    """Runs Grounding DINO inference with a text query."""
    if model is None or not text_query: return []
    
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, text=text_query, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[image.size[::-1]] # (height, width)
        )
        
        detections = []
        # results[0]: one image per call
        for i, (box, label, score) in enumerate(zip(results[0]["boxes"], results[0]["labels"], results[0]["scores"])):
            detections.append({
                "box": box.tolist(),
                "confidence": score.item(),
                "phrase": label.lower(),
                "index": i
            })
        return detections
    except Exception as e:
        logger.error(f"Grounding DINO failed for {image_path}: {e}")
        return []

def run_owlv2(processor, model, image_path: str, object_names: List[str]) -> List[Dict]:
    """Runs OWLv2 inference with a list of object names."""
    if model is None or not object_names: return []
    
    try:
        image = Image.open(image_path).convert("RGB")
        # OWLv2 expects nested list for batch dimension: [[name1, name2]]
        texts = [object_names]
        inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        # Target sizes: (height, width)
        target_sizes = torch.Tensor([image.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=CONFIDENCE_THRESHOLD_OWL
        )

        detections = []
        result = results[0]
        for i, (box, score, label) in enumerate(zip(result["boxes"], result["scores"], result["labels"])):
            detections.append({
                "box": box.tolist(),
                "confidence": score.item(),
                # label is an index into the provided text list
                "phrase": texts[0][label].lower(), 
                "index": i
            })
        return detections
    except Exception as e:
        logger.error(f"OWLv2 failed for {image_path}: {e}")
        return []


def load_saved_predictions(image_path: str, suffix: str) -> Optional[List[Dict]]:
    """Read <image>_<suffix>_result.json; None if the file does not exist."""
    path = f"{os.path.splitext(image_path)[0]}_{suffix}_result.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        detections = json.load(f).get("detections", [])
    return [{"box": d["box"], "confidence": d.get("confidence", 0.0),
             "phrase": d["phrase"].strip().lower(), "index": i}
            for i, d in enumerate(detections)]


def load_models() -> Dict[str, Any]:
    """Load the detectors; a model that fails to load is skipped."""
    logger.info("Initializing models...")
    models = {}

    if YOLO is not None:
        try:
            # Model path assumes relative to execution or in standard cache
            yolo_model = YOLO("yolo11l.pt")
            models["YOLOv11-large"] = lambda img, names, txt: run_yolo(yolo_model, img)
        except Exception as e:
            logger.error(f"Failed to load YOLO: {e}")

    if RFDETRLarge is not None:
        try:
            rfdetr_model = RFDETRLarge(pretrained=True)
            models["RF-DETR-large"] = lambda img, names, txt: run_rfdetr(rfdetr_model, img, debug=False)
        except Exception as e:
            logger.error(f"Failed to load RF-DETR: {e}")

    if AutoProcessor is not None:
        try:
            gd_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            gd_model = GroundingDinoForObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
            models["GroundingDINO"] = lambda img, names, txt: run_grounding_dino(gd_processor, gd_model, img, txt)
        except Exception as e:
            logger.error(f"Failed to load GroundingDINO: {e}")

        try:
            owl_processor = AutoProcessor.from_pretrained("google/owlv2-large-patch14-ensemble")
            owl_model = AutoModelForZeroShotObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")
            models["OWLv2"] = lambda img, names, txt: run_owlv2(owl_processor, owl_model, img, names)
        except Exception as e:
            logger.error(f"Failed to load OWLv2: {e}")

    logger.info(f"Models loaded: {list(models.keys())}")
    return models


def main():
    parser = argparse.ArgumentParser(description="Object detection evaluation on generated artworks")
    parser.add_argument("--data-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory with one sub-directory per style (default: this script's directory)")
    parser.add_argument("--from-saved", action="store_true",
                        help="Score the saved <image>_<model>_result.json files instead of running the "
                             "models (GroundingDINO, OWLv2, YOLOv11; RF-DETR has no saved results)")
    parser.add_argument("--output", default=None,
                        help="Results CSV (default: <data-dir>/evaluation_results.csv)")
    args = parser.parse_args()

    if args.from_saved:
        models = {name: (lambda img, names, txt, suffix=suffix: load_saved_predictions(img, suffix))
                  for name, suffix in SAVED_RESULT_SUFFIX.items()}
    else:
        models = load_models()

    base_dir = args.data_dir
    logger.info(f"Data directory: {base_dir}")

    # Discover style directories
    # Filters out hidden directories and non-directories
    styles = [
        d for d in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, d)) 
        and not d.startswith('.')
        and not d.startswith('_') # Convention for internal folders
    ]
    
    if not styles:
        logger.warning(f"No style directories found in {base_dir}.")
        return

    logger.info(f"Found styles: {styles}")

    results_list = []
    all_classes = set()
    missing = {model: 0 for model in models}
    
    # Initialize data storage
    # Structure: style -> model -> {preds: [], gts: []}
    style_data = {
        style: {
            model: {"preds": [], "gts": []} for model in models
        } for style in styles
    }

    print("\n" + "="*80)
    print("Starting Evaluation Loop")
    print("="*80)

    for style in styles:
        print(f"\nProcessing Style: {style}")
        style_dir = os.path.join(base_dir, style)
        
        # Use simple glob for cross-platform compatibility
        image_paths = glob.glob(os.path.join(style_dir, DEFAULT_IMAGE_EXT))
        
        if not image_paths:
            logger.warning(f"No images found in {style_dir}")
            continue

        print(f"  Found {len(image_paths)} images.")

        for i, image_path in enumerate(image_paths):
            image_filename = os.path.basename(image_path)
            
            # Construct Ground Truth Filename
            # foo.png -> foo_groundtruth.json
            base_name = os.path.splitext(image_filename)[0]
            json_filename = f"{base_name}{GROUND_TRUTH_SUFFIX}"
            json_path = os.path.join(style_dir, json_filename)
            
            if not os.path.exists(json_path):
                continue

            ground_truth, object_names = get_ground_truth(json_path)
            if not ground_truth:
                 continue

            # Construct Text Query for Open-Set Models
            # e.g. "cat. dog. person"
            text_query = ". ".join(object_names)
            if not text_query.strip():
                 text_query = "object" # Fallback if empty

            # Update global class list
            for gt in ground_truth:
                all_classes.add(gt["phrase"])

            # Run each model on this image
            if i % 10 == 0:
                print(f"  [{i}/{len(image_paths)}] Processing {image_filename}...")

            for model_name, model_func in models.items():
                try:
                    predictions = model_func(image_path, object_names, text_query)
                    # No saved result for this image: leave it out for this model.
                    if predictions is None:
                        missing[model_name] += 1
                        continue

                    # Store Predictions
                    for pred in predictions:
                        pred["image_id"] = image_filename
                        style_data[style][model_name]["preds"].append(pred)
                    
                    # Store Ground Truths (duplicated for each model for independent calc)
                    for gt in ground_truth:
                        gt_copy = gt.copy()
                        gt_copy["image_id"] = image_filename
                        style_data[style][model_name]["gts"].append(gt_copy)
                        
                except Exception as e:
                    logger.error(f"Error running {model_name} on {image_filename}: {e}")

    for model_name, count in missing.items():
        if count:
            logger.warning(f"{model_name}: no saved result for {count} images; they are left out for this model")

    print("\n" + "="*80)
    print("Calculating Metrics")
    print("="*80 + "\n")

    # Metrics Calculation Loop
    for style in styles:
        for model_name in models:
            metrics_aps = []
            total_tp, total_fp, total_fn = 0, 0, 0

            style_preds = style_data[style][model_name]["preds"]
            style_gts = style_data[style][model_name]["gts"]

            if not style_gts:
                continue

            # Calculate AP per class
            for cls in all_classes:
                gts_cls = [gt for gt in style_gts if gt["phrase"] == cls]
                preds_cls = [pred for pred in style_preds if pred["phrase"] == cls]

                if not gts_cls and not preds_cls:
                    continue

                ap, tp, fp, fn = calculate_ap(gts_cls, preds_cls, iou_threshold=IOU_THRESHOLD)
                
                if not np.isnan(ap):
                    metrics_aps.append(ap)

                total_tp += tp
                total_fp += fp
                total_fn += fn

            map_score = np.mean(metrics_aps) if metrics_aps else 0.0
            
            # Micro-averaged Precision/Recall
            denom_prec = total_tp + total_fp
            denom_rec = total_tp + total_fn
            
            precision = total_tp / denom_prec if denom_prec > 0 else 0.0
            recall = total_tp / denom_rec if denom_rec > 0 else 0.0

            results_list.append({
                "Style": style, 
                "Model": model_name, 
                "mAP": map_score, 
                "Precision": precision, 
                "Recall": recall
            })
            
            print(f"Style: {style:<20} Model: {model_name:<15} mAP: {map_score:.4f}")

    # Calculate Overall Results (Across all styles)
    print("\nCalculating overall metrics...")
    for model_name in models:
        overall_aps = []
        all_tp = 0
        all_fp = 0
        all_fn = 0

        # Flatten all data for this model across all styles
        all_gts_flat = []
        all_preds_flat = []
        for style in styles:
            all_gts_flat.extend(style_data[style][model_name]["gts"])
            all_preds_flat.extend(style_data[style][model_name]["preds"])
        
        if not all_gts_flat:
            continue

        for cls in all_classes:
            gts_cls = [gt for gt in all_gts_flat if gt["phrase"] == cls]
            preds_cls = [pred for pred in all_preds_flat if pred["phrase"] == cls]

            if not gts_cls and not preds_cls:
                continue

            ap, tp, fp, fn = calculate_ap(gts_cls, preds_cls, iou_threshold=IOU_THRESHOLD)
            
            if not np.isnan(ap):
                overall_aps.append(ap)
            
            all_tp += tp
            all_fp += fp
            all_fn += fn

        map_score = np.mean(overall_aps) if overall_aps else 0.0
        
        denom_prec = all_tp + all_fp
        denom_rec = all_tp + all_fn
        
        precision = all_tp / denom_prec if denom_prec > 0 else 0.0
        recall = all_tp / denom_rec if denom_rec > 0 else 0.0

        results_list.append({
            "Style": "Overall", 
            "Model": model_name, 
            "mAP": map_score, 
            "Precision": precision, 
            "Recall": recall
        })

    # Save Results
    if results_list:
        results_df = pd.DataFrame(results_list)
        output_path = args.output or os.path.join(base_dir, "evaluation_results.csv")
        
        print("\n" + "="*80)
        print(f"Saving results to: {output_path}")
        print(results_df)
        results_df.to_csv(output_path, index=False)
        print("="*80 + "\n")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    main()
