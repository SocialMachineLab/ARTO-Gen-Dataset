"""Detect single object in generated validation images using YOLO, OWLv2, Grounding DINO, and VLM"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from ultralytics import YOLO
import warnings
import logging

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    VLM_AVAILABLE = True
except ImportError:
    VLM_AVAILABLE = False


class YOLO11Detector:
    """YOLO11 object detector"""
    def __init__(self, model_size='x', device='cuda'):
        self.device = device
        print(f"Loading YOLO11-{model_size}")
        self.model = YOLO(f'yolo11{model_size}.pt')
        self.model.to(device)
        self.class_names = self.model.names

    def detect(self, image_path, conf_threshold=0.25):
        results = self.model.predict(
            source=image_path,
            conf=conf_threshold,
            verbose=False,
            device=self.device
        )
        detections = []
        for i in range(len(results[0].boxes)):
            box = results[0].boxes[i]
            detections.append({
                'class_id': int(box.cls.item()),
                'class_name': self.class_names[int(box.cls.item())],
                'confidence': float(box.conf.item()),
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'model': 'YOLO11'
            })
        return detections


class OWLv2Detector:
    """OWLv2 open-vocabulary detector"""
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading OWLv2")
        model_name = 'google/owlv2-base-patch16-ensemble'
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def detect(self, image_path, text_queries, conf_threshold=0.1):
        if not text_queries:
            return []

        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=conf_threshold,
            target_sizes=target_sizes
        )[0]

        detections = []
        boxes = results['boxes'].cpu().numpy()
        scores = results['scores'].cpu().numpy()
        labels = results['labels'].cpu().numpy()

        for box, score, label_idx in zip(boxes, scores, labels):
            detections.append({
                'query': text_queries[label_idx],
                'confidence': float(score),
                'bbox': box.tolist(),
                'model': 'OWLv2'
            })
        return detections


class GroundingDINODetector:
    """Grounding DINO detector via Hugging Face"""
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading Grounding DINO")
        try:
            model_id = "IDEA-Research/grounding-dino-base"
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
            self.model = self.model.to(device)
            self.model.eval()
            self.available = True
        except Exception as e:
            print(f"Grounding DINO not available: {e}")
            self.available = False

    def detect(self, image_path, text_queries, conf_threshold=0.25):
        if not self.available or not text_queries:
            return []

        try:
            image = Image.open(image_path).convert('RGB')
            text = ". ".join(text_queries) + "."

            inputs = self.processor(images=image, text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)

            target_sizes = torch.tensor([image.size[::-1]], device=self.device)
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                input_ids=inputs['input_ids'],
                threshold=conf_threshold,
                target_sizes=target_sizes
            )[0]

            detections = []
            boxes = results['boxes'].cpu().numpy()
            scores = results['scores'].cpu().numpy()
            labels = results['labels']

            for box, score, label in zip(boxes, scores, labels):
                detections.append({
                    'query': label,
                    'confidence': float(score),
                    'bbox': box.tolist(),
                    'model': 'GroundingDINO'
                })
            return detections
        except Exception as e:
            print(f"Grounding DINO detection failed: {e}")
            return []


class VLMObjectDetector:
    """VLM-based object detector using Qwen2.5-VL (fallback when YOLO fails)"""
    def __init__(self, model_path="Qwen/Qwen2.5-VL-32B-Instruct-AWQ", device='cuda'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None
        self.available = False

        if not VLM_AVAILABLE:
            print("VLM not available (vLLM not installed)")
            return

        try:
            print(f"Loading VLM ({model_path})")

            vllm_config = {
                "model": model_path,
                "trust_remote_code": True,
                "max_model_len": 4096,
                "enforce_eager": True,
                "quantization": "AWQ",
                "dtype": "float16",
                "limit_mm_per_prompt": {"image": 1, "video": 1},
            }

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count >= 2:
                    vllm_config["tensor_parallel_size"] = 2
                    vllm_config["gpu_memory_utilization"] = 0.85
                else:
                    vllm_config["tensor_parallel_size"] = 1
                    vllm_config["gpu_memory_utilization"] = 0.85

            self.model = LLM(**vllm_config)
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.available = True

        except Exception as e:
            print(f"VLM not available: {e}")
            self.available = False

    def detect_objects(self, image_path):
        if not self.available:
            return {'objects': [], 'raw_response': '', 'success': False}

        try:
            prompt = """List all distinct objects you can see in this image.
Only provide object names, one per line, without numbering or descriptions.
Focus on concrete, tangible objects (people, animals, furniture, vehicles, etc.).
Do not include abstract concepts, colors, or scene descriptions.

Example output format:
person
dog
chair
table
car"""

            abs_image_path = os.path.abspath(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file:///{abs_image_path}"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            formatted_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            image_inputs, _ = process_vision_info(messages)
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs

            llm_inputs = {
                "prompt": formatted_prompt,
                "multi_modal_data": mm_data,
            }

            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=256,
                stop=["<|endoftext|>", "<|im_end|>"]
            )

            outputs = self.model.generate([llm_inputs], sampling_params=sampling_params)

            if outputs and len(outputs) > 0:
                raw_response = outputs[0].outputs[0].text.strip()
            else:
                return {'objects': [], 'raw_response': '', 'success': False}

            objects = self._parse_objects(raw_response)

            return {
                'objects': objects,
                'raw_response': raw_response,
                'success': True
            }

        except Exception as e:
            print(f"VLM detection failed: {e}")
            return {'objects': [], 'raw_response': '', 'success': False}

    def _parse_objects(self, response):
        objects = []
        lines = response.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line and line[0].isdigit():
                line = line.split('.', 1)[-1].strip()
            line = line.lstrip('•-*').strip()

            if line and len(line.split()) <= 3:
                objects.append(line.lower())

        seen = set()
        unique = []
        for obj in objects:
            if obj not in seen:
                seen.add(obj)
                unique.append(obj)

        return unique


def parse_filename(filename):
    """Parse ground truth object from filename: {object}_{style}_{id}.png"""
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_')

    if len(parts) >= 3:
        object_name = parts[0]
        return object_name
    else:
        print(f"Cannot parse filename: {filename}")
        return None


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])

    intersection = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def find_matching_detection(detections, ground_truth_object):
    """Find detection that matches ground truth object"""
    matches = []

    for det in detections:
        detected_obj = det.get('class_name') or det.get('query', '')
        detected_obj = detected_obj.lower()
        gt_obj = ground_truth_object.lower()

        if detected_obj == gt_obj or gt_obj in detected_obj or detected_obj in gt_obj:
            matches.append(det)

    if not matches:
        return None

    return max(matches, key=lambda x: x['confidence'])


def visualize_single_object_detection(image_path, ground_truth,
                                       yolo_det, owl_det, dino_det,
                                       bbox_consistency, output_dir):
    image = Image.open(image_path).convert('RGB')
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    title_suffix = f"Ground Truth: {ground_truth}"

    # Plot 1: YOLO
    ax1 = axes[0, 0]
    ax1.imshow(image)
    ax1.set_title(f"YOLO11\n{title_suffix}", fontsize=12, fontweight='bold')
    ax1.axis('off')

    if yolo_det:
        x1, y1, x2, y2 = yolo_det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor='green', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-10, f"✓ {yolo_det['class_name']} ({yolo_det['confidence']:.2f})",
                color='green', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax1.text(10, 30, "✗ Not detected", color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 2: OWLv2
    ax2 = axes[0, 1]
    ax2.imshow(image)
    ax2.set_title(f"OWLv2\n{title_suffix}", fontsize=12, fontweight='bold')
    ax2.axis('off')

    if owl_det:
        x1, y1, x2, y2 = owl_det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor='green', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-10, f"✓ {owl_det['query']} ({owl_det['confidence']:.2f})",
                color='green', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax2.text(10, 30, "✗ Not detected", color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 3: Grounding DINO
    ax3 = axes[1, 0]
    ax3.imshow(image)
    ax3.set_title(f"Grounding DINO\n{title_suffix}", fontsize=12, fontweight='bold')
    ax3.axis('off')

    if dino_det:
        x1, y1, x2, y2 = dino_det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=3, edgecolor='green', facecolor='none')
        ax3.add_patch(rect)
        ax3.text(x1, y1-10, f"✓ {dino_det['query']} ({dino_det['confidence']:.2f})",
                color='green', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax3.text(10, 30, "✗ Not detected", color='red', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Plot 4: Overlay + IoU info
    ax4 = axes[1, 1]
    ax4.imshow(image)

    if bbox_consistency:
        title = f"All Models\nAvg IoU: {bbox_consistency['avg_iou']:.3f}"
    else:
        title = "All Models\n(Not all detected)"
    ax4.set_title(title, fontsize=12, fontweight='bold')
    ax4.axis('off')

    if yolo_det:
        x1, y1, x2, y2 = yolo_det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='red', facecolor='none',
                                 linestyle='--', label='YOLO', alpha=0.7)
        ax4.add_patch(rect)

    if owl_det:
        x1, y1, x2, y2 = owl_det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='blue', facecolor='none',
                                 linestyle='--', label='OWL', alpha=0.7)
        ax4.add_patch(rect)

    if dino_det:
        x1, y1, x2, y2 = dino_det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='purple', facecolor='none',
                                 linestyle='--', label='DINO', alpha=0.7)
        ax4.add_patch(rect)

    if yolo_det or owl_det or dino_det:
        ax4.legend(loc='upper right', fontsize=10)

    if bbox_consistency:
        iou_text = f"IoU:\n"
        iou_text += f"Y-O: {bbox_consistency.get('iou_yolo_owl', 0):.3f}\n"
        iou_text += f"Y-D: {bbox_consistency.get('iou_yolo_dino', 0):.3f}\n"
        iou_text += f"O-D: {bbox_consistency.get('iou_owl_dino', 0):.3f}"
        ax4.text(10, 30, iou_text, color='black', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.tight_layout()

    img_name = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{img_name}_detection.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_generated_images(yolo_detector, owl_detector, dino_detector, vlm_detector=None,
                             validation_dir='outputs/validation_images',
                             output_base='outputs/three_model_consensus',
                             save_visualizations=True,
                             num_visualizations=10):
    print("Processing Generated Validation Images")

    generated_dir = os.path.join(output_base, 'generated')
    detections_dir = os.path.join(generated_dir, 'detections')
    visualizations_dir = os.path.join(generated_dir, 'visualizations')
    summary_dir = os.path.join(generated_dir, 'summary')

    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    style_mapping = {
        'Ink and wash painting': 'Chinese_painting',
        'Art Nouveau Modern': 'Art_Nouveau',
        'Post-Impressionism': 'Post_Impressionism'
    }

    for style in config.STYLES:
        print(f"[{style}]")

        folder_name = style_mapping.get(style, style)
        style_dir = os.path.join(validation_dir, folder_name)

        if not os.path.exists(style_dir):
            print(f"Directory not found: {style_dir}")
            continue

        image_files = list(Path(style_dir).glob('*.png')) + list(Path(style_dir).glob('*.jpg'))

        if not image_files:
            print(f"No images found")
            continue

        print(f"Processing {len(image_files)} images")

        style_det_dir = os.path.join(detections_dir, style)
        os.makedirs(style_det_dir, exist_ok=True)

        if save_visualizations:
            style_vis_dir = os.path.join(visualizations_dir, style)
            os.makedirs(style_vis_dir, exist_ok=True)

        all_results = []

        for idx, img_path in enumerate(tqdm(image_files, desc=f"Detecting {style}")):
            try:
                ground_truth = parse_filename(img_path.name)
                if not ground_truth:
                    continue

                yolo_dets = yolo_detector.detect(str(img_path))
                yolo_match = find_matching_detection(yolo_dets, ground_truth)

                vlm_result = None
                if len(yolo_dets) == 0 and vlm_detector and vlm_detector.available:
                    vlm_result = vlm_detector.detect_objects(str(img_path))
                    if vlm_result and vlm_result['success']:
                        vlm_result['gt_match'] = any(
                            ground_truth.lower() in obj.lower() or obj.lower() in ground_truth.lower()
                            for obj in vlm_result['objects']
                        )

                if len(yolo_dets) > 0:
                    queries = list(set([d['class_name'] for d in yolo_dets]))
                else:
                    queries = [ground_truth]

                owl_dets = owl_detector.detect(str(img_path), queries)
                owl_match = find_matching_detection(owl_dets, ground_truth)

                dino_dets = dino_detector.detect(str(img_path), queries)
                dino_match = find_matching_detection(dino_dets, ground_truth)

                bbox_consistency = None
                if yolo_match and owl_match and dino_match:
                    iou_yolo_owl = compute_iou(yolo_match['bbox'], owl_match['bbox'])
                    iou_yolo_dino = compute_iou(yolo_match['bbox'], dino_match['bbox'])
                    iou_owl_dino = compute_iou(owl_match['bbox'], dino_match['bbox'])

                    bbox_consistency = {
                        'iou_yolo_owl': iou_yolo_owl,
                        'iou_yolo_dino': iou_yolo_dino,
                        'iou_owl_dino': iou_owl_dino,
                        'avg_iou': np.mean([iou_yolo_owl, iou_yolo_dino, iou_owl_dino]),
                        'min_iou': min(iou_yolo_owl, iou_yolo_dino, iou_owl_dino)
                    }
                elif yolo_match and owl_match:
                    iou = compute_iou(yolo_match['bbox'], owl_match['bbox'])
                    bbox_consistency = {'iou_yolo_owl': iou, 'avg_iou': iou}
                elif yolo_match and dino_match:
                    iou = compute_iou(yolo_match['bbox'], dino_match['bbox'])
                    bbox_consistency = {'iou_yolo_dino': iou, 'avg_iou': iou}
                elif owl_match and dino_match:
                    iou = compute_iou(owl_match['bbox'], dino_match['bbox'])
                    bbox_consistency = {'iou_owl_dino': iou, 'avg_iou': iou}

                result = {
                    'image_path': str(img_path),
                    'image_name': img_path.name,
                    'ground_truth': ground_truth,
                    'all_detections': {
                        'yolo': yolo_dets,
                        'owl': owl_dets,
                        'dino': dino_dets
                    },
                    'matched_detections': {
                        'yolo': yolo_match,
                        'owl': owl_match,
                        'dino': dino_match
                    },
                    'detection_status': {
                        'yolo_detected': yolo_match is not None,
                        'owl_detected': owl_match is not None,
                        'dino_detected': dino_match is not None,
                        'all_3_detected': all([yolo_match, owl_match, dino_match])
                    },
                    'bbox_consistency': bbox_consistency,
                    'counts': {
                        'yolo_total': len(yolo_dets),
                        'owl_total': len(owl_dets),
                        'dino_total': len(dino_dets)
                    },
                    'issues': {
                        'yolo_zero_detection': len(yolo_dets) == 0,
                        'over_detection': (len(yolo_dets) > 1 or len(owl_dets) > 1 or len(dino_dets) > 1),
                        'all_over_detect': (len(yolo_dets) > 1 and len(owl_dets) > 1 and len(dino_dets) > 1)
                    },
                    'vlm_fallback': vlm_result if vlm_result else None,
                    'used_vlm': (vlm_result is not None and vlm_result.get('success', False))
                }

                all_results.append(result)

                if save_visualizations and idx < num_visualizations:
                    try:
                        visualize_single_object_detection(
                            str(img_path), ground_truth,
                            yolo_match, owl_match, dino_match,
                            bbox_consistency, style_vis_dir
                        )
                    except Exception as viz_error:
                        print(f"Visualization failed: {viz_error}")

            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue

        style_output = {
            'style': style,
            'total_images': len(all_results),
            'per_image_results': all_results
        }

        output_file = os.path.join(style_det_dir, f'{style}_generated_results.json')
        with open(output_file, 'w') as f:
            json.dump(style_output, f, indent=2)

        print(f"Saved {len(all_results)} results to {output_file}")

    print("Generated Images Processing Complete")
    print(f"Output saved to: {generated_dir}")


def main():
    print("Generated Images Single-Object Three-Model Detection")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading Models")

    yolo_detector = YOLO11Detector(model_size='x', device=device)
    owl_detector = OWLv2Detector(device=device)
    dino_detector = GroundingDINODetector(device=device)
    vlm_detector = VLMObjectDetector(device=device) if VLM_AVAILABLE else None

    if vlm_detector and vlm_detector.available:
        print("VLM fallback enabled")
    else:
        print("VLM fallback disabled")

    process_generated_images(
        yolo_detector=yolo_detector,
        owl_detector=owl_detector,
        dino_detector=dino_detector,
        vlm_detector=vlm_detector,
        save_visualizations=True,
        num_visualizations=10
    )


if __name__ == '__main__':
    main()
