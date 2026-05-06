"""Detect objects in WikiArt images using an ensemble of YOLO, OWLv2, Grounding DINO, and VLM"""

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


def visualize_wikiart_detection(image_path, yolo_dets, owl_dets, dino_dets, output_dir):
    image = Image.open(image_path).convert('RGB')
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Plot 1: YOLO
    ax1 = axes[0, 0]
    ax1.imshow(image)
    ax1.set_title(f"YOLO11 ({len(yolo_dets)} objects)", fontsize=14, fontweight='bold')
    ax1.axis('off')
    for det in yolo_dets:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax1.add_patch(rect)
        ax1.text(x1, y1-5, f"{det['class_name']} {det['confidence']:.2f}",
                color='red', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot 2: OWLv2
    ax2 = axes[0, 1]
    ax2.imshow(image)
    ax2.set_title(f"OWLv2 ({len(owl_dets)} objects)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    for det in owl_dets:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='blue', facecolor='none')
        ax2.add_patch(rect)
        ax2.text(x1, y1-5, f"{det['query']} {det['confidence']:.2f}",
                color='blue', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot 3: Grounding DINO
    ax3 = axes[1, 0]
    ax3.imshow(image)
    ax3.set_title(f"Grounding DINO ({len(dino_dets)} objects)", fontsize=14, fontweight='bold')
    ax3.axis('off')
    for det in dino_dets:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor='purple', facecolor='none')
        ax3.add_patch(rect)
        ax3.text(x1, y1-5, f"{det['query']} {det['confidence']:.2f}",
                color='purple', fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Plot 4: All together
    ax4 = axes[1, 1]
    ax4.imshow(image)
    ax4.set_title(f"All Models (Y:{len(yolo_dets)}, O:{len(owl_dets)}, D:{len(dino_dets)})",
                 fontsize=14, fontweight='bold')
    ax4.axis('off')

    for det in yolo_dets:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1, edgecolor='red', facecolor='none', alpha=0.6)
        ax4.add_patch(rect)

    for det in owl_dets:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1, edgecolor='blue', facecolor='none', alpha=0.6)
        ax4.add_patch(rect)

    for det in dino_dets:
        x1, y1, x2, y2 = det['bbox']
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                 linewidth=1, edgecolor='purple', facecolor='none', alpha=0.6)
        ax4.add_patch(rect)

    plt.tight_layout()

    img_name = Path(image_path).stem
    save_path = os.path.join(output_dir, f"{img_name}_detection.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def load_sampled_images(metadata_path='outputs/metadata/sampled_images.json'):
    """Load WikiArt sampled images"""
    if not os.path.exists(metadata_path):
        print(f"{metadata_path} not found")
        return None

    with open(metadata_path, 'r') as f:
        data = json.load(f)

    if 'sampled_images' in data:
        return data['sampled_images']
    return data


def process_wikiart_images(yolo_detector, owl_detector, dino_detector, vlm_detector=None,
                           output_base='outputs/three_model_consensus',
                           save_visualizations=True,
                           num_visualizations=10):
    print("Processing WikiArt Images")

    sampled_images = load_sampled_images()
    if not sampled_images:
        print("Cannot load sampled images")
        return

    print(f"Loaded {len(sampled_images)} styles")

    wikiart_dir = os.path.join(output_base, 'wikiart')
    detections_dir = os.path.join(wikiart_dir, 'detections')
    visualizations_dir = os.path.join(wikiart_dir, 'visualizations')
    summary_dir = os.path.join(wikiart_dir, 'summary')

    os.makedirs(detections_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    for style in config.STYLES:
        if style not in sampled_images:
            print(f"Skipping {style}: No sampled images found")
            continue

        print(f"Processing style: {style}")

        image_paths = sampled_images[style]
        image_paths = [p for p in image_paths if os.path.exists(p)]

        if not image_paths:
            print(f"No valid image files found for {style}")
            continue

        style_det_dir = os.path.join(detections_dir, style)
        os.makedirs(style_det_dir, exist_ok=True)

        if save_visualizations:
            style_vis_dir = os.path.join(visualizations_dir, style)
            os.makedirs(style_vis_dir, exist_ok=True)

        all_results = []

        for idx, img_path in enumerate(tqdm(image_paths, desc=f"Detecting {style}")):
            try:
                yolo_dets = yolo_detector.detect(img_path)

                vlm_result = None
                if len(yolo_dets) == 0 and vlm_detector and vlm_detector.available:
                    vlm_result = vlm_detector.detect_objects(img_path)

                if len(yolo_dets) > 0:
                    query_classes = list(set([d['class_name'] for d in yolo_dets]))
                elif vlm_result and vlm_result['success']:
                    query_classes = vlm_result['objects']
                else:
                    query_classes = []

                owl_dets = owl_detector.detect(img_path, query_classes)
                dino_dets = dino_detector.detect(img_path, query_classes)

                result = {
                    'image_path': img_path,
                    'image_name': os.path.basename(img_path),
                    'detections': {
                        'yolo': yolo_dets,
                        'owl': owl_dets,
                        'dino': dino_dets
                    },
                    'counts': {
                        'yolo': len(yolo_dets),
                        'owl': len(owl_dets),
                        'dino': len(dino_dets)
                    },
                    'vlm_fallback': vlm_result if vlm_result else None,
                    'used_vlm': (vlm_result is not None and vlm_result['success'])
                }

                all_results.append(result)

                if save_visualizations and idx < num_visualizations:
                    try:
                        visualize_wikiart_detection(
                            img_path, yolo_dets, owl_dets, dino_dets,
                            style_vis_dir
                        )
                    except Exception as viz_error:
                        print(f"Visualization failed: {viz_error}")

            except Exception as e:
                print(f"Error processing {os.path.basename(img_path)}: {e}")
                continue

        style_output = {
            'style': style,
            'total_images': len(all_results),
            'per_image_results': all_results
        }

        output_file = os.path.join(style_det_dir, f'{style}_wikiart_results.json')
        with open(output_file, 'w') as f:
            json.dump(style_output, f, indent=2)

        print(f"Saved {len(all_results)} results to {output_file}")


def main():
    print("WikiArt Three-Model Consensus Detection")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    yolo_detector = YOLO11Detector(model_size='x', device=device)
    owl_detector = OWLv2Detector(device=device)
    dino_detector = GroundingDINODetector(device=device)
    vlm_detector = VLMObjectDetector(device=device) if VLM_AVAILABLE else None

    if vlm_detector and vlm_detector.available:
        print("VLM fallback enabled")
    else:
        print("VLM fallback disabled")

    process_wikiart_images(
        yolo_detector=yolo_detector,
        owl_detector=owl_detector,
        dino_detector=dino_detector,
        vlm_detector=vlm_detector,
        save_visualizations=True,
        num_visualizations=10
    )


if __name__ == '__main__':
    main()
