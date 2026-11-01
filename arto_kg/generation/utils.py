"""Logging, GPU memory, image sanity-check and generation-info helpers."""

import os
import json
import torch
import logging

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging system"""
    # Configure log format
    log_format = "[%(asctime)s] %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(),  # Console output
        ]
    )
    
    # If log file specified, add file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        
        logging.getLogger().addHandler(file_handler)
    
    logging.info(f"Logging setup complete. Level: {log_level}")


def cleanup_memory():
    """Clean up GPU and CPU memory"""
    import gc

    # Force garbage collection
    gc.collect()

    if torch.cuda.is_available():
        # Clean cache for each GPU device
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

        # Record memory usage after cleanup
        total_allocated = 0
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total_allocated += allocated

        logging.debug(f"Memory cleanup completed. Total GPU memory: {total_allocated:.1f}GB")


def get_gpu_memory_info() -> Dict[str, Any]:
    """Get GPU memory usage info"""
    if not torch.cuda.is_available():
        return {'available': False}
    
    memory_info = {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            'device_id': i,
            'name': torch.cuda.get_device_name(i),
            'allocated_gb': torch.cuda.memory_allocated(i) / 1024**3,
            'reserved_gb': torch.cuda.memory_reserved(i) / 1024**3,
            'max_memory_gb': torch.cuda.max_memory_allocated(i) / 1024**3
        }
        memory_info['devices'].append(device_info)
    
    return memory_info


def is_complete_image(path) -> bool:
    """Whether the file at path decodes to a whole, non-empty image.

    A truncated or damaged file fails to load and is reported as incomplete.
    """
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except Exception:
        return False


def looks_like_noise(image) -> Dict[str, Any]:
    """Cheap check that a generated image is a picture and not noise.

    Uses mean adjacent-pixel difference, which separates noise from even highly
    textured paintings. Returns a report dict for the info JSON.
    """
    try:
        import numpy as np

        pixels = np.asarray(image.convert("L"), dtype=np.float32)
        if pixels.size < 4:
            return {"checked": False, "reason": "image too small"}

        # Mean absolute difference between horizontally adjacent pixels, in 0-1.
        neighbour_diff = float(np.abs(np.diff(pixels, axis=1)).mean()) / 255.0

        return {
            "checked": True,
            "neighbour_difference": round(neighbour_diff, 4),
            "threshold": NOISE_NEIGHBOUR_THRESHOLD,
            "is_noise": neighbour_diff > NOISE_NEIGHBOUR_THRESHOLD,
        }
    except Exception as exc:  # never let the check break generation
        return {"checked": False, "reason": str(exc)}


# Adjacent-pixel difference above which an image is treated as noise. Uniform
# random pixels average about 0.33; real images are far lower.
NOISE_NEIGHBOUR_THRESHOLD = 0.15


def save_generation_info(processed_data: Dict[str, Any],
                        output_path: Path,
                        info_path: Path,
                        model_id: Optional[str] = None,
                        substitute_model: bool = False,
                        image_check: Optional[Dict[str, Any]] = None):
    """Save generation info to a JSON file.

    Records the checkpoint (`model_id`) and whether it differs from the release
    model (`substitute_model`).
    """
    generation_info = {
        'artwork_id': processed_data['artwork_id'],
        'output_image': str(output_path),
        'generation_time': datetime.now().isoformat(),
        'model_id': model_id,
        'substitute_model': substitute_model,
        # Result of looks_like_noise.
        'image_check': image_check or {},
        'final_prompt': processed_data['final_prompt'],
        'style_info': processed_data['style_info'],
        'generation_params': processed_data['generation_params'],
        'gpu_memory': get_gpu_memory_info(),
        'prompts_used': processed_data['prompts'],
        'source_file': processed_data['source_data'].get('_source_file', 'unknown')
    }
    
    try:
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save generation info to {info_path}: {e}")
        logging.error(f"Error details: {type(e).__name__}: {str(e)}")

