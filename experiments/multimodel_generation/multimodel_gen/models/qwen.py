"""
Qwen-Image Model Implementation
Qwen implementation based on arto_kg/generation/main.py
"""

import os
import sys
import math
import torch
from pathlib import Path
from typing import Dict, Any
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

from .base import BaseModel

# Add arto_kg to path to reuse code
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arto_kg.generation.prompt_processor import PromptProcessor
from arto_kg.generation.style_handler import StyleHandler
from arto_kg.generation.utils import cleanup_memory


def build_scheduler():
    """Build Qwen specific scheduler"""
    cfg = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    return FlowMatchEulerDiscreteScheduler.from_config(cfg)


class QwenModel(BaseModel):
    """Qwen-Image Model Implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.prompt_processor = PromptProcessor()
        self.style_handler = StyleHandler()
    
    def load(self):
        """Load Qwen-Image model"""
        print(f"[INFO] Loading {self.display_name}...")
        
        scheduler = build_scheduler()
        
        self.pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            scheduler=scheduler,
            torch_dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True,
        )
        
        # Enable memory optimizations
        print("[INFO] Enabling memory optimizations...")
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        
        try:
            self.pipe.enable_attention_slicing(1)
            print("[INFO] Attention slicing enabled")
        except:
            print("[INFO] Attention slicing not available")
        
        print(f"[INFO] {self.display_name} loaded successfully!")
    
    def generate(self, prompt: str, params: Dict[str, Any]):
        """
        Generate image
        
        Args:
            prompt: Text prompt
            params: Generation parameters, should include:
                - width: Image width
                - height: Image height
                - num_inference_steps: Inference steps
                - true_cfg_scale: CFG scale
                - seed: Random seed
        
        Returns:
            PIL.Image: Generated image
        """
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        cleanup_memory()
        
        with torch.inference_mode():
            generator = torch.Generator().manual_seed(params.get('seed', 42))
            
            images = self.pipe(
                prompt=prompt,
                width=params.get('width', 1024),
                height=params.get('height', 1024),
                num_inference_steps=params.get('num_inference_steps', 80),
                true_cfg_scale=params.get('true_cfg_scale', 3.5),
                generator=generator,
            ).images
        
        cleanup_memory()
        return images[0]
    
    def process_artwork_data(self, artwork_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process artwork data, extract prompts and parameters
        
        Args:
            artwork_data: Artwork JSON data
            
        Returns:
            Dictionary containing prompt and params
        """
        # Extract prompts
        prompts = self.prompt_processor.extract_prompts(artwork_data)
        
        # Extract style info
        style_info = self.style_handler.extract_style_info(artwork_data)
        
        # Build final prompt
        final_prompt = self.prompt_processor.build_final_prompt(
            prompts, style_info, artwork_data
        )
        
        # Extract generation parameters
        params = self.get_params()
        
        # Adaptive CFG scale (based on object count)
        if self.config.get('adaptive_cfg', {}).get('enabled', False):
            obj_count = len(artwork_data.get('objects', {}).get('object_names', []))
            thresholds = self.config['adaptive_cfg']['obj_count_thresholds']
            
            if obj_count >= 9:
                params['true_cfg_scale'] = thresholds['9+']
            elif obj_count >= 5:
                params['true_cfg_scale'] = thresholds['5-8']
            else:
                params['true_cfg_scale'] = thresholds['1-4']
        
        return {
            'prompt': final_prompt,
            'params': params,
            'artwork_id': artwork_data.get('artwork_id', 'unknown')
        }
