"""
Stable Diffusion XL Base Model Implementation
"""

import torch
from typing import Dict, Any
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler

from .base import BaseModel


class SDXLModel(BaseModel):
    """SDXL Base 1.0 Model Implementation"""
    
    def load(self):
        """Load SDXL model"""
        print(f"[INFO] Loading {self.display_name}...")
        
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16"
        )
        self.pipe.to("cuda")
        
        print(f"[INFO] {self.display_name} loaded successfully!")
    
    def generate(self, prompt: str, params: Dict[str, Any]):
        """Generate image"""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        with torch.inference_mode():
            generator = torch.Generator("cuda").manual_seed(params.get('seed', 42))
            
            image = self.pipe(
                prompt=prompt,
                height=params.get('height', 1024),
                width=params.get('width', 1024),
                num_inference_steps=params.get('num_inference_steps', 50),
                guidance_scale=params.get('guidance_scale', 7.5),
                generator=generator,
            ).images[0]
        
        return image
