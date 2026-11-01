"""
Stable Diffusion 3.5 Large Model Implementation
"""

import torch
from typing import Dict, Any
from diffusers import StableDiffusion3Pipeline

from .base import BaseModel


class SD35Model(BaseModel):
    """SD 3.5 Large Model Implementation"""
    
    def load(self):
        """Load SD3.5 model"""
        print(f"[INFO] Loading {self.display_name}...")
        
        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16
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
                guidance_scale=params.get('guidance_scale', 4.5),
                generator=generator,
            ).images[0]
        
        return image
