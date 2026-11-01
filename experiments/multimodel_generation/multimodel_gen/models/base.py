"""
Base Model Interface
Abstract base class for all generative models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

class BaseModel(ABC):
    """Generative Model Base Class"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model
        
        Args:
            config: Model configuration dictionary, from config/models.json
        """
        self.config = config
        self.model_id = config['model_id']
        self.model_type = config['model_type']
        self.display_name = config['display_name']
        self.default_params = config['default_params']
        self.pipe = None
    
    @abstractmethod
    def load(self):
        """Load model into memory"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, params: Dict[str, Any]):
        """
        Generate image
        
        Args:
            prompt: Text prompt
            params: Generation parameters
            
        Returns:
            PIL.Image: Generated image
        """
        pass
    
    def unload(self):
        """Unload model to free memory"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            torch.cuda.empty_cache()
    
    def get_params(self, **overrides) -> Dict[str, Any]:
        """
        Get generation parameters, supporting overrides
        
        Args:
            **overrides: Parameters to override
            
        Returns:
            Merged parameter dictionary
        """
        params = self.default_params.copy()
        params.update(overrides)
        return params
