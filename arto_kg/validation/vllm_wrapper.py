"""
vLLM VLM Inference Wrapper
Use vLLM for high-performance VLM inference, auto-support multi-GPU
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from vllm import LLM, SamplingParams
    from vllm.multimodal.utils import fetch_image
    VLLM_AVAILABLE = True
    logger.info("vLLM is available")
except ImportError as e:
    logger.warning(f"vLLM not available: {e}")
    VLLM_AVAILABLE = False


class VLLMWrapper:
    """vLLM VLM Inference Wrapper"""
    
    def __init__(self, 
                 model_path: str = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9):
        """
        Initialize vLLM model
        
        Args:
            model_path: Model path
            tensor_parallel_size: Tensor parallel size (Number of GPUs)
            gpu_memory_utilization: GPU memory utilization
        """
        self.model_path = model_path
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.llm = None
        self.is_initialized = False
        
        if VLLM_AVAILABLE:
            self.initialize_model()
    
    def initialize_model(self):
        """Initialize vLLM model"""
        try:
            logger.info(f"Loading {self.model_path} with vLLM...")
            logger.info(f"   Tensor parallel size: {self.tensor_parallel_size}")
            logger.info(f"   GPU memory utilization: {self.gpu_memory_utilization}")
            
            # Use same config as src/vllm_wrapper.py
            vllm_config = {
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
                "max_model_len": 4096,  # Consistent with VALIDATION_VLM_CONFIG
                "max_num_seqs": 2,  # Limit concurrent sequences
                "enforce_eager": True,  # Critical: Disable CUDA graphs
                "disable_custom_all_reduce": True,  # Critical: Disable custom all-reduce
                "dtype": "float16",
                "allowed_local_media_path": os.getenv("VLLM_MEDIA_PATH", os.getcwd()),  # Allow access to project root directory
            }
            
            logger.info(f"vLLM configuration: {vllm_config}")
            
            # Initialize vLLM
            self.llm = LLM(**vllm_config)
            
            self.is_initialized = True
            logger.info(f"vLLM model loaded successfully")
            logger.info(f"Using {self.tensor_parallel_size} GPU(s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            import traceback
            traceback.print_exc()
            self.is_initialized = False
            return False
    
    def generate_response(self, 
                         prompt: str, 
                         image_path: str, 
                         max_tokens: int = 512) -> Dict[str, Any]:
        """
        Single sample inference
        
        Args:
            prompt: Text prompt
            image_path: Image path
            max_tokens: Max generated tokens
            
        Returns:
            Response dictionary
        """
        if not self.is_initialized:
            return {
                'response': 'Model not initialized',
                'success': False,
                'error': 'Model not initialized'
            }
        
        try:
            abs_path = os.path.abspath(image_path)
            
            # Construct message - Use image_url format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"file://{abs_path}"
                            }
                        }
                    ]
                }
            ]
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            # Generate
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=sampling_params,
            )
            
            # Extract result
            output_text = outputs[0].outputs[0].text
            
            return {
                'response': output_text,
                'success': True,
                'model_used': 'vLLM-Qwen2.5-VL-32B-AWQ'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': f'Generation failed: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def generate_batch(self,
                      prompts: List[str],
                      image_paths: List[str],
                      max_tokens: int = 512) -> List[Dict[str, Any]]:
        """
        Batch inference - Core advantage of vLLM
        
        Args:
            prompts: List of prompts
            image_paths: List of image paths
            max_tokens: Max generated tokens
            
        Returns:
            List of responses
        """
        if not self.is_initialized:
            return [{'response': 'Model not initialized', 'success': False} 
                   for _ in prompts]
        
        if len(prompts) != len(image_paths):
            raise ValueError("prompts and image_paths must have same length")
        
        try:
            # Construct batch messages - Use image_url format (vLLM standard format)
            batch_messages = []
            for prompt, img_path in zip(prompts, image_paths):
                abs_path = os.path.abspath(img_path)
                batch_messages.append([
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"file://{abs_path}"
                                }
                            }
                        ]
                    }
                ])
            
            # Sampling parameters
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            
            # Batch generation - Use chat API
            outputs = self.llm.chat(
                messages=batch_messages,
                sampling_params=sampling_params,
            )
            
            # Extract result
            results = []
            for i, output in enumerate(outputs):
                output_text = output.outputs[0].text
                results.append({
                    'response': output_text,
                    'success': True,
                    'batch_index': i,
                    'model_used': 'vLLM-Qwen2.5-VL-32B-AWQ-Batch'
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return [{'response': f'Batch failed: {str(e)}', 
                    'success': False, 
                    'error': str(e)} 
                   for _ in prompts]


def create_vllm_wrapper(model_path: Optional[str] = None,
                       tensor_parallel_size: int = 1,
                       gpu_memory_utilization: float = 0.9) -> VLLMWrapper:
    """
    Create vLLM wrapper
    
    Args:
        model_path: Model path
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: GPU memory utilization
        
    Returns:
        VLLMWrapper instance
    """
    if model_path is None:
        model_path = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"
    
    return VLLMWrapper(model_path, tensor_parallel_size, gpu_memory_utilization)


__all__ = ['VLLMWrapper', 'create_vllm_wrapper']
