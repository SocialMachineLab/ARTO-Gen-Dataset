"""
vLLM wrapper for VLM inference, with optional multi-GPU tensor parallelism.
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

# Optional JSON-schema guided decoding; without it output is unconstrained.
try:
    from vllm.sampling_params import GuidedDecodingParams
    GUIDED_DECODING_AVAILABLE = True
except ImportError:
    logger.warning("vllm.sampling_params.GuidedDecodingParams not available: "
                   "structured output will be requested in the prompt only, "
                   "which the model is free to ignore.")
    GUIDED_DECODING_AVAILABLE = False

# The VLM used by every VLM-based validation step.
DEFAULT_VLM = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

# Environment variable that overrides the model path; such runs are flagged
# via `is_substitute`.
VLM_ENV_VAR = "VALIDATION_VLM"


def resolve_vlm_path(model_path: Optional[str] = None) -> str:
    """Release model unless explicitly overridden, argument beating environment."""
    return model_path or os.getenv(VLM_ENV_VAR) or DEFAULT_VLM


def resolve_dtype(model_path: str, dtype: Optional[str] = None) -> str:
    """Weight dtype for a model path.

    float16 for the release AWQ checkpoint, "auto" for any other model; an
    explicit argument always wins.
    """
    if dtype:
        return dtype
    return "float16" if model_path == DEFAULT_VLM else "auto"


class VLLMWrapper:
    """vLLM VLM Inference Wrapper"""

    def __init__(self,
                 model_path: Optional[str] = None,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.9,
                 dtype: Optional[str] = None):
        """
        Initialize vLLM model

        Args:
            model_path: Model path. Defaults to DEFAULT_VLM, or to $VALIDATION_VLM
                        when that is set. Anything other than DEFAULT_VLM marks
                        the wrapper as a substitution.
            tensor_parallel_size: Tensor parallel size (Number of GPUs)
            gpu_memory_utilization: GPU memory utilization
            dtype: Weight dtype. None selects it via resolve_dtype.
        """
        self.model_path = resolve_vlm_path(model_path)
        self.is_substitute = self.model_path != DEFAULT_VLM
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.dtype = resolve_dtype(self.model_path, dtype)
        self.llm = None
        self.is_initialized = False

        if VLLM_AVAILABLE:
            self.initialize_model()

    def model_info(self) -> Dict[str, Any]:
        """Model provenance to record alongside validation results."""
        return {
            "vlm_model": self.model_path,
            "substitute_model": self.is_substitute,
            "release_model": DEFAULT_VLM,
            "dtype": self.dtype,
        }

    def initialize_model(self):
        """Initialize vLLM model"""
        try:
            logger.info(f"Loading {self.model_path} with vLLM...")
            if self.is_substitute:
                logger.warning("=" * 68)
                logger.warning(f"SUBSTITUTE VLM: {self.model_path}")
                logger.warning(f"The release model is {DEFAULT_VLM}.")
                logger.warning("Scores from this run are NOT comparable to results produced")
                logger.warning("with the release model. Every result carries")
                logger.warning("substitute_model=true.")
                logger.warning("=" * 68)
            logger.info(f"   Tensor parallel size: {self.tensor_parallel_size}")
            logger.info(f"   GPU memory utilization: {self.gpu_memory_utilization}")

            vllm_config = {
                "model": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "trust_remote_code": True,
                "max_model_len": 4096,
                "max_num_seqs": 2,  # Limit concurrent sequences
                "enforce_eager": True,  # Disable CUDA graphs
                "disable_custom_all_reduce": True,
                "dtype": self.dtype,
                # Where local images may be read from; real paths on both sides, so
                # symbolic links in either cannot make an image fall outside it.
                "allowed_local_media_path": os.path.realpath(os.getenv("VLLM_MEDIA_PATH", os.getcwd())),
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
                         max_tokens: int = 512,
                         json_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Single sample inference

        Args:
            prompt: Text prompt
            image_path: Image path
            max_tokens: Max generated tokens
            json_schema: Optional JSON schema to constrain decoding to.

        Returns:
            Response dictionary; `finish_reason` is 'length' when the reply was
            cut off at max_tokens.
        """
        if not self.is_initialized:
            return {
                'response': 'Model not initialized',
                'success': False,
                'error': 'Model not initialized'
            }
        
        try:
            abs_path = os.path.realpath(image_path)
            
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
            guided = None
            if json_schema is not None and GUIDED_DECODING_AVAILABLE:
                guided = GuidedDecodingParams(json=json_schema)
            sampling_params = SamplingParams(
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.9,
                guided_decoding=guided,
            )

            # Generate
            outputs = self.llm.chat(
                messages=[messages],
                sampling_params=sampling_params,
            )

            # Extract result
            completion = outputs[0].outputs[0]
            output_text = completion.text

            finish_reason = getattr(completion, 'finish_reason', None)
            if finish_reason == 'length':
                logger.warning(
                    f"Reply hit the {max_tokens}-token budget and was cut off. "
                    "A truncated reply is not a short one: structured output "
                    "will be incomplete and will not parse.")

            return {
                'response': output_text,
                'success': True,
                'model_used': self.model_path,
                'substitute_model': self.is_substitute,
                # 'length' means the token budget ran out, 'stop' means the model finished.
                'finish_reason': finish_reason,
                'generated_tokens': len(getattr(completion, 'token_ids', []) or []),
                'guided_json': guided is not None,
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
        Batch inference

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
                abs_path = os.path.realpath(img_path)
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
                    'model_used': self.model_path
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
                       gpu_memory_utilization: float = 0.9,
                       dtype: Optional[str] = None) -> VLLMWrapper:
    """
    Create vLLM wrapper
    
    Args:
        model_path: Model path. None means DEFAULT_VLM, or $VALIDATION_VLM if set.
        tensor_parallel_size: Number of GPUs
        gpu_memory_utilization: GPU memory utilization

    Returns:
        VLLMWrapper instance
    """
    return VLLMWrapper(model_path, tensor_parallel_size, gpu_memory_utilization, dtype)


__all__ = ['VLLMWrapper', 'create_vllm_wrapper', 'resolve_vlm_path', 'resolve_dtype',
           'DEFAULT_VLM', 'VLM_ENV_VAR']
