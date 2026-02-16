import os
from typing import List, Dict, Any, Optional, Union
import torch

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None
from arto_kg.conceptualization.utils import setup_logger, extract_response_content, parse_json_response
from arto_kg.config.model_config import GENERATION_LLM_CONFIG, FALLBACK_LLM_CONFIG, GENERATION_CONFIG, PROMPT_TEMPLATES
from .output_parsers.unified_parser import UnifiedOutputParser


class VLLMWrapper:
    """vLLM wrapper"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, use_fallback: bool = False):
        self.logger = setup_logger("vllm_wrapper")
        
        # Select configuration
        if config is not None:
            self.config = config
        elif use_fallback:
            self.config = FALLBACK_LLM_CONFIG
            self.logger.info("Using fallback model configuration")
        else:
            self.config = GENERATION_LLM_CONFIG

        self.model = None
        self.sampling_params = None
        self.output_parser = UnifiedOutputParser()
        self.current_model_name = None
        if VLLM_AVAILABLE:
            self._setup_sampling_params()
        
    def _setup_sampling_params(self) -> None:
        """Set sampling parameters"""
        self.sampling_params = SamplingParams(
            temperature=GENERATION_CONFIG["temperature"],
            top_p=GENERATION_CONFIG["top_p"],
            max_tokens=GENERATION_CONFIG["max_tokens"],
            stop=GENERATION_CONFIG["stop_tokens"]
        )
        
    def load_model(self) -> bool:
        """Load vLLM model"""
        if not VLLM_AVAILABLE:
            self.logger.error("vLLM is not installed or import failed")
            return False

        try:
            self.logger.info(f"Loading model: {self.config['model']}")
            
            # Set PyTorch memory management
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            
            # Check GPU status
            self._check_gpu_status()
            
            # Auto-adjust configuration based on available GPUs
            self._adjust_config_for_gpu()
            
            self.logger.info(self.config)
            # Load model
            self.model = LLM(**self.config)
            
            # Record current model name
            self.current_model_name = self.config.get('model', 'unknown')
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def _check_gpu_status(self) -> None:
        """Check GPU status"""
        self.logger.info("=== GPU Status Check ===")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"GPU count: {torch.cuda.device_count()}")
            self.logger.info(f"Current GPU: {torch.cuda.current_device()}")
            self.logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.logger.info(f"GPU memory: {gpu_memory:.1f} GB")
            self.logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            self.logger.warning("CUDA not available!")

    def _adjust_config_for_gpu(self) -> None:
        """Auto-adjust configuration based on available GPUs"""
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, keeping original config")
            return
            
        gpu_count = torch.cuda.device_count()
        
        # Auto-adjust tensor_parallel_size
        if 'tensor_parallel_size' in self.config:
            requested_gpus = self.config['tensor_parallel_size']
            if requested_gpus > gpu_count:
                self.logger.warning(f"Requested {requested_gpus} GPUs but only {gpu_count} available")
                self.config['tensor_parallel_size'] = gpu_count
                self.logger.info(f"✅ Adjusted tensor_parallel_size: {requested_gpus} → {gpu_count}")
            else:
                self.logger.info(f"✅ GPU configuration OK: using {requested_gpus}/{gpu_count} GPUs")
        
        # If only 1 GPU, remove distributed settings that might cause issues
        if gpu_count == 1 and self.config.get('tensor_parallel_size', 1) == 1:
            if 'disable_custom_all_reduce' in self.config:
                # This setting may not be needed for single GPU
                self.logger.info("✅ Single GPU detected, keeping disable_custom_all_reduce setting")
            
    def format_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """
        Format system and user prompts into chat template
        Replace ollama chat format in original code
        """
        return PROMPT_TEMPLATES["chat_template"].format(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )
    
    def generate_single(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Single prompt generation
        Replace chat() call in original code
        """
        if self.model is None:
            return {"error": "Model not loaded"}
            
        try:
            # Format prompt
            formatted_prompt = self.format_chat_prompt(system_prompt, user_prompt)
            print(formatted_prompt)
            # Generate response
            outputs = self.model.generate([formatted_prompt], self.sampling_params)
            
            if not outputs or not outputs[0].outputs:
                return {"error": "No output generated"}
                
            # Extract generated text
            generated_text = outputs[0].outputs[0].text.strip()
            
            # Parse output using unified parser
            parsed_result = self.output_parser.parse(generated_text, self.current_model_name)
            
            # Merge raw result and parsed result
            result = {
                "success": True,
                "content": generated_text,
                "raw_output": outputs[0],
                **parsed_result
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {"error": str(e)}
    
    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch generation
        Core feature of vLLM
        """
        if self.model is None:
            return [{"error": "Model not loaded"}] * len(prompts)
            
        try:
            self.logger.info(f"Starting batch generation for {len(prompts)} prompts")
            
            # Batch generation
            outputs = self.model.generate(prompts, self.sampling_params)
            
            # Process results
            results = []
            for i, output in enumerate(outputs):
                if output.outputs:
                    generated_text = output.outputs[0].text.strip()
                    
                    # Parse output using unified parser
                    parsed_result = self.output_parser.parse(generated_text, self.current_model_name)
                    
                    # Merge results
                    result = {
                        "success": True,
                        "content": generated_text,
                        "index": i,
                        "raw_output": output,
                        **parsed_result
                    }
                    results.append(result)
                else:
                    results.append({
                        "error": "No output generated",
                        "index": i
                    })
            
            self.logger.info(f"Batch generation completed: {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch generation error: {e}")
            return [{"error": str(e), "index": i} for i in range(len(prompts))]
    
    def generate_json_response(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Generate JSON response and parse
        Reuse JSON parsing logic from original code
        """
        # Generate response
        result = self.generate_single(system_prompt, user_prompt)
        
        if "error" in result:
            return {
                "error": result["error"],
                "raw_output": "",
                "parsing_details": {"status": "generation_failed"}
            }
            
        # result now contains results from unified parser
        enhanced_result = {
            "raw_output": result.get("raw_output", result["content"]),
            "parsing_details": result.get("parsing_details", {})
        }
        
        # If parsed JSON data exists, use it
        if result.get("json_data"):
            enhanced_result.update(result["json_data"])
        elif result.get("final"):
            # Try to parse JSON from final content
            try:
                import json as json_lib
                json_data = json_lib.loads(result["final"])
                enhanced_result.update(json_data)
            except:
                # If unified parser also fails, fallback to old parsing method
                json_data = parse_json_response(
                    result["content"], 
                    PROMPT_TEMPLATES["json_parsing_patterns"]
                )
                if "error" not in json_data:
                    enhanced_result.update(json_data)
                else:
                    enhanced_result["error"] = json_data["error"]
        else:
            enhanced_result["error"] = "No JSON data found"
            
        return enhanced_result
    
    def generate_json_response_legacy(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """
        Legacy method for generating JSON response (for backward compatibility)
        """
        # Generate response
        result = self.generate_single(system_prompt, user_prompt)
        
        if "error" in result:
            return result
            
        # Parse JSON
        json_data = parse_json_response(
            result["content"], 
            PROMPT_TEMPLATES["json_parsing_patterns"]
        )
        
        return json_data
    
    def generate_batch_json_responses(self, prompt_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """
        Batch generate JSON responses
        prompt_pairs: [(system_prompt, user_prompt), ...]
        """
        # Format all prompts
        formatted_prompts = [
            self.format_chat_prompt(system_prompt, user_prompt)
            for system_prompt, user_prompt in prompt_pairs
        ]
        
        # Batch generation
        batch_results = self.generate_batch(formatted_prompts)
        
        # Process results - batch_results now contains parsed results
        json_results = []
        for i, result in enumerate(batch_results):
            if "error" in result:
                enhanced_result = {
                    "error": result["error"],
                    "raw_output": "",
                    "parsing_details": {"status": "generation_failed"},
                    "index": i
                }
                json_results.append(enhanced_result)
            else:
                # result now contains results from unified parser
                # Keep original interface format, while adding new parsing info
                enhanced_result = {
                    "raw_output": result.get("raw_output", result["content"]),
                    "parsing_details": result.get("parsing_details", {
                        "status": "success" if result.get("json_data") else "failed"
                    }),
                    "index": i
                }
                
                # If parsed JSON data exists, add to result
                if result.get("json_data"):
                    enhanced_result.update(result["json_data"])
                elif result.get("final"):
                    # Try to parse JSON from final content
                    try:
                        import json as json_lib
                        json_data = json_lib.loads(result["final"])
                        enhanced_result.update(json_data)
                    except:
                        enhanced_result["error"] = "JSON parsing failed"
                else:
                    enhanced_result["error"] = "No JSON data found"
                    
                json_results.append(enhanced_result)
                
        return json_results
    
    def test_model_response(self) -> Dict[str, Any]:
        """
        Test model response
        Reuse debugging functionality from original code
        """
        test_system = "You are a helpful assistant."
        test_user = 'Say hello in JSON format: {"greeting": "hello"}'
        
        self.logger.info("Testing model response...")
        result = self.generate_json_response(test_system, test_user)
        
        self.logger.info(f"Test result: {result}")
        return result
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model info"""
        return {
            "model": self.config.get("model", "Unknown"),
            "max_model_len": self.config.get("max_model_len", 0),
            "gpu_memory_utilization": self.config.get("gpu_memory_utilization", 0),
            "is_loaded": self.model is not None,
            "sampling_params": {
                "temperature": self.sampling_params.temperature,
                "top_p": self.sampling_params.top_p,
                "max_tokens": self.sampling_params.max_tokens,
                "stop_tokens": self.sampling_params.stop
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources"""
        if self.model is not None:
            del self.model
            self.model = None
            torch.cuda.empty_cache()
            self.logger.info("Model resources cleaned up")


class VLLMManager:
    """vLLM manager for handling model initialization and fallback"""
    
    def __init__(self):
        self.logger = setup_logger("vllm_manager")
        self.primary_wrapper = None
        self.fallback_wrapper = None
        self.current_wrapper = None
        
    def initialize(self) -> bool:
        """Initialize the model with automatic fallback support"""
        try:
            # first try to load the primary model
            self.logger.info("Attempting to load primary model...")
            self.primary_wrapper = VLLMWrapper(use_fallback=False)
            
            if self.primary_wrapper.load_model():
                self.current_wrapper = self.primary_wrapper
                self.logger.info("Primary model loaded successfully")
                return True
            else:
                self.logger.warning("Primary model failed, trying fallback...")
                
        except Exception as e:
            self.logger.error(f"Primary model initialization failed: {e}")
            
        # if primary model fails, try the fallback model
        try:
            self.fallback_wrapper = VLLMWrapper(use_fallback=True)
            if self.fallback_wrapper.load_model():
                self.current_wrapper = self.fallback_wrapper
                self.logger.info("Fallback model loaded successfully")
                return True
            else:
                self.logger.error("Fallback model also failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Fallback model initialization failed: {e}")
            return False
    
    def get_wrapper(self) -> Optional[VLLMWrapper]:
        """Get currently available wrapper"""
        return self.current_wrapper
    
    def is_ready(self) -> bool:
        """Check if ready for inference"""
        return self.current_wrapper is not None and self.current_wrapper.model is not None
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        if self.primary_wrapper:
            self.primary_wrapper.cleanup()
        if self.fallback_wrapper:
            self.fallback_wrapper.cleanup()
        self.current_wrapper = None
        self.logger.info("All model resources cleaned up")
