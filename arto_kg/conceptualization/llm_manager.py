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
from arto_kg.conceptualization.utils import setup_logger
from arto_kg.config.model_config import (GENERATION_LLM_CONFIG, FALLBACK_LLM_CONFIG, GENERATION_CONFIG,
                                         CHAT_TEMPLATE_KWARGS)
from .output_parsers.base_parser import extract_json
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
        # Arguments for this model's chat template (e.g. Qwen3 without thinking).
        self.chat_template_kwargs = dict(CHAT_TEMPLATE_KWARGS.get(self.config.get("model"), {}))
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
            self.logger.warning("CUDA not available")

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
                self.logger.info(f"Adjusted tensor_parallel_size: {requested_gpus} -> {gpu_count}")
            else:
                self.logger.info(f"GPU configuration OK: using {requested_gpus}/{gpu_count} GPUs")
        
        # Single GPU: log that disable_custom_all_reduce is kept
        if gpu_count == 1 and self.config.get('tensor_parallel_size', 1) == 1:
            if 'disable_custom_all_reduce' in self.config:
                self.logger.info("Single GPU detected, keeping disable_custom_all_reduce setting")
            
    def _apply_tokenizer_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply the tokenizer's own chat template, for vLLM builds without llm.chat().

        Raises if the tokenizer has no template rather than guessing a format.
        """
        tokenizer = self.model.get_tokenizer()
        if not getattr(tokenizer, "chat_template", None):
            raise RuntimeError(
                "llm.chat() is unavailable and the tokenizer has no chat template; "
                "refusing to guess a prompt format for this model")
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             **(getattr(self, "chat_template_kwargs", None) or {}))

    def _chat(self, conversations):
        """llm.chat with this model's chat-template arguments."""
        kwargs = getattr(self, "chat_template_kwargs", None)
        if kwargs:
            return self.model.chat(conversations, self.sampling_params,
                                   chat_template_kwargs=kwargs)
        return self.model.chat(conversations, self.sampling_params)

    def _generate_with_model_template(self, system_prompt: str, user_prompt: str):
        """Generate, letting vLLM apply the model's own chat template.

        Errors from llm.chat() propagate so the caller can retry or fail the stage.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.model, "chat"):
            return self._chat(messages)

        formatted_prompt = self._apply_tokenizer_template(messages)
        return self.model.generate([formatted_prompt], self.sampling_params)
    
    @staticmethod
    def _stop_info(completion) -> Dict[str, Any]:
        """Why generation ended, and how many tokens it produced.

        finish_reason "length" means the token budget ran out; "stop" means a stop
        string (named by `stop_reason`) or, with stop_reason None, the model's EOS.
        """
        return {
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
            "generated_tokens": len(getattr(completion, "token_ids", []) or []),
        }

    def _log_short_completion(self, stop_info: Dict[str, Any], text: str) -> None:
        """Log a warning when a generation ends early, with the reason."""
        reason = stop_info.get("finish_reason")
        tokens = stop_info.get("generated_tokens", 0)
        if reason == "length":
            self.logger.warning(
                f"Generation hit the token budget ({tokens} tokens); the answer "
                f"is truncated. Raise GENERATION_CONFIG['max_tokens'].")
        elif tokens and tokens < 200:
            self.logger.warning(
                f"Generation ended after only {tokens} tokens "
                f"(finish_reason={reason!r}, stop_reason={stop_info.get('stop_reason')!r}, "
                f"{len(text)} chars). Not a budget problem.")

    def generate_single(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Generate and parse one response for a system/user prompt pair."""
        if self.model is None:
            return {"error": "Model not loaded"}
            
        try:
            outputs = self._generate_with_model_template(system_prompt, user_prompt)

            if not outputs or not outputs[0].outputs:
                return {"error": "No output generated"}

            completion = outputs[0].outputs[0]
            generated_text = completion.text.strip()
            stop_info = self._stop_info(completion)
            self._log_short_completion(stop_info, generated_text)

            # Parse output using unified parser
            parsed_result = self.output_parser.parse(generated_text, self.current_model_name)

            # Merge raw result and parsed result
            result = {
                "success": True,
                "content": generated_text,
                "raw_output": outputs[0],
                **stop_info,
                **parsed_result
            }

            return result
            
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return {"error": str(e)}
    
    def generate_batch_chat(self, conversations: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Batch generation with the model's own chat template.

        conversations: [[{"role": "system", ...}, {"role": "user", ...}], ...]
        """
        if self.model is None:
            return [{"error": "Model not loaded"}] * len(conversations)

        if not hasattr(self.model, "chat"):
            # Use the tokenizer's own template (see _apply_tokenizer_template).
            self.logger.warning("llm.chat() unavailable; applying the tokenizer's chat template")
            try:
                prompts = [self._apply_tokenizer_template(c) for c in conversations]
            except Exception as e:
                self.logger.error(f"Batch chat generation error: {e}")
                return [{"error": str(e), "index": i} for i in range(len(conversations))]
            return self.generate_batch(prompts)

        try:
            self.logger.info(f"Starting batch generation for {len(conversations)} conversations")
            outputs = self._chat(conversations)
            return self._collect_outputs(outputs)

        except Exception as e:
            self.logger.error(f"Batch chat generation error: {e}")
            return [{"error": str(e), "index": i} for i in range(len(conversations))]

    def _collect_outputs(self, outputs) -> List[Dict[str, Any]]:
        """Turn vLLM RequestOutputs into parsed result dicts."""
        results = []
        for i, output in enumerate(outputs):
            if output.outputs:
                completion = output.outputs[0]
                generated_text = completion.text.strip()
                stop_info = self._stop_info(completion)
                self._log_short_completion(stop_info, generated_text)
                parsed_result = self.output_parser.parse(generated_text, self.current_model_name)
                results.append({
                    "success": True,
                    "content": generated_text,
                    "index": i,
                    "raw_output": output,
                    **stop_info,
                    **parsed_result
                })
            else:
                results.append({"error": "No output generated", "index": i})

        self.logger.info(f"Batch generation completed: {len(results)} results")
        return results

    def generate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Batch generation from pre-formatted prompt strings.
        The caller applies the chat template; generate_batch_chat() applies it itself.
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
        """Generate a response and return its parsed JSON fields, or an error entry."""
        # Generate response
        result = self.generate_single(system_prompt, user_prompt)
        
        if "error" in result:
            return {
                "error": result["error"],
                "raw_output": "",
                "parsing_details": {"status": "generation_failed"}
            }
            
        # The answer's fields, then the reply record (including why generation
        # ended), which a field of the same name in the answer does not replace.
        enhanced_result = self._answer_fields(result)
        enhanced_result.update({
            "raw_output": result.get("raw_output", result["content"]),
            "parsing_details": result.get("parsing_details", {}),
            "finish_reason": result.get("finish_reason"),
            "stop_reason": result.get("stop_reason"),
            "generated_tokens": result.get("generated_tokens"),
        })
        return enhanced_result

    @staticmethod
    def _answer_fields(result: Dict[str, Any]) -> Dict[str, Any]:
        """The answer's JSON fields, or {"error": ...} when there is no answer.

        Only the parser's answer (`json_data`, else the `final` segment) is
        read, never the reasoning. An answer whose own JSON carries an `error`
        field is the model declining the task and is reported as a failure.
        """
        data = result.get("json_data") or extract_json(result.get("final"))
        if not data:
            details = result.get("parsing_details") or {}
            return {"error": details.get("parsing_error") or "No JSON data found"}
        return dict(data)
    
    def generate_batch_json_responses(self, prompt_pairs: List[tuple]) -> List[Dict[str, Any]]:
        """
        Batch generate JSON responses
        prompt_pairs: [(system_prompt, user_prompt), ...]
        """
        # vLLM applies the model's own chat template.
        conversations = [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            for system_prompt, user_prompt in prompt_pairs
        ]

        batch_results = self.generate_batch_chat(conversations)
        
        # batch_results already holds parsed results
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
                enhanced_result = self._answer_fields(result)
                enhanced_result.update({
                    "raw_output": result.get("raw_output", result["content"]),
                    "parsing_details": result.get("parsing_details", {
                        "status": "success" if result.get("json_data") else "failed"
                    }),
                    "index": i
                })
                json_results.append(enhanced_result)
                
        return json_results
    
    def test_model_response(self) -> Dict[str, Any]:
        """Send a trivial JSON request to check that the model responds."""
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
    
    def __init__(self, use_fallback_only: bool = False, allow_fallback: bool = False):
        self.logger = setup_logger("vllm_manager")
        self.primary_wrapper = None
        self.fallback_wrapper = None
        self.current_wrapper = None
        # Use only the fallback model, by explicit choice.
        self.use_fallback_only = use_fallback_only
        # Substituting the fallback when the primary fails to load is opt-in,
        # since the output does not show which model ran.
        self.allow_fallback = allow_fallback

    def initialize(self) -> bool:
        """Load the generation model.

        Falls back to the smaller model only when allowed or requested; the model
        actually loaded is logged and reported by get_active_model().
        """
        if self.use_fallback_only:
            self.logger.warning("Fallback model requested explicitly; skipping primary model")
        else:
            try:
                self.logger.info("Attempting to load primary model...")
                self.primary_wrapper = VLLMWrapper(use_fallback=False)

                if self.primary_wrapper.load_model():
                    self.current_wrapper = self.primary_wrapper
                    self.logger.info(
                        f"ACTIVE GENERATION MODEL (primary): {self.get_active_model()}"
                    )
                    return True

                self.logger.error("Primary model failed to load")

            except Exception as e:
                self.logger.error(f"Primary model initialization failed: {e}")

            if not self.allow_fallback:
                self.logger.error(
                    "Refusing to substitute the fallback model. The output would be "
                    "indistinguishable from a primary-model run. Check the primary model, "
                    "or re-run with --fallback-llm to use the smaller model on purpose."
                )
                return False

        # Load the fallback model (FALLBACK_LLM_CONFIG).
        try:
            self.fallback_wrapper = VLLMWrapper(use_fallback=True)
            if self.fallback_wrapper.load_model():
                self.current_wrapper = self.fallback_wrapper
                self.logger.warning(
                    f"ACTIVE GENERATION MODEL (FALLBACK): {self.get_active_model()} "
                    "- results are NOT from the primary model"
                )
                return True

            self.logger.error("Fallback model also failed")
            return False

        except Exception as e:
            self.logger.error(f"Fallback model initialization failed: {e}")
            return False

    def get_active_model(self) -> str:
        """Name of the model that is actually loaded."""
        if self.current_wrapper is None:
            return "none"
        config = getattr(self.current_wrapper, "config", {}) or {}
        return config.get("model", "unknown")
    
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
