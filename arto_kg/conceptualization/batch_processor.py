import random
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from arto_kg.conceptualization.utils import setup_logger, chunk_list, format_duration
from arto_kg.conceptualization.style_selector import StyleSelector
from arto_kg.config.model_config import BATCH_CONFIG


@dataclass
class BatchRequest:
    """Batch request data structure"""
    request_id: str
    stage: str  # Processing stage
    system_prompt: str
    user_prompt: str
    metadata: Dict[str, Any]


@dataclass
class BatchResult:
    """Batch result data structure"""
    request_id: str
    success: bool
    content: Any
    error: Optional[str]
    processing_time: float


class BatchProcessor:
    """Batch Processor"""
    
    def __init__(self, vllm_wrapper, object_selector=None, style_selector=None):
        self.logger = setup_logger("batch_processor")
        self.vllm_wrapper = vllm_wrapper
        self.object_selector = object_selector
        # Same selector as the single-artwork path, so both draw styles from the
        # same pool and reject an unusable style the same way.
        self.style_selector = style_selector or StyleSelector()
        self.batch_size = BATCH_CONFIG["default_batch_size"]
        
    def process_batch_requests(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Process requests in chunks of `batch_size` and return one result per request."""
        if not requests:
            return []
        
        self.logger.info(f"Processing {len(requests)} batch requests")
        start_time = time.time()
        
        all_results = []
        
        # Group requests for processing
        request_chunks = chunk_list(requests, self.batch_size)
        
        for i, chunk in enumerate(request_chunks):
            self.logger.info(f"Processing batch {i+1}/{len(request_chunks)} ({len(chunk)} requests)")
            
            chunk_results = self._process_single_batch(chunk)
            all_results.extend(chunk_results)
            
            # Short pause between batches to avoid overload
            if i < len(request_chunks) - 1:
                time.sleep(0.5)
        
        total_time = time.time() - start_time
        success_count = sum(1 for result in all_results if result.success)
        
        self.logger.info(f"Batch processing completed: {success_count}/{len(requests)} successful in {format_duration(total_time)}")
        
        return all_results
    
    def _process_single_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """Run one batch through the LLM in a single call."""
        if not requests:
            return []
        
        batch_start_time = time.time()
        
        # Prepare batch prompts
        prompt_pairs = [(req.system_prompt, req.user_prompt) for req in requests]
        
        try:
            # Execute batch inference
            batch_responses = self.vllm_wrapper.generate_batch_json_responses(prompt_pairs)
            
            # Process results
            results = []
            for i, (request, response) in enumerate(zip(requests, batch_responses)):
                processing_time = time.time() - batch_start_time
                
                if "error" in response:
                    result = BatchResult(
                        request_id=request.request_id,
                        success=False,
                        content=None,
                        error=response.get("error", "Unknown error"),
                        processing_time=processing_time
                    )
                else:
                    result = BatchResult(
                        request_id=request.request_id,
                        success=True,
                        content=response,
                        error=None,
                        processing_time=processing_time
                    )
                
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            
            # Create error results
            error_results = []
            for request in requests:
                error_result = BatchResult(
                    request_id=request.request_id,
                    success=False,
                    content=None,
                    error=str(e),
                    processing_time=time.time() - batch_start_time
                )
                error_results.append(error_result)
            
            return error_results
    
    def create_artwork_batch_requests(self, artwork_count: int, 
                                    styles: Optional[List[str]] = None,
                                    max_secondary_objects: int = 8) -> List[Dict[str, Any]]:
        """Create the initial specification (id and style) for each artwork in a batch."""
        from arto_kg.conceptualization.utils import generate_artwork_id

        artwork_specs = []

        for i in range(artwork_count):
            # Same style pool as single runs; objects are selected later.
            style = self.style_selector.select_style(
                preference=random.choice(styles) if styles else None
            )

            artwork_specs.append({
                "artwork_id": generate_artwork_id(),
                "index": i,
                "style": style,
                "max_secondary_objects": max_secondary_objects,
                "creation_time": time.time()
            })

        self.logger.info(f"Created {len(artwork_specs)} artwork specifications for batch processing")
        return artwork_specs
    
    def process_compatibility_batch(self, artwork_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select objects and infer the scene for each spec; failed specs are dropped."""
        self.logger.info(f"Processing compatibility checks and scene inference for {len(artwork_specs)} artworks")

        updated_specs = []

        rejected = 0
        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            style = spec.get("style")
            try:
                # Same two-layer selection as the single-artwork path.
                selection_result = self.object_selector.select_objects(
                    max_secondary_objects=spec.get("max_secondary_objects", 8),
                    style=style
                )

                spec["instances"] = selection_result["instances"]
                spec["object_ids"] = selection_result["object_ids"]
                spec["objects"] = selection_result["object_names"]
                spec["scene_inference"] = selection_result.get("scene_inference", {})
                spec["compatibility_status"] = "compatible"

                self.logger.info(
                    f"Processed {artwork_id}: {len(spec['instances'])} objects "
                    f"({', '.join(spec['objects'])}), scene: "
                    f"{spec['scene_inference'].get('selected_scene', {}).get('scene_type', 'N/A')}"
                )
                updated_specs.append(spec)

            except Exception as e:
                # Drop the artwork.
                rejected += 1
                self.logger.warning(f"Rejected {artwork_id} during selection: {e}")

        if rejected:
            self.logger.warning(f"Dropped {rejected}/{len(artwork_specs)} artworks during selection")

        compatible_count = sum(1 for spec in updated_specs if spec.get("compatibility_status") == "compatible")
        self.logger.info(f"Compatibility processing completed: {compatible_count}/{len(updated_specs)} compatible")

        return updated_specs
