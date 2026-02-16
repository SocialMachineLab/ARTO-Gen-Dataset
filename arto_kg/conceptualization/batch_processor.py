import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from arto_kg.conceptualization.utils import setup_logger, chunk_list, format_duration
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
    
    def __init__(self, vllm_wrapper, object_selector=None):
        self.logger = setup_logger("batch_processor")
        self.vllm_wrapper = vllm_wrapper
        self.object_selector = object_selector
        self.batch_size = BATCH_CONFIG["default_batch_size"]
        self.max_batch_size = BATCH_CONFIG["max_batch_size"]
        self.timeout = BATCH_CONFIG["timeout_seconds"]
        self.max_retries = BATCH_CONFIG["max_retries"]
        
    def process_batch_requests(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """
        Process batch requests
        
        Args:
            requests: Batch request list
            
        Returns:
            Batch result list
        """
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
        """
        Process single batch
        
        Args:
            requests: List of requests in a single batch
            
        Returns:
            List of results in a single batch
        """
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
    
    def collect_stage_requests(self, artwork_specs: List[Dict[str, Any]], 
                             stage: str, system_prompt: str) -> List[BatchRequest]:
        """
        Collect batch requests for a specific stage
        
        Args:
            artwork_specs: List of artwork specifications
            stage: Processing stage name
            system_prompt: System prompt
            
        Returns:
            List of batch requests
        """
        requests = []
        
        for i, spec in enumerate(artwork_specs):
            user_prompt = self._build_stage_user_prompt(spec, stage)
            
            request = BatchRequest(
                request_id=f"{stage}_{i}_{spec.get('artwork_id', f'unknown_{i}')}",
                stage=stage,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                metadata={
                    "artwork_index": i,
                    "artwork_id": spec.get("artwork_id", ""),
                    "stage": stage,
                    "timestamp": time.time()
                }
            )
            requests.append(request)
        
        return requests
    
    def _build_stage_user_prompt(self, artwork_spec: Dict[str, Any], stage: str) -> str:
        """
        Build user prompt for a specific stage
        
        Args:
            artwork_spec: Artwork specification
            stage: Processing stage
            
        Returns:
            User prompt
        """
        objects = artwork_spec.get("objects", [])
        style = artwork_spec.get("style", "Abstract")
        
        if stage == "spatial_relationships":
            enhanced_objects = artwork_spec.get("enhanced_objects", [])
            if enhanced_objects:
                objects_info = []
                for obj in enhanced_objects:
                    obj_desc = f"- {obj.get('name', 'unknown')}: size={obj.get('size', 'Medium')}"
                    objects_info.append(obj_desc)
                return f"Enhanced Objects:\n{chr(10).join(objects_info)}\n\nArt Style: {style}\n\nDesign basic spatial relationships for these objects in a {style} artwork."
            else:
                return f"Objects: {', '.join(objects)}\nArt Style: {style}\n\nDesign basic spatial relationships for these objects in a {style} artwork."
        
        elif stage == "environment_details":
            spatial_data = artwork_spec.get("spatial_relationships_data", {})
            return f"Objects: {', '.join(objects)}\nArt Style: {style}\nSpatial relationships: {spatial_data}\n\nSpecify the detailed environmental conditions."
        
        elif stage == "artistic_expression":
            spatial_data = artwork_spec.get("spatial_relationships_data", {})
            environment_data = artwork_spec.get("environment_details_data", {})
            return f"Objects: {', '.join(objects)}\nArt Style: {style}\nAll previous stages: {{'spatial_relationships': {spatial_data}, 'environment': {environment_data}}}\n\nDetermine the artistic expression elements and create a final image generation prompt."
        
        # Legacy support for old stage names
        elif stage == "scene_framework":
            return f"Objects: {', '.join(objects)}\nArt Style: {style}\n\nDetermine the appropriate scene framework for these objects in the given style."
        
        elif stage == "spatial_layout":
            framework = artwork_spec.get("framework_data", {})
            return f"Objects: {', '.join(objects)}\nArt Style: {style}\nScene Framework: {framework}\n\nDesign the spatial layout and object relationships."
        
        else:
            return f"Objects: {', '.join(objects)}\nArt Style: {style}\n\nProcess this artwork specification."
    
    def process_pipeline_stage(self, artwork_specs: List[Dict[str, Any]], 
                             stage_name: str, system_prompt: str, 
                             output_manager=None) -> List[Dict[str, Any]]:
        """
        Process specific stage in the pipeline
        
        Args:
            artwork_specs: List of artwork specifications
            stage_name: Stage name
            system_prompt: System prompt
            
        Returns:
            List of processed artwork specifications
        """

        
        # Collect requests
        requests = self.collect_stage_requests(artwork_specs, stage_name, system_prompt)
        
        # Execute batch processing
        results = self.process_batch_requests(requests)
        
        # Distribute results back to artwork specs and save detailed output
        updated_specs = []
        for spec, result in zip(artwork_specs, results):
            updated_spec = spec.copy()
            artwork_id = spec.get('artwork_id', 'unknown')
            
            if result.success:
                updated_spec[f"{stage_name}_data"] = result.content
                updated_spec[f"{stage_name}_status"] = "success"
                
                # Save detailed output to file system
                if output_manager:
                    try:
                        # Get raw output and parsing details
                        raw_output = result.content.get('raw_output', '') if isinstance(result.content, dict) else str(result.content)
                        parsing_details = result.content.get('parsing_details', {}) if isinstance(result.content, dict) else {}
                        
                        # Create clean parsed result (excluding raw_output and parsing_details)
                        clean_result = {k: v for k, v in result.content.items() 
                                      if k not in ["raw_output", "parsing_details"]} if isinstance(result.content, dict) else result.content
                        
                        # Create processing info
                        processing_info = {
                            "stage": stage_name,
                            "artwork_id": artwork_id,
                            "objects": spec.get("objects", []),
                            "style": spec.get("style", ""),
                            "system_prompt": system_prompt,
                            "user_prompt": f"Objects: {', '.join(spec.get('objects', []))}\nArt Style: {spec.get('style', '')}",
                            "parsing_details": parsing_details,
                            "processing_time": result.processing_time
                        }
                        
                        # Save stage output
                        output_manager.save_stage_output(
                            stage=stage_name,
                            artwork_id=artwork_id,
                            raw_output=raw_output,
                            parsed_result=clean_result,
                            processing_info=processing_info
                        )
                        
                        self.logger.info(f"Saved detailed output for {artwork_id} stage {stage_name}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to save detailed output for {artwork_id} stage {stage_name}: {e}")
                
            else:
                self.logger.warning(f"Stage {stage_name} failed for {artwork_id}: {result.error}")
                updated_spec[f"{stage_name}_data"] = {}
                updated_spec[f"{stage_name}_status"] = "failed"
                updated_spec[f"{stage_name}_error"] = result.error
                
                # Also save failed detailed output
                if output_manager:
                    try:
                        failed_result = {"error": result.error, "raw_response": str(result.content)}
                        processing_info = {
                            "stage": stage_name,
                            "artwork_id": artwork_id,
                            "objects": spec.get("objects", []),
                            "style": spec.get("style", ""),
                            "system_prompt": system_prompt,
                            "user_prompt": f"Objects: {', '.join(spec.get('objects', []))}\nArt Style: {spec.get('style', '')}",
                            "parsing_details": {"status": "failed", "parsing_error": result.error},
                            "processing_time": result.processing_time
                        }
                        
                        output_manager.save_stage_output(
                            stage=stage_name,
                            artwork_id=artwork_id,
                            raw_output=str(result.content),
                            parsed_result=failed_result,
                            processing_info=processing_info
                        )
                        
                        self.logger.info(f"Saved failed output for {artwork_id} stage {stage_name}")
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to save failed output for {artwork_id} stage {stage_name}: {e}")
            
            updated_spec[f"{stage_name}_processing_time"] = result.processing_time
            updated_specs.append(updated_spec)
        
        return updated_specs
    
    def retry_failed_requests(self, failed_requests: List[BatchRequest], 
                            max_retries: Optional[int] = None) -> List[BatchResult]:
        """
        Retry failed requests
        
        Args:
            failed_requests: List of failed requests
            max_retries: Maximum number of retries
            
        Returns:
            List of retry results
        """
        if max_retries is None:
            max_retries = self.max_retries
        
        if not failed_requests:
            return []
        
        self.logger.info(f"Retrying {len(failed_requests)} failed requests")
        
        retry_results = []
        
        for attempt in range(max_retries):
            self.logger.info(f"Retry attempt {attempt + 1}/{max_retries}")
            
            # Process failed requests
            current_results = self.process_batch_requests(failed_requests)
            
            # Separate successful and failed results
            success_results = [r for r in current_results if r.success]
            still_failed = [req for req, res in zip(failed_requests, current_results) if not res.success]
            
            retry_results.extend(success_results)
            
            # If all succeeded, break retry loop
            if not still_failed:
                break
            
            # Prepare for next retry
            failed_requests = still_failed
            
            # Retry interval
            if attempt < max_retries - 1:
                time.sleep(BATCH_CONFIG["retry_delay"])
        
        self.logger.info(f"Retry completed: {len(retry_results)} successful")
        return retry_results
    
    def get_batch_statistics(self, results: List[BatchResult]) -> Dict[str, Any]:
        """
        Get batch processing statistics
        
        Args:
            results: List of batch results
            
        Returns:
            Statistics
        """
        if not results:
            return {"total": 0, "success": 0, "failed": 0, "success_rate": 0.0}
        
        total_count = len(results)
        success_count = sum(1 for r in results if r.success)
        failed_count = total_count - success_count
        
        # Calculate processing time statistics
        processing_times = [r.processing_time for r in results if r.processing_time > 0]
        
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        min_time = min(processing_times) if processing_times else 0
        max_time = max(processing_times) if processing_times else 0
        
        # Group statistics by stage
        stage_stats = {}
        for result in results:
            stage = getattr(result, 'stage', 'unknown')
            if stage not in stage_stats:
                stage_stats[stage] = {"total": 0, "success": 0, "failed": 0}
            
            stage_stats[stage]["total"] += 1
            if result.success:
                stage_stats[stage]["success"] += 1
            else:
                stage_stats[stage]["failed"] += 1
        
        # Calculate success rate for each stage
        for stage, stats in stage_stats.items():
            stats["success_rate"] = stats["success"] / stats["total"] if stats["total"] > 0 else 0
        
        return {
            "total": total_count,
            "success": success_count,
            "failed": failed_count,
            "success_rate": success_count / total_count,
            "processing_time": {
                "average": avg_time,
                "minimum": min_time,
                "maximum": max_time,
                "total": sum(processing_times)
            },
            "stage_statistics": stage_stats,
            "throughput": {
                "requests_per_second": total_count / sum(processing_times) if processing_times else 0,
                "average_batch_size": self.batch_size
            }
        }
    
    def optimize_batch_size(self, sample_requests: List[BatchRequest], 
                          target_success_rate: float = 0.95) -> int:
        """
        Optimize batch size
        
        Args:
            sample_requests: List of sample requests
            target_success_rate: Target success rate
            
        Returns:
            Optimized batch size
        """
        if not sample_requests:
            return self.batch_size
        
        self.logger.info("Optimizing batch size...")
        
        # Test different batch sizes
        test_sizes = [4, 8, 16, 24, 32]
        test_sizes = [size for size in test_sizes if size <= len(sample_requests)]
        
        if not test_sizes:
            return min(self.batch_size, len(sample_requests))
        
        best_size = self.batch_size
        best_score = 0
        
        for test_size in test_sizes:
            if test_size > self.max_batch_size:
                continue
                
            # Test using samples
            test_requests = sample_requests[:test_size]
            
            start_time = time.time()
            test_results = self._process_single_batch(test_requests)
            test_time = time.time() - start_time
            
            # Calculate score (success rate + throughput)
            success_rate = sum(1 for r in test_results if r.success) / len(test_results)
            throughput = len(test_results) / test_time if test_time > 0 else 0
            
            # Comprehensive score
            score = success_rate * 0.7 + min(throughput / 10, 0.3)  # Success rate has higher weight
            
            self.logger.info(f"Batch size {test_size}: success_rate={success_rate:.3f}, throughput={throughput:.2f}, score={score:.3f}")
            
            if success_rate >= target_success_rate and score > best_score:
                best_score = score
                best_size = test_size
        
        self.logger.info(f"Optimized batch size: {best_size}")
        return best_size
    
    def create_artwork_batch_requests(self, artwork_count: int, 
                                    styles: Optional[List[str]] = None,
                                    max_secondary_objects: int = 8) -> List[Dict[str, Any]]:
        """
        Create initial specifications for batch artwork generation
        
        Args:
            artwork_count: Number of artworks
            styles: Optional list of styles
            max_secondary_objects: Maximum number of secondary objects
            
        Returns:
            List of artwork specifications
        """
        from arto_kg.config.model_config import ART_STYLES, COCO_OBJECTS
        import random
        
        artwork_specs = []
        
        for i in range(artwork_count):
            # Generate basic specifications
            artwork_id = f"batch_artwork_{i:04d}_{int(time.time())}"
            
            # Select style
            if styles:
                style = random.choice(styles)
            else:
                style = random.choice(ART_STYLES)
            
            # Select objects
            main_object = random.choice(list(COCO_OBJECTS.keys()))
            secondary_count = random.randint(0, max_secondary_objects)
            available_objects = [obj_id for obj_id in COCO_OBJECTS.keys() if obj_id != main_object]
            secondary_objects = random.sample(available_objects, min(secondary_count, len(available_objects)))
            
            all_object_ids = [main_object] + secondary_objects
            object_names = [COCO_OBJECTS[obj_id] for obj_id in all_object_ids]
            
            spec = {
                "artwork_id": artwork_id,
                "index": i,
                "style": style,
                "object_ids": all_object_ids,
                "objects": object_names,
                "max_secondary_objects": max_secondary_objects,
                "creation_time": time.time()
            }
            
            artwork_specs.append(spec)
        
        self.logger.info(f"Created {len(artwork_specs)} artwork specifications for batch processing")
        return artwork_specs
    
    def process_compatibility_batch(self, artwork_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Batch process object compatibility check + scene inference
        Use object_selector to process each artwork spec

        Args:
            artwork_specs: List of artwork specifications

        Returns:
            List of processed specifications (including scene_inference)
        """
        self.logger.info(f"Processing compatibility checks and scene inference for {len(artwork_specs)} artworks")

        updated_specs = []

        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                object_ids = spec.get("object_ids", [])
                style = spec.get("style", "Abstract")

                if not object_ids:
                    self.logger.warning(f"No objects for {artwork_id}, skipping")
                    spec["compatibility_status"] = "failed"
                    spec["scene_inference"] = {}
                    updated_specs.append(spec)
                    continue

                main_object_id = object_ids[0]
                secondary_object_ids = object_ids[1:]

                # Process using object_selector (with conflict resolution and scene inference)
                selection_result = self.object_selector.select_specific_objects(
                    main_object_id=main_object_id,
                    secondary_object_ids=secondary_object_ids,
                    style=style
                )

                # Update spec
                # Update spec with defensive check for object_names
                spec["object_ids"] = selection_result["object_ids"]
                object_names = selection_result.get("object_names", [])
                if not object_names:
                    object_names = self.object_selector.get_object_names(spec["object_ids"])
                spec["objects"] = object_names
                spec["scene_inference"] = selection_result.get("scene_inference", {})
                spec["compatibility_status"] = "compatible"

                self.logger.info(f"Processed {artwork_id}: {len(spec['object_ids'])} objects, "
                               f"scene: {spec['scene_inference'].get('selected_scene', {}).get('scene_type', 'N/A')}")

            except Exception as e:
                self.logger.error(f"Compatibility processing failed for {artwork_id}: {e}")
                spec["compatibility_status"] = "failed"
                spec["scene_inference"] = {}
                spec["error"] = str(e)

            updated_specs.append(spec)

        compatible_count = sum(1 for spec in updated_specs if spec.get("compatibility_status") == "compatible")
        self.logger.info(f"Compatibility processing completed: {compatible_count}/{len(updated_specs)} compatible")

        return updated_specs
    
    def estimate_processing_time(self, request_count: int, 
                               stages: List[str] = None) -> Dict[str, float]:
        """
        Estimate processing time
        
        Args:
            request_count: Number of requests
            stages: List of processing stages
            
        Returns:
            Time estimate information
        """
        if stages is None:
            stages = ["compatibility", "scene_framework", "spatial_layout", "environment_details", "artistic_expression"]
        
        # Experience-based time estimate (seconds/request)
        stage_time_estimates = {
            "compatibility": 2.0,
            "scene_framework": 3.0,
            "spatial_layout": 3.5,
            "environment_details": 3.0,
            "artistic_expression": 4.0,
            "object_enhancement": 2.5
        }
        
        total_time = 0
        stage_times = {}
        
        for stage in stages:
            stage_time_per_request = stage_time_estimates.get(stage, 3.0)
            batches_needed = (request_count + self.batch_size - 1) // self.batch_size
            stage_total_time = batches_needed * stage_time_per_request * self.batch_size / request_count * request_count
            
            stage_times[stage] = stage_total_time
            total_time += stage_total_time
        
        return {
            "total_estimated_time": total_time,
            "stage_times": stage_times,
            "estimated_batches": (request_count + self.batch_size - 1) // self.batch_size,
            "requests_per_batch": self.batch_size,
            "estimated_completion": time.time() + total_time
        }
    
    def monitor_batch_progress(self, results: List[BatchResult], 
                             total_expected: int) -> Dict[str, Any]:
        """
        Monitor batch processing progress
        
        Args:
            results: Current results list
            total_expected: Total expected count
            
        Returns:
            Progress information
        """
        completed = len(results)
        success = sum(1 for r in results if r.success)
        failed = completed - success
        
        progress_percentage = (completed / total_expected) * 100 if total_expected > 0 else 0
        
        return {
            "completed": completed,
            "total_expected": total_expected,
            "success": success,
            "failed": failed,
            "progress_percentage": progress_percentage,
            "success_rate": success / completed if completed > 0 else 0,
            "remaining": total_expected - completed,
            "status": "in_progress" if completed < total_expected else "completed"
        }

        
        
