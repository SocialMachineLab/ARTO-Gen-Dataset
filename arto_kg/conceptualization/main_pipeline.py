"""
Main artwork conceptualization pipeline: initialization, single and batch generation.
"""

import os
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

from .utils import setup_logger, save_json, generate_artwork_id, get_timestamp, print_progress, create_batch_output_structure
from .object_instance import (
    ObjectInstance,
    to_enhanced_objects,
    to_instance_index,
    to_object_ids,
    to_object_names,
)
from .llm_manager import VLLMManager
from .style_selector import StyleSelector
from .object_selector import ObjectSelector
from .object_enhancer import ObjectEnhancer
from .scene_composer import SceneComposer
from .environment_designer import EnvironmentDesigner
from .prompt_generator import PromptGenerator
from .batch_processor import BatchProcessor



class ArtworkImagePipeline:
    """Artwork conceptualization pipeline."""
    
    def __init__(self, batch_mode: bool = True, output_base_dir: str = "data/output",
                 use_fallback_llm: bool = False, allow_llm_fallback: bool = False):
        self.logger = setup_logger("artwork_pipeline")
        self.batch_mode = batch_mode
        self.output_base_dir = output_base_dir
        
        # Output manager (created during batch generation)
        self.current_batch_dir = None
        self.current_output_manager = None
        
        # Initialize components
        self.logger.info("Initializing artwork pipeline...")
        
        # vLLM manager
        self.vllm_manager = VLLMManager(use_fallback_only=use_fallback_llm,
                                        allow_fallback=allow_llm_fallback)
        self.vllm_wrapper = None
        
        # Functional components (set to None during initialization, initialized after vLLM is loaded)
        self.style_manager = None
        self.object_selector = None
        self.object_enhancer = None
        self.scene_composer = None
        self.environment_designer = None
        self.prompt_generator = None
        self.batch_processor = None

        
        # State tracking
        self.is_initialized = False
        self.initialization_time = None
        
    def initialize(self) -> bool:
        """Load the LLM and create all components; return whether it succeeded."""
        start_time = time.time()
        self.logger.info("Starting pipeline initialization...")
        
        try:
            # Initialize vLLM
            if not self.vllm_manager.initialize():
                self.logger.error("Failed to initialize vLLM")
                return False
            
            self.vllm_wrapper = self.vllm_manager.get_wrapper()
            if not self.vllm_wrapper:
                self.logger.error("Failed to get vLLM wrapper")
                return False
            
            # Initialize functional components (output manager not passed, set later during generation)
            self.style_manager = StyleSelector()
            self.object_selector = ObjectSelector(self.vllm_wrapper)
            self.object_enhancer = ObjectEnhancer(self.vllm_wrapper)
            self.scene_composer = SceneComposer(self.vllm_wrapper)
            self.environment_designer = EnvironmentDesigner(self.vllm_wrapper)
            self.prompt_generator = PromptGenerator(self.vllm_wrapper)
            
            if self.batch_mode:
                self.batch_processor = BatchProcessor(
                    self.vllm_wrapper, self.object_selector, self.style_manager
                )
            

            
            # Create output directories
            os.makedirs(self.output_base_dir, exist_ok=True)
            os.makedirs(f"{self.output_base_dir}/artworks", exist_ok=True)
            os.makedirs(f"{self.output_base_dir}/logs", exist_ok=True)
            
            self.initialization_time = time.time() - start_time
            self.is_initialized = True
            
            self.logger.info(f"Pipeline initialized successfully in {self.initialization_time:.2f}s")
            
            # Test model response
            if not self._test_model_functionality():
                self.logger.warning("Model test failed, but continuing...")



            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    def _test_model_functionality(self) -> bool:
        """Test model functionality"""
        try:
            test_result = self.vllm_wrapper.test_model_response()
            return "error" not in test_result
        except Exception as e:
            self.logger.error(f"Model test error: {e}")
            return False
    
    def _setup_output_manager_for_components(self, output_manager) -> None:
        """Set up output manager for components"""
        if hasattr(self.scene_composer, 'output_manager'):
            self.scene_composer.output_manager = output_manager
        if hasattr(self.environment_designer, 'output_manager'):
            self.environment_designer.output_manager = output_manager
        if hasattr(self.object_enhancer, 'output_manager'):
            self.object_enhancer.output_manager = output_manager
        if hasattr(self.prompt_generator, 'output_manager'):
            self.prompt_generator.output_manager = output_manager
    
    def generate_single_artwork(self, style: Optional[str] = None,
                              max_secondary_objects: int = 8,
                              output_dir: Optional[str] = None,
                              use_detailed_output: bool = False) -> Dict[str, Any]:
        """Generate and save one artwork specification.

        Returns the artwork data, or a dict with an "error" key on failure.
        """
        if not self.is_initialized:
            self.logger.error("Pipeline not initialized")
            return {"error": "Pipeline not initialized"}
        
        if output_dir is None:
            if use_detailed_output:
                # Create detailed output structure
                self.current_batch_dir, self.current_output_manager = create_batch_output_structure(self.output_base_dir)
                output_dir = self.current_batch_dir
                self._setup_output_manager_for_components(self.current_output_manager)
            else:
                output_dir = f"{self.output_base_dir}/artworks"
        elif use_detailed_output:
            # If output directory is specified and detailed output is required
            from arto_kg.conceptualization.utils import StageOutputManager
            os.makedirs(output_dir, exist_ok=True)
            self.current_output_manager = StageOutputManager(output_dir)
            self._setup_output_manager_for_components(self.current_output_manager)
        
        artwork_id = generate_artwork_id()
        self.logger.info(f"Generating single artwork: {artwork_id}")
        
        try:
            start_time = time.time()
            
            # step 1: select style
            selected_style = self.style_manager.select_style(style)
            self.logger.info(f"Selected style: {selected_style}")

            # step 2: select objects WITH scene inference
            selection_result = self.object_selector.select_objects(
                max_secondary_objects=max_secondary_objects,
                style=selected_style
            )
            instances = [ObjectInstance.from_dict(d) for d in selection_result["instances"]]
            scene_inference = selection_result.get('scene_inference', {})
            self.logger.info(f"Selected objects: {to_object_names(instances)}")
            self.logger.info(f"Scene inference: {scene_inference.get('selected_scene', {}).get('scene_type', 'N/A')}")

            # step 3: determine environment details (using scene inference)
            selected_scene = scene_inference.get('selected_scene', {})
            scene_brief = selected_scene.get('scene_brief', selected_scene.get('scene_type', 'abstract scene'))
            object_interpretations = selected_scene.get('object_interpretations', {})

            environment_details = self.environment_designer.determine_environment_details(
                scene_brief=scene_brief,
                style=selected_style,
                objects=to_object_names(instances),
                object_interpretations=object_interpretations
            )

            # How the selected scene reinterprets each instance, for enhancement.
            if object_interpretations:
                environment_details['object_interpretations'] = object_interpretations

            self.logger.info("Determined environment details from scene inference")

            # step 4: enhance object visual descriptions (using environment details)
            instances = self.object_enhancer.enhance_objects(
                instances,
                selected_style,
                environment=environment_details
            )
            enhanced_objects = to_enhanced_objects(instances)
            self.logger.info(f"Enhanced {len(instances)} objects with environment-aware descriptions")

            # step 5: create complete spatial composition (with semantic relations)
            composition_data = self.scene_composer.create_complete_composition(
                instances,
                selected_style,
                environment_context=environment_details,
                artwork_id=artwork_id if use_detailed_output else None
            )

            self.logger.info("Created complete spatial composition")

            # Package environment data for final output
            environment_data = {
                "environment_details": environment_details,
                "scene_inference": scene_inference
            }




            # step 6: generate final prompts
            # A PromptGenerationError fails the artwork via the handler below.
            final_prompt_package = self.prompt_generator.create_final_prompt_package(
                enhanced_objects, selected_style, composition_data, environment_data
            )

            self.logger.info("Generated final prompt package")

            # assemble final data
            generation_time = time.time() - start_time
            
            final_data = {
                "artwork_id": artwork_id,
                "generation_timestamp": get_timestamp(),
                "generation_time": generation_time,
                "style": selected_style,
                "objects": {
                    # enhanced_objects is the record readers use; instances is a
                    # compact two-layer index (see to_instance_index).
                    "instances": to_instance_index(instances),
                    "object_ids": to_object_ids(instances),
                    "object_names": to_object_names(instances),
                    "enhanced_objects": enhanced_objects
                },
                "composition": composition_data,
                "environment": environment_data,
                "final_prompts": final_prompt_package,
                "metadata": {
                    "max_secondary_objects": max_secondary_objects,
                    "batch_mode": self.batch_mode,
                    "model_info": self.vllm_wrapper.get_model_info()
                }
            }
            
            # Specifications go to <output_dir>/final_results/, the directory
            # Stage 2 reads, for single and batch runs alike.
            filename = os.path.join(output_dir, "final_results", f"{artwork_id}.json")
            save_json(final_data, filename)

            self.logger.info(f"Artwork generated successfully in {generation_time:.2f}s")
            self.logger.info(f"Saved to: {filename}")
            
            return final_data
            
        except Exception as e:
            self.logger.error(f"Single artwork generation failed: {e}")
            return {"error": str(e), "artwork_id": artwork_id}
    
    def generate_batch_artworks(self, count: int,
                              styles: Optional[List[str]] = None,
                              max_secondary_objects: int = 8,
                              output_dir: Optional[str] = None,
                              save_intermediate: bool = True) -> List[Dict[str, Any]]:
        """Generate `count` artworks stage by stage across the whole batch.

        Artworks with a failed stage are dropped; a prompt-generation failure
        is returned as an error entry.
        """
        if not self.is_initialized:
            self.logger.error("Pipeline not initialized")
            return []
        
        if not self.batch_mode or not self.batch_processor:
            self.logger.warning("Batch mode not available, falling back to sequential generation")
            return self._generate_sequential_batch(count, styles, max_secondary_objects, output_dir, True)
        
        # Create the batch output directory structure
        if output_dir is None:
            self.current_batch_dir, self.current_output_manager = create_batch_output_structure(self.output_base_dir)
            output_dir = self.current_batch_dir
        else:
            # Use the stage output structure in the given directory
            from arto_kg.conceptualization.utils import StageOutputManager
            os.makedirs(output_dir, exist_ok=True)
            self.current_output_manager = StageOutputManager(output_dir)
        
        # Set up output manager for all components
        self._setup_output_manager_for_components(self.current_output_manager)
        
        self.logger.info(f"Starting batch generation of {count} artworks")
        batch_start_time = time.time()
        
        try:
            # Step 1: Create initial artwork specifications
            artwork_specs = self.batch_processor.create_artwork_batch_requests(
                count, styles, max_secondary_objects
            )
            
            # Step 2: Batch object compatibility check + scene inference
            print_progress(1, 6, "Pipeline Progress")

            artwork_specs = self.batch_processor.process_compatibility_batch(artwork_specs)

            # Write a checkpoint after every batch stage so a timeout does not
            # lose the run. `stage` records the last completed stage.
            def checkpoint(stage: str) -> None:
                if not save_intermediate:
                    return
                try:
                    save_json({"stage": stage, "data": artwork_specs},
                              os.path.join(output_dir,
                                           "intermediate_spatial_relationships.json"))
                    self.logger.info(f"Checkpoint written after {stage}")
                except Exception as exc:
                    # A failed checkpoint never stops the run.
                    self.logger.warning(f"Could not write checkpoint after {stage}: {exc}")

            checkpoint("compatibility")

            # Step 3: Batch environment refinement (based on scene inference)
            print_progress(2, 6, "Pipeline Progress")
            self.logger.info("Processing environment details based on scene inference")
            self._process_environment_details_batch(artwork_specs)
            checkpoint("environment_details")

            # Step 4: Object enhancement (based on environment details)
            print_progress(3, 6, "Pipeline Progress")
            self._process_object_enhancement_batch(artwork_specs)
            checkpoint("object_enhancement")

            # Step 5: Batch spatial composition (using scene_composer)
            print_progress(4, 6, "Pipeline Progress")
            self.logger.info("Processing spatial composition batch")
            self._process_spatial_composition_batch(artwork_specs)
            checkpoint("spatial_relationships")

            # Step 6: Final prompt generation and organization
            print_progress(5, 6, "Pipeline Progress")
            final_artworks = self._finalize_batch_artworks(artwork_specs, output_dir)
            
            batch_time = time.time() - batch_start_time
            success_count = len([art for art in final_artworks if "error" not in art])
            
            # Save batch summary
            batch_summary = {
                "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "total_requested": count,
                "total_generated": len(final_artworks),
                "successful": success_count,
                "failed": count - success_count,
                "success_rate": success_count / count if count else 0,
                "generation_time": batch_time,
                "average_time_per_artwork": batch_time / len(final_artworks) if final_artworks else 0,
                "styles_used": sorted({art["style"] for art in final_artworks if art.get("style")}),
                "output_directory": output_dir,
                "batch_statistics": self._compute_stage_statistics(artwork_specs)
            }
            
            # Save batch summary to final results folder
            final_results_dir = os.path.join(output_dir, "final_results")
            save_json(batch_summary, os.path.join(final_results_dir, "batch_summary.json"))
            
            # Also save to the output root
            save_json(batch_summary, os.path.join(output_dir, "batch_summary.json"))
            
            self.logger.info(f"Batch generation completed: {success_count}/{count} successful in {batch_time:.2f}s")
            return final_artworks
            
        except Exception as e:
            self.logger.error(f"Batch generation failed: {e}")
            return []
    
    def _generate_sequential_batch(self, count: int, styles: Optional[List[str]], 
                                 max_secondary_objects: int, output_dir: Optional[str], 
                                 use_detailed_output: bool = True) -> List[Dict[str, Any]]:
        """Generate batch artworks sequentially (fallback method)"""
        self.logger.info("Using sequential generation mode")
        
        if output_dir is None:
            output_dir = f"{self.output_base_dir}/artworks/sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = []
        
        for i in range(count):
            print_progress(i + 1, count, "Sequential Generation")
            
            style = None
            if styles:
                style = random.choice(styles)
            
            try:
                artwork = self.generate_single_artwork(
                    style, max_secondary_objects, output_dir, use_detailed_output
                )
                results.append(artwork)
            except Exception as e:
                self.logger.error(f"Failed to generate artwork {i+1}: {e}")
                results.append({"error": str(e), "index": i})
        
        return results
    
    def _process_environment_details_batch(self, artwork_specs: List[Dict[str, Any]]) -> None:
        """Batch process environment details (based on scene inference results)"""
        self.logger.info("Processing environment details batch")

        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                style = spec.get("style")
                scene_inference = spec.get("scene_inference", {})
                selected_scene = scene_inference.get("selected_scene", {})
                scene_brief = selected_scene.get("scene_brief", selected_scene.get("scene_type", "abstract scene"))
                object_interpretations = selected_scene.get("object_interpretations", {})

                # Determine environment details (objects are part of the input:
                # the environment is designed for this specific set of objects)
                environment_details = self.environment_designer.determine_environment_details(
                    scene_brief=scene_brief,
                    style=style,
                    objects=spec.get("objects", []),
                    object_interpretations=object_interpretations
                )

                if object_interpretations:
                    environment_details['object_interpretations'] = object_interpretations

                spec["environment_details"] = environment_details
                # "fallback" marks the default environment used after the LLM
                # failed; the artwork is kept.
                spec["environment_details_status"] = (
                    "fallback" if environment_details.get("source") == "fallback" else "success")

                self.logger.info(f"Environment details determined for {artwork_id}: "
                               f"{environment_details.get('scene_brief', 'N/A')}, "
                               f"{environment_details.get('time_of_day', 'N/A')}, "
                               f"{environment_details.get('period', 'N/A')}")

            except Exception as e:
                self.logger.error(f"Environment details determination failed for {artwork_id}: {e}")
                spec["environment_details"] = {}
                spec["environment_details_status"] = "failed"
                spec["environment_details_error"] = str(e)

    def _process_spatial_composition_batch(self, artwork_specs: List[Dict[str, Any]]) -> None:
        """Batch process spatial composition (using scene_composer)"""
        self.logger.info("Processing spatial composition batch")

        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                style = spec.get("style")
                instances = [ObjectInstance.from_dict(d) for d in spec.get("instances", [])]
                environment_details = spec.get("environment_details", {})

                # Call scene_composer to generate spatial and semantic relations
                composition_data = self.scene_composer.create_complete_composition(
                    instances,
                    style=style,
                    environment_context=environment_details,
                    artwork_id=artwork_id
                )

                spec["spatial_relationships_data"] = composition_data
                spec["spatial_relationships_status"] = "success"

                self.logger.info(f"Composition for {artwork_id}: "
                               f"{len(composition_data.get('spatial_relations', []))} spatial + "
                               f"{len(composition_data.get('semantic_relations', []))} semantic relations")

            except Exception as e:
                self.logger.error(f"Spatial composition failed for {artwork_id}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                spec["spatial_relationships_data"] = {}
                spec["spatial_relationships_status"] = "failed"
                spec["spatial_relationships_error"] = str(e)

    def _process_object_enhancement_batch(self, artwork_specs: List[Dict[str, Any]]) -> None:
        """Batch process object enhancement"""
        self.logger.info("Processing object enhancement batch")
        
        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                instances = [ObjectInstance.from_dict(d) for d in spec.get("instances", [])]
                style = spec.get("style")

                # Get environment details as scene context
                scene_context = spec.get("environment_details", {})

                # Get detailed output
                enhancement_result = self.object_enhancer.enhance_objects(
                    instances, style, environment=scene_context, return_details=True
                )

                if isinstance(enhancement_result, dict) and "enhanced_objects" in enhancement_result:
                    # Get detailed information
                    instances = enhancement_result["instances"]
                    spec["instances"] = [i.to_dict() for i in instances]
                    enhanced_objects = enhancement_result["enhanced_objects"]
                    raw_output = enhancement_result.get("raw_output", "")
                    parsing_details = enhancement_result.get("parsing_details", {})
                    user_prompt = enhancement_result.get("user_prompt", "")
                    system_prompt = enhancement_result.get("system_prompt", "")
                    
                    spec["enhanced_objects"] = enhanced_objects
                    spec["object_enhancement_status"] = "success"
                    
                    # Save detailed output
                    if self.current_output_manager:
                        try:
                            processing_info = {
                                "stage": "object_enhancement",
                                "artwork_id": artwork_id,
                                "objects": spec.get("objects", []),
                                "style": style,
                                "scene_context": scene_context,
                                "system_prompt": system_prompt,
                                "user_prompt": user_prompt,
                                "parsing_details": parsing_details
                            }
                            
                            clean_result = {"enhanced_objects": enhanced_objects}
                            
                            self.current_output_manager.save_stage_output(
                                stage="object_enhancement",
                                artwork_id=artwork_id,
                                raw_output=raw_output,
                                parsed_result=clean_result,
                                processing_info=processing_info
                            )
                            
                            self.logger.info(f"Saved detailed object enhancement output for {artwork_id}")
                            
                        except Exception as e:
                            self.logger.warning(f"Failed to save object enhancement output for {artwork_id}: {e}")
                else:
                    # enhance_objects returned a plain list
                    spec["enhanced_objects"] = enhancement_result if isinstance(enhancement_result, list) else []
                    spec["object_enhancement_status"] = "success"
                
            except Exception as e:
                self.logger.error(f"Object enhancement failed for {artwork_id}: {e}")
                spec["enhanced_objects"] = []
                spec["object_enhancement_status"] = "failed"
                spec["object_enhancement_error"] = str(e)
                
                # Save failed detailed output
                if self.current_output_manager:
                    try:
                        failed_result = {"error": str(e), "enhanced_objects": []}
                        processing_info = {
                            "stage": "object_enhancement",
                            "artwork_id": artwork_id,
                            "objects": spec.get("objects", []),
                            "style": spec.get("style", "Abstract"),
                            "parsing_details": {"status": "failed", "parsing_error": str(e)}
                        }
                        
                        self.current_output_manager.save_stage_output(
                            stage="object_enhancement",
                            artwork_id=artwork_id,
                            raw_output=f"Error: {str(e)}",
                            parsed_result=failed_result,
                            processing_info=processing_info
                        )
                        
                        self.logger.info(f"Saved failed object enhancement output for {artwork_id}")
                        
                    except Exception as save_e:
                        self.logger.warning(f"Failed to save failed object enhancement output for {artwork_id}: {save_e}")
    
    def _finalize_batch_artworks(self, artwork_specs: List[Dict[str, Any]], 
                               output_dir: str) -> List[Dict[str, Any]]:
        """Generate prompts for and save each artwork that passed every stage."""
        self.logger.info("Finalizing batch artworks")
        
        final_artworks = []
        
        dropped = 0
        for spec in artwork_specs:
            try:
                # Collect all data
                artwork_id = spec.get("artwork_id", generate_artwork_id())

                # An artwork with a failed stage is discarded.
                failed_stages = [
                    stage for stage, status in (
                        ("object_enhancement", spec.get("object_enhancement_status")),
                        ("composition", spec.get("spatial_relationships_status")),
                        ("environment_details", spec.get("environment_details_status")),
                    ) if status == "failed"
                ]
                if failed_stages:
                    dropped += 1
                    self.logger.warning(f"Discarding {artwork_id}: failed at {failed_stages}")
                    continue

                style = spec.get("style", "Abstract")
                object_names = spec.get("objects", [])
                enhanced_objects = spec.get("enhanced_objects", [])

                self.logger.info(f"Finalizing {artwork_id}")

                # Build final data structure
                final_data = {
                    "artwork_id": artwork_id,
                    "generation_timestamp": get_timestamp(),
                    "style": style,
                    "objects": {
                        # enhanced_objects is the record readers use; instances
                        # is a compact two-layer index (see to_instance_index).
                        "instances": to_instance_index(spec.get("instances", [])),
                        "object_ids": spec.get("object_ids", []),
                        "object_names": object_names,
                        "enhanced_objects": enhanced_objects
                    },
                    "composition": spec.get("spatial_relationships_data", {}),
                    "environment": {
                        "environment_details": spec.get("environment_details", {}),
                        "scene_inference": spec.get("scene_inference", {})
                    },
                    "processing_status": {
                        "compatibility": spec.get("compatibility_status", "unknown"),
                        "composition": spec.get("spatial_relationships_status", "unknown"),
                        "environment_details": spec.get("environment_details_status", "unknown"),
                        "object_enhancement": spec.get("object_enhancement_status", "unknown")
                    },
                    "metadata": {
                        
                        "batch_mode": self.batch_mode,
                        "max_secondary_objects": spec.get("max_secondary_objects", 8),
                        "model_info": self.vllm_wrapper.get_model_info() if self.vllm_wrapper else {},
                        "generation_index": spec.get("index", 0)
                    }
                }

                # A PromptGenerationError reaches the handler below: the artwork
                # is reported as an error and not saved.
                final_prompt_package = self.prompt_generator.create_final_prompt_package(
                    enhanced_objects,
                    style,
                    final_data["composition"],
                    final_data["environment"]
                )
                final_data["final_prompts"] = final_prompt_package
                self.logger.info(f"Final prompts generated successfully for {artwork_id}")

                # Save the artwork JSON to final_results/
                final_results_dir = os.path.join(output_dir, "final_results")
                filename = os.path.join(final_results_dir, f"{artwork_id}.json")
                save_json(final_data, filename)
                

                
                final_artworks.append(final_data)
                
            except Exception as e:
                self.logger.error(f"Failed to finalize artwork {spec.get('artwork_id', 'unknown')}: {e}")
                error_data = {
                    "artwork_id": spec.get("artwork_id", "error_artwork"),
                    "error": str(e),
                    "spec": spec
                }
                final_artworks.append(error_data)

        if dropped:
            self.logger.warning(
                f"Discarded {dropped}/{len(artwork_specs)} artworks with a failed stage"
            )

        return final_artworks
    
    def cleanup(self) -> None:
        """Clean up pipeline resources"""
        self.logger.info("Cleaning up pipeline resources...")
        
        if self.vllm_manager:
            self.vllm_manager.cleanup()
        
        # Reset state
        self.is_initialized = False
        self.vllm_wrapper = None
        self.style_manager = None
        self.object_selector = None
        self.object_enhancer = None
        self.scene_composer = None
        self.environment_designer = None
        self.prompt_generator = None
        self.batch_processor = None

        
        self.logger.info("Pipeline cleanup completed")
    

    

    

    
    def _compute_stage_statistics(self, artwork_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Count per-stage statuses; anything other than success/compatible counts as failed."""
        stage_stats: Dict[str, Dict[str, int]] = {}
        for spec in artwork_specs:
            for key, val in spec.items():
                if key.endswith("_status"):
                    stage = key[: -len("_status")]
                    entry = stage_stats.setdefault(stage, {"success": 0, "failed": 0})
                    if val in ("success", "compatible"):
                        entry["success"] += 1
                    else:
                        entry["failed"] += 1
        return stage_stats

    def validate_pipeline_integrity(self) -> Dict[str, Any]:
        """Check that the pipeline and all its components are ready."""
        validation = {
            "is_valid": True,
            "issues": [],
            "component_checks": {}
        }
        
        # Check initialization status
        if not self.is_initialized:
            validation["is_valid"] = False
            validation["issues"].append("Pipeline not initialized")
            return validation
        
        # Check components
        components = {
            "vllm_wrapper": self.vllm_wrapper,
            "style_manager": self.style_manager,
            "object_selector": self.object_selector,
            "object_enhancer": self.object_enhancer,
            "scene_composer": self.scene_composer,
            "environment_designer": self.environment_designer,
            "prompt_generator": self.prompt_generator
        }
        
        if self.batch_mode:
            components["batch_processor"] = self.batch_processor
        
        for name, component in components.items():
            if component is None:
                validation["is_valid"] = False
                validation["issues"].append(f"Component {name} not initialized")
                validation["component_checks"][name] = False
            else:
                validation["component_checks"][name] = True
        
        # Check vLLM model status
        if self.vllm_wrapper and hasattr(self.vllm_wrapper, 'model'):
            if self.vllm_wrapper.model is None:
                validation["is_valid"] = False
                validation["issues"].append("vLLM model not loaded")
        
        return validation


# Convenience functions
def create_pipeline(batch_mode: bool = True, output_dir: str = "data/output",
                    use_fallback_llm: bool = False,
                    allow_llm_fallback: bool = False) -> ArtworkImagePipeline:
    """Create and initialize pipeline"""
    pipeline = ArtworkImagePipeline(batch_mode=batch_mode, output_base_dir=output_dir,
                                    use_fallback_llm=use_fallback_llm,
                                    allow_llm_fallback=allow_llm_fallback)
    
    if pipeline.initialize():
        return pipeline
    else:
        raise RuntimeError("Failed to initialize pipeline")


def quick_generate(count: int = 1, style: Optional[str] = None, 
                  batch_mode: bool = True) -> List[Dict[str, Any]]:
    """Quick generation function"""
    pipeline = create_pipeline(batch_mode=batch_mode)
    
    try:
        if count == 1:
            result = pipeline.generate_single_artwork(style=style)
            return [result]
        else:
            return pipeline.generate_batch_artworks(count=count, styles=[style] if style else None)
    finally:
        pipeline.cleanup()
        
        
        