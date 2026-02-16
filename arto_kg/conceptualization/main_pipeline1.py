"""
main pipeline for artwork generation
This module provides the main pipeline for generating artworks using various components.
It includes initialization, single and batch generation, and utility functions.
"""

import os
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .utils import setup_logger, save_json, generate_artwork_id, get_timestamp, print_progress, create_batch_output_structure
from .llm_manager import VLLMManager
from .style_selector import StyleSelector
from .object_selector import ObjectSelector
from .object_enhancer import ObjectEnhancer
from .scene_composer import SceneComposer
from .environment_designer import EnvironmentDesigner
from .prompt_generator import PromptGenerator
from .batch_processor import BatchProcessor



class ArtworkImagePipeline:
    """Optimized artwork generation pipeline"""
    
    def __init__(self, batch_mode: bool = True, output_base_dir: str = "data/output"):
        self.logger = setup_logger("artwork_pipeline")
        self.batch_mode = batch_mode
        self.output_base_dir = output_base_dir
        
        # Output manager (created during batch generation)
        self.current_batch_dir = None
        self.current_output_manager = None
        
        # Initialize components
        self.logger.info("Initializing artwork pipeline...")
        
        # vLLM manager
        self.vllm_manager = VLLMManager()
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
        """
        Initialize pipeline
        
        Returns:
            Whether initialization was successful
        """
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
                self.batch_processor = BatchProcessor(self.vllm_wrapper, self.object_selector)
            

            
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
            
            # Detailed vLLM status check
            vllm_status = self._debug_check_vllm_status()
            if not vllm_status:
                self.logger.error("❌ vLLM status check failed! This may cause composition and environment generation issues.")
            
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
        """
        Generate single artwork
        
        Args:
            style: Specified style
            max_secondary_objects: Maximum number of secondary objects
            output_dir: Output directory
            
        Returns:
            Generated artwork data
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
            selected_objects = selection_result['object_ids']
            # Add fallback for object_names if missing
            object_names = selection_result.get('object_names', [])
            if not object_names:
                object_names = self.object_selector.get_object_names(selected_objects)
            scene_inference = selection_result.get('scene_inference', {})
            self.logger.info(f"Selected objects: {object_names}")
            self.logger.info(f"Scene inference: {scene_inference.get('selected_scene', {}).get('scene_type', 'N/A')}")

            # step 3: determine environment details (using scene inference)

            selected_scene = scene_inference.get('selected_scene', {})
            # Prioritize detailed scene_brief, fallback to scene_type
            scene_brief = selected_scene.get('scene_brief', selected_scene.get('scene_type', 'abstract scene'))

            environment_details = self.environment_designer.determine_environment_details(
                scene_brief=scene_brief,
                style=selected_style
            )

            # Add object_contexts from scene_inference to environment_details
            if 'object_contexts' in scene_inference:
                environment_details['object_contexts'] = scene_inference['object_contexts']


            self.logger.info("Determined environment details from scene inference")

            # step 4: enhance object visual descriptions (using environment details)
            enhanced_objects = self.object_enhancer.enhance_objects(
                selected_objects,
                selected_style,
                scene_context=environment_details  # Pass full environment details
            )
            self.logger.info(f"Enhanced {len(enhanced_objects)} objects with environment-aware descriptions")

            # step 5: create complete spatial composition (with semantic relations)


            composition_data = self.scene_composer.create_complete_composition(
                enhanced_objects,
                selected_style,
                environment_context=environment_details,
                artwork_id=artwork_id if use_detailed_output else None
            )


            self._debug_check_composition_data(composition_data, artwork_id)
            self.logger.info("Created complete spatial composition")

            # Package environment data for final output
            environment_data = {
                "environment_details": environment_details,
                "scene_inference": scene_inference
            }


            self._debug_check_environment_data(environment_data, artwork_id)

            # step 6: generate final prompts (removed artistic_expression generation)
            self.logger.info("Skipping artistic expression generation as requested")
            
            # Generate final prompt package
            try:
                final_prompt_package = self.prompt_generator.create_final_prompt_package(
                    enhanced_objects, selected_style, composition_data, environment_data
                )
            except Exception as e:
                self.logger.warning(f"Failed to create final prompts: {e}")
                # Create basic final_prompts structure
                final_prompt_package = {
                    "main_prompt": f"{selected_style} artwork with {len(enhanced_objects)} objects",
                    "negative_prompt": "blurry, low quality",
                    "prompt_formats": {
                        "simple": f"{selected_style} artwork",
                        "complex": f"detailed {selected_style} artwork", 
                        "comma_separated": f"{selected_style.lower()}, artwork, high quality",
                        "tags": f"#{selected_style.replace(' ', '')} #artwork"
                    },
                    "objects_count": len(enhanced_objects),
                    "style": selected_style,
                    "format_count": 4,
                    "error": str(e)
                }
            
            self.logger.info("Generated final artistic expression and prompt package")

            # assemble final data
            generation_time = time.time() - start_time
            
            final_data = {
                "artwork_id": artwork_id,
                "generation_timestamp": get_timestamp(),
                "generation_time": generation_time,
                "style": selected_style,
                "objects": {
                    "object_ids": selected_objects,
                    "object_names": object_names,
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
            
            # Save results
            if use_detailed_output:
                # Save to final_results folder
                final_results_dir = os.path.join(output_dir, "final_results")
                filename = os.path.join(final_results_dir, f"{artwork_id}.json")
            else:
                # Save to traditional location
                filename = os.path.join(output_dir, f"{artwork_id}.json")
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
        """
        Generate batch artworks
        
        Args:
            count: Number of artworks to generate
            styles: Optional list of styles
            max_secondary_objects: Maximum number of secondary objects
            output_dir: Output directory
            save_intermediate: Whether to save intermediate results
            
        Returns:
            List of generated artwork data
        """
        if not self.is_initialized:
            self.logger.error("Pipeline not initialized")
            return []
        
        if not self.batch_mode or not self.batch_processor:
            self.logger.warning("Batch mode not available, falling back to sequential generation")
            return self._generate_sequential_batch(count, styles, max_secondary_objects, output_dir, True)
        
        # Create new batch output structure
        if output_dir is None:
            self.current_batch_dir, self.current_output_manager = create_batch_output_structure(self.output_base_dir)
            output_dir = self.current_batch_dir
        else:
            # If output directory is specified, still use the new structure
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
            print_progress(0, 6, "Pipeline Progress")

            artwork_specs = self.batch_processor.process_compatibility_batch(artwork_specs)


            # Step 3: Batch environment refinement (based on scene inference)
            print_progress(1, 6, "Pipeline Progress")
            self.logger.info("Processing environment details based on scene inference")
            self._process_environment_details_batch(artwork_specs)

            # Step 4: Object enhancement (based on environment details)
            print_progress(2, 6, "Pipeline Progress")
            self._process_object_enhancement_batch(artwork_specs)

            # Step 5: Batch spatial composition (using scene_composer)
            print_progress(3, 6, "Pipeline Progress")
            print_progress(4, 6, "Pipeline Progress")
            self.logger.info("Processing spatial composition batch")
            self._process_spatial_composition_batch(artwork_specs)

            # Save intermediate results
            if save_intermediate:
                intermediate_file = os.path.join(output_dir, f"intermediate_spatial_relationships.json")
                save_json({"stage": "spatial_relationships", "data": artwork_specs}, intermediate_file)

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
                "failed": len(final_artworks) - success_count,
                "success_rate": success_count / len(final_artworks) if final_artworks else 0,
                "generation_time": batch_time,
                "average_time_per_artwork": batch_time / len(final_artworks) if final_artworks else 0,
                "styles_used": list(set(art.get("style", "") for art in final_artworks if "style" in art)),
                "output_directory": output_dir,
                "batch_statistics": self.batch_processor.get_batch_statistics([])  # Need to pass actual results
            }
            
            # Save batch summary to final results folder
            final_results_dir = os.path.join(output_dir, "final_results")
            save_json(batch_summary, os.path.join(final_results_dir, "batch_summary.json"))
            
            # Also save to root directory for compatibility  
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
    
    def _get_stage_system_prompts(self) -> Dict[str, str]:
        """Get system prompts for each stage - simplified version"""
        return {
            # Simplified composition system prompts
            "spatial_relationships": self.scene_composer.spatial_relationship_system_prompt,
            "simple_prompt": self.prompt_generator.simple_prompt_system_prompt,
            
            # Environment and artistic expression
            "environment_details": self.environment_designer.environment_system_prompt,
            "artistic_expression": self.prompt_generator.artistic_expression_system_prompt
        }
    
    def _process_environment_details_batch(self, artwork_specs: List[Dict[str, Any]]) -> None:
        """Batch process environment details (based on scene inference results)"""
        self.logger.info("Processing environment details batch")

        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                style = spec.get("style", "Abstract")
                scene_inference = spec.get("scene_inference", {})
                selected_scene = scene_inference.get("selected_scene", {})
                # Prioritize detailed scene_brief, fallback to scene_type
                scene_brief = selected_scene.get("scene_brief", selected_scene.get("scene_type", "abstract scene"))

                # Determine environment details
                environment_details = self.environment_designer.determine_environment_details(
                    scene_brief=scene_brief,
                    style=style
                )

                # Add object context information
                if 'object_contexts' in scene_inference:
                    environment_details['object_contexts'] = scene_inference['object_contexts']

                spec["environment_details"] = environment_details
                spec["environment_details_status"] = "success"

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
                style = spec.get("style", "Abstract")
                enhanced_objects = spec.get("enhanced_objects", [])
                environment_details = spec.get("environment_details", {})

                # Call scene_composer to generate spatial and semantic relations
                composition_data = self.scene_composer.create_complete_composition(
                    enhanced_objects=enhanced_objects,
                    style=style,
                    environment_context=environment_details,
                    artwork_id=artwork_id
                )

                
                self.logger.info(f"[SPATIAL_COMP]   Composition data keys: {list(composition_data.keys())}")
                self.logger.info(f"[SPATIAL_COMP]   Primary focus: {composition_data.get('primary_focus', 'N/A')}")



                spec["spatial_relationships_data"] = composition_data
                # spec["spatial_relationships_data"] = composition_data.get("spatial_relations", {})
                spec["spatial_relationships_status"] = "success"

                self.logger.info(f"[SPATIAL_COMP] ✓ Success for {artwork_id}: "
                               f"{len(composition_data.get('spatial_relations', []))} spatial + "
                               f"{len(composition_data.get('semantic_relations', []))} semantic relations")
                 
                # self.logger.info(f"Spatial composition created for {artwork_id}: "
                #                f"{len(composition_data.get('spatial_relations', []))} spatial relations, "
                #                f"{len(composition_data.get('semantic_relations', []))} semantic relations")
                # self.logger.info(f"Spatial composition created for {artwork_id}: "
                #                f"{composition_data.get('spatial_relations_count', 0)} spatial relations, "
                #                f"{composition_data.get('semantic_relations_count', 0)} semantic relations")

            except Exception as e:
                self.logger.error(f"Spatial composition failed for {artwork_id}: {e}")
                import traceback
                self.logger.error(f"[SPATIAL_COMP] Traceback: {traceback.format_exc()}")
                spec["spatial_relationships_data"] = {}
                spec["spatial_relationships_status"] = "failed"
                spec["spatial_relationships_error"] = str(e)

    def _process_object_enhancement_batch(self, artwork_specs: List[Dict[str, Any]]) -> None:
        """Batch process object enhancement"""
        self.logger.info("Processing object enhancement batch")
        
        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                object_ids = spec.get("object_ids", [])
                style = spec.get("style", "Abstract")

                # Get environment details as scene context
                scene_context = spec.get("environment_details", {})
                
                # Get detailed output
                enhancement_result = self.object_enhancer.enhance_objects(
                    object_ids, style, scene_context, return_details=True
                )
                
                if isinstance(enhancement_result, dict) and "enhanced_objects" in enhancement_result:
                    # Get detailed information
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
                    # Backward compatibility, if listing is returned
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
    
    def _process_color_scheme_batch(self, artwork_specs: List[Dict[str, Any]]) -> None:
        """Batch process color scheme generation"""
        self.logger.info("Processing color scheme batch")
        
        for spec in artwork_specs:
            artwork_id = spec.get("artwork_id", "unknown")
            try:
                # Get necessary data
                style = spec.get("style", "Abstract")
                object_names = spec.get("objects", [])
                enhanced_objects = spec.get("enhanced_objects", [])
                environment_details = spec.get("environment_details_data", {})
                
                # Use environment designer to generate color scheme
                if environment_details:
                    color_scheme = self.environment_designer.design_color_scheme(
                        object_names, style, environment_details, enhanced_objects
                    )
                    
                    # Add color scheme to spec
                    spec["color_scheme_data"] = color_scheme
                    spec["color_scheme_status"] = "success"
                    
                    self.logger.info(f"Generated color scheme for {artwork_id}: {color_scheme.get('main_palette', {}).get('overall_tone', 'undefined')} tone")
                    
                else:
                    # Create fallback color scheme
                    color_scheme = self.environment_designer._create_fallback_color_scheme(style, {})
                    spec["color_scheme_data"] = color_scheme
                    spec["color_scheme_status"] = "fallback"
                    
                    self.logger.warning(f"Using fallback color scheme for {artwork_id}")
                    
            except Exception as e:
                self.logger.error(f"Color scheme generation failed for {artwork_id}: {e}")
                # Create empty color scheme
                spec["color_scheme_data"] = {}
                spec["color_scheme_status"] = "failed"
                spec["color_scheme_error"] = str(e)
    
    def _finalize_batch_artworks(self, artwork_specs: List[Dict[str, Any]], 
                               output_dir: str) -> List[Dict[str, Any]]:
        """Finalize batch artworks"""
        self.logger.info("Finalizing batch artworks")
        
        final_artworks = []
        
        for spec in artwork_specs:
            try:
                # Collect all data
                artwork_id = spec.get("artwork_id", generate_artwork_id())
                style = spec.get("style", "Abstract")
                object_names = spec.get("objects", [])
                enhanced_objects = spec.get("enhanced_objects", [])
                
                
                self.logger.info(f"[FINALIZE] Processing {artwork_id}")
                self.logger.info(f"[FINALIZE]   spatial_relationships_data keys: {list(spec.get('spatial_relationships_data', {}).keys())}")
                self.logger.info(f"[FINALIZE]   spatial_relationships_status: {spec.get('spatial_relationships_status', 'N/A')}")

                # Build final data structure
                final_data = {
                    "artwork_id": artwork_id,
                    "generation_timestamp": get_timestamp(),
                    "style": style,
                    "objects": {
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
                
                # Debug check final data
                self.logger.info(f"[DEBUG FINAL] Checking final data for {artwork_id}")
                self._debug_check_composition_data(final_data.get("composition", {}), artwork_id)
                self._debug_check_environment_data(final_data.get("environment", {}), artwork_id)
                
                # Generate final prompt package (always try to generate)
                try:
                    final_prompt_package = self.prompt_generator.create_final_prompt_package(
                        enhanced_objects,
                        style,
                        final_data["composition"],
                        final_data["environment"]
                    )
                    final_data["final_prompts"] = final_prompt_package
                    self.logger.info(f"[DEBUG FINAL] ✅ Final prompts generated successfully for {artwork_id}")
                except Exception as e:
                    self.logger.warning(f"[DEBUG FINAL] ❌ Failed to create final prompts for {artwork_id}: {e}")
                    # Create basic final_prompts structure
                    final_data["final_prompts"] = {
                        "main_prompt": f"{style} artwork with {len(enhanced_objects)} objects",
                        "negative_prompt": "blurry, low quality",
                        "prompt_formats": {
                            "simple": f"{style} artwork",
                            "complex": f"detailed {style} artwork",
                            "comma_separated": f"{style.lower()}, artwork, high quality",
                            "tags": f"#{style.replace(' ', '')} #artwork"
                        },
                        "objects_count": len(enhanced_objects),
                        "style": style,
                        "format_count": 4,
                        "error": str(e)
                    }
                
                # Save single artwork folder
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
        
        return final_artworks
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get pipeline status information
        
        Returns:
            Pipeline status dictionary
        """
        status = {
            "is_initialized": self.is_initialized,
            "initialization_time": self.initialization_time,
            "batch_mode": self.batch_mode,
            "output_base_dir": self.output_base_dir,
            "components_status": {
                "vllm_manager": self.vllm_manager is not None,
                "vllm_wrapper": self.vllm_wrapper is not None,
                "style_manager": self.style_manager is not None,
                "object_selector": self.object_selector is not None,
                "object_enhancer": self.object_enhancer is not None,
                "scene_composer": self.scene_composer is not None,
                "environment_designer": self.environment_designer is not None,
                "prompt_generator": self.prompt_generator is not None,
                "batch_processor": self.batch_processor is not None,

            }
        }
        
        if self.vllm_wrapper:
            status["model_info"] = self.vllm_wrapper.get_model_info()
        
        if self.style_manager:
            status["available_styles"] = len(self.style_manager.get_available_styles())
        
        return status
    
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
    
    def _debug_check_composition_data(self, composition_data: Dict[str, Any], artwork_id: str = None):
        """Check integrity of composition data and LLM output"""
        self.logger.info(f"[DEBUG COMPOSITION] Checking composition data for artwork {artwork_id}")
        
        if not composition_data:
            self.logger.error(f"[DEBUG COMPOSITION] ❌ CRITICAL: composition_data is completely empty!")
            return
        
        # Check spatial_relationships
        # Check spatial_relations (Flat structure check)
        spatial_rel_list = composition_data.get("spatial_relations", [])
        if not spatial_rel_list:
            self.logger.error(f"[DEBUG COMPOSITION] ❌ CRITICAL: spatial_relations list is empty!")
        else:
            self.logger.info(f"[DEBUG COMPOSITION] ✅ spatial_relations exists with {len(spatial_rel_list)} items")
            
            # Check LLM raw output (Usually at root in flat structure)
            llm_output = composition_data.get("llm_raw_output", "") 
            if not llm_output:
                self.logger.warning(f"[DEBUG COMPOSITION] ⚠️ No LLM raw output found at root.")
            else:
                self.logger.info(f"[DEBUG COMPOSITION] ✅ LLM raw output exists, length: {len(llm_output)} chars")
            
            # Check debug info
            debug_info = composition_data.get("debug_info", {})
            if debug_info:
                self.logger.info(f"[DEBUG COMPOSITION] Debug info: {debug_info}")
                if not debug_info.get("vllm_success", True):
                    self.logger.error(f"[DEBUG COMPOSITION] ❌ vLLM call failed: {debug_info.get('error', 'unknown')}")
            else:
                self.logger.warning(f"[DEBUG COMPOSITION] ⚠️ No debug info available")
        
        # Check other fields - use new stage names
        # Check other fields - use new stage names (Updated for flat structure)
        for field in ["spatial_relations", "semantic_relations", "primary_focus"]:
            if field in composition_data:
                field_data = composition_data[field]
                if field_data:
                    self.logger.info(f"[DEBUG COMPOSITION] ✅ {field} exists with data")
                else:
                    self.logger.warning(f"[DEBUG COMPOSITION] ⚠️ {field} is empty")
            else:
                self.logger.warning(f"[DEBUG COMPOSITION] ⚠️ {field} is missing")
    
    def _debug_check_environment_data(self, environment_data: Dict[str, Any], artwork_id: str = None):
        """Check integrity of environment data and LLM output"""
        self.logger.info(f"[DEBUG ENVIRONMENT] Checking environment data for artwork {artwork_id}")
        
        if not environment_data:
            self.logger.error(f"[DEBUG ENVIRONMENT] ❌ CRITICAL: environment_data is completely empty!")
            return
        
        # Check environment_details
        env_details = environment_data.get("environment_details", {})
        if not env_details:
            self.logger.error(f"[DEBUG ENVIRONMENT] ❌ CRITICAL: environment_details is empty!")
        else:
            self.logger.info(f"[DEBUG ENVIRONMENT] ✅ environment_details exists with {len(env_details)} fields")
            
            # Check required fields (new format)
            required_fields = ["scene_brief", "time_of_day", "lighting"]
            missing_fields = [field for field in required_fields if not env_details.get(field)]
            if missing_fields:
                self.logger.error(f"[DEBUG ENVIRONMENT] ❌ Missing required fields: {missing_fields}")
            else:
                self.logger.info(f"[DEBUG ENVIRONMENT] ✅ All required fields present")
            
            # Check debug info
            debug_info = env_details.get("debug_info", {})
            if debug_info:
                self.logger.info(f"[DEBUG ENVIRONMENT] Debug info: {debug_info}")
                if not debug_info.get("vllm_success", True):
                    self.logger.error(f"[DEBUG ENVIRONMENT] ❌ vLLM call failed: {debug_info.get('error', 'unknown')}")
            else:
                self.logger.warning(f"[DEBUG ENVIRONMENT] ⚠️ No debug info available")
        
        # Check color_scheme
        color_scheme = environment_data.get("color_scheme", {})
        if not color_scheme:
            self.logger.warning(f"[DEBUG ENVIRONMENT] ⚠️ color_scheme is empty")
        else:
            self.logger.info(f"[DEBUG ENVIRONMENT] ✅ color_scheme exists with {len(color_scheme)} fields")
    
    def _debug_check_vllm_status(self):
        """Check vLLM status"""
        self.logger.info("[DEBUG VLLM] Checking vLLM status...")
        
        if not self.vllm_wrapper:
            self.logger.error("[DEBUG VLLM] ❌ CRITICAL: vllm_wrapper is None!")
            return False
        
        try:
            # Test simple model call
            test_result = self.vllm_wrapper.test_model_response()
            # Check for error in response
            # Note: Do not check string representation as it may contain keys like 'parsing_error': None
            has_error = False
            
            if not isinstance(test_result, dict):
                has_error = True
            elif "error" in test_result:
                has_error = True
            elif "greeting" not in test_result:
                # If we don't see the expected output but no explicit error, 
                # check if parsing details failed
                parsing_details = test_result.get("parsing_details", {})
                if parsing_details.get("status") == "failed":
                    has_error = True
            
            if has_error:
                self.logger.error(f"[DEBUG VLLM] ❌ Model test failed: {test_result}")
                return False
            else:
                self.logger.info(f"[DEBUG VLLM] ✅ Model test successful: {test_result}")
                return True
        except Exception as e:
            self.logger.error(f"[DEBUG VLLM] ❌ Exception during model test: {e}")
            import traceback
            self.logger.error(f"[DEBUG VLLM] Full traceback: {traceback.format_exc()}")
            return False
    
    def resume_batch_from_intermediate(self, intermediate_file: str, 
                                     output_dir: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Resume batch generation from intermediate results
        
        Args:
            intermediate_file: Intermediate result file path
            output_dir: Output directory
            
        Returns:
            List of completed artworks
        """
        if not self.is_initialized:
            self.logger.error("Pipeline not initialized")
            return []
        
        try:
            # Load intermediate results
            from arto_kg.conceptualization.utils import load_json
            intermediate_data = load_json(intermediate_file)
            
            stage = intermediate_data.get("stage", "unknown")
            artwork_specs = intermediate_data.get("data", [])
            
            self.logger.info(f"Resuming batch from stage: {stage}")
            self.logger.info(f"Found {len(artwork_specs)} artwork specifications")
            
            if output_dir is None:
                output_dir = f"{self.output_base_dir}/artworks/resumed_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Continue processing based on stage
            remaining_stages = self._get_remaining_stages(stage)
            
            for remaining_stage in remaining_stages:
                self.logger.info(f"Processing remaining stage: {remaining_stage}")
                stage_prompts = self._get_stage_system_prompts()
                
                artwork_specs = self.batch_processor.process_pipeline_stage(
                    artwork_specs, remaining_stage, stage_prompts[remaining_stage]
                )
            
            # Finish processing
            self._process_object_enhancement_batch(artwork_specs)
            final_artworks = self._finalize_batch_artworks(artwork_specs, output_dir)
            
            self.logger.info(f"Resumed batch completed: {len(final_artworks)} artworks")
            return final_artworks
            
        except Exception as e:
            self.logger.error(f"Failed to resume batch: {e}")
            return []
    
    def _get_remaining_stages(self, completed_stage: str) -> List[str]:
        """Get remaining stages to process"""
        all_stages = ["spatial_relationships", "environment_details"]
        
        try:
            current_index = all_stages.index(completed_stage)
            return all_stages[current_index + 1:]
        except ValueError:
            # If stage name is not in the list, return all stages
            return all_stages
    
    def validate_pipeline_integrity(self) -> Dict[str, Any]:
        """
        Validate pipeline integrity
        
        Returns:
            Validation result
        """
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
    
    def get_generation_estimate(self, count: int) -> Dict[str, Any]:
        """
        Get generation time estimate
        
        Args:
            count: Number to generate
            
        Returns:
            Time estimate information
        """
        if not self.batch_processor:
            # Sequential generation estimate
            estimated_time_per_artwork = 30  # seconds
            total_time = count * estimated_time_per_artwork
            
            return {
                "mode": "sequential",
                "estimated_total_time": total_time,
                "estimated_time_per_artwork": estimated_time_per_artwork,
                "estimated_completion": time.time() + total_time
            }
        else:
            # Batch generation estimate
            return self.batch_processor.estimate_processing_time(count)


# Convenience functions
def create_pipeline(batch_mode: bool = True, output_dir: str = "data/output") -> ArtworkImagePipeline:
    """Create and initialize pipeline"""
    pipeline = ArtworkImagePipeline(batch_mode=batch_mode, output_base_dir=output_dir)
    
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
        
        
        