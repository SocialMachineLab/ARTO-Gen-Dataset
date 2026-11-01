"""
Batch Processor - Orchestrate the entire validation pipeline
"""

import argparse
import functools
import os
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from .vllm_wrapper import VLLMWrapper, create_vllm_wrapper
from .detections import extract_detections, duplicate_keys
from . import instance_assignment as ia
from . import measurements
from . import result_envelope as env
from . import spatial_predicates as sp
from . import colour_temperature as ct
from .utils import (
    load_gt_data,
    extract_expected_colors,
    save_validation_result,
    write_json_atomic
)



class BatchProcessor:
    """Batch Validation Processor"""
    
    def __init__(self, vlm_wrapper: VLLMWrapper, batch_size: int = 10, enable_od: bool = True,
                 annotation_dir: str = "annotations"):
        """
        Initialize batch processor

        Args:
            vlm_wrapper: VLLMWrapper instance
            batch_size: Batch size
            enable_od: Enable automatic object detection
            annotation_dir: Scratch directory for box-marked images shown to the
                VLM; files are overwritten as the run proceeds.
        """
        # Refuse an uninitialised model, since every VLM call would fail.
        if vlm_wrapper is not None and not getattr(vlm_wrapper, "is_initialized", True):
            raise RuntimeError(
                "The validation VLM is not initialised. Check the model, or pass "
                "--no-vlm to run the checks that need no VLM.")
        self.vlm = vlm_wrapper
        self.batch_size = batch_size
        self.enable_od = enable_od
        self.annotation_dir = annotation_dir
        
        from .validators import (
            ObjectDetectionValidator,
            SizeValidator,
            SpatialValidator,
            AlignmentValidator,
            ColorValidator
        )
        
        self.object_detector = ObjectDetectionValidator() if enable_od else None
        self.color_validator = ColorValidator()
        self.size_validator = SizeValidator()
        self.spatial_validator = SpatialValidator(vlm_wrapper=vlm_wrapper)
        self.alignment_validator = AlignmentValidator(vlm_wrapper=vlm_wrapper)

        # Style gate uses the VLM; without one the gate is reported inactive.
        from .validators.vlm_style_validator import VLMStyleValidator
        from .validators.vlm_state_validator import VLMStateValidator
        self.style_validator = VLMStyleValidator(vlm_wrapper) if vlm_wrapper else None
        if self.style_validator is None:
            print("[BatchProcessor] No VLM wrapper: style gate is inactive.")

        # State is judged per instance, with the object marked on the image.
        self.vlm_state_validator = VLMStateValidator(vlm_wrapper) if vlm_wrapper else None

        # Semantic relations: directed check with both endpoints marked A and B.
        from .validators.vlm_semantic_validator import VLMSemanticValidator
        self.semantic_validator = VLMSemanticValidator(vlm_wrapper) if vlm_wrapper else None

        # VLM channel for the depth and part-whole relations geometry defers.
        from .validators.vlm_spatial_validator import VLMSpatialValidator
        self.vlm_spatial_validator = VLMSpatialValidator(vlm_wrapper) if vlm_wrapper else None

    # --- gates and steps ----------------------------------------------------

    def _run_style_gate(self, json_data: Dict[str, Any], image_path: str) -> Dict[str, Any]:
        """Five-way style classification. Argmax, so no threshold is involved."""
        expected = json_data.get('style') or json_data.get('metadata', {}).get('style')

        if self.style_validator is None:
            print("  ! no VLM: artwork not gated on style")
            return {
                'status': 'not_configured',
                'gate': 'skipped',
                'expected_style': expected,
                'reason': 'no VLM wrapper',
            }

        # A specification that declares no style cannot be checked against
        # one: no gate decision, and the artwork is counted as failed.
        if not (isinstance(expected, str) and expected.strip()):
            return {
                'status': 'no_declared_style',
                'gate': 'error',
                'expected_style': None,
                'reason': 'the specification declares no style',
            }

        result = self.style_validator.main_evaluation(json_data, image_path)

        # A failed call or an unparseable reply says nothing about the image:
        # no gate decision is made, and the caller saves no result, so the
        # artwork is classified again on the next run.
        if result.get('status') in ('error', 'unparseable'):
            result['gate'] = 'error'
            return result
        result['gate'] = 'passed' if result.get('matched') else 'failed'
        if not result.get('expected_style'):
            result['expected_style'] = expected
        return result

    def _add_scene_colour(self, colour_result: Dict[str, Any], json_data: Dict[str, Any],
                          image_path: str, measurement_block: Dict[str, Any]) -> None:
        """Scene-level colour: background sample and declared temperature.

        Background colour is K-means over the image outside the detected boxes;
        the declared temperature is verified against the whole-image palette.
        `harmony_type` and `contrast_level` are not validated.
        """
        scheme = self.color_validator._find_colour_scheme(json_data)

        boxes = [entry.get('bbox') for entry in measurement_block.get('objects', [])
                 if entry.get('bbox')]
        background = self.color_validator.extract_background_colors(image_path, boxes)

        colour_result['background'] = {
            'method': 'bbox_inverse_mask',
            'objects_removed': len(boxes),
            'clusters': background[:5],
            'measured': bool(background),
        }

        # Compare the declared background colour against the sample.
        declared_background = scheme.get('background_colour')
        if declared_background and background:
            comparison = self.color_validator.measure_colour_differences(
                background, [declared_background])
            colour_result['background']['declared'] = declared_background
            colour_result['background']['delta_e'] = comparison['mean_delta_e']
            # One colour was asked about, so `matched` is 0 or 1.
            colour_result['background']['matched'] = comparison['matched'] > 0
            colour_result['background']['unresolved'] = comparison['unresolved']
        elif declared_background:
            colour_result['background']['declared'] = declared_background
            colour_result['background']['delta_e'] = None
            colour_result['background']['status'] = 'background_not_sampled'

        # Declared temperature: `colour_temperature` at the top of the scheme or
        # under `color_harmony`, in either spelling.
        temperature = scheme.get('colour_temperature') or scheme.get('color_temperature')
        if not temperature:
            harmony = scheme.get('color_harmony')
            if isinstance(harmony, dict):
                temperature = (harmony.get('colour_temperature')
                               or harmony.get('color_temperature'))

        # Temperature is a whole-image property.
        palette = self.color_validator.extract_dominant_colors(image_path)
        colour_result['temperature'] = ct.verify(temperature, palette)
        colour_result['palette_clusters'] = self.color_validator.last_chosen_k

        colour_result['scheme_present'] = bool(scheme)

    def _run_spatial_validation(self, json_data: Dict[str, Any], image_path: str,
                                bboxes_dict: Dict[str, Any],
                                measurement_block: Dict[str, Any],
                                artwork_id: str) -> Dict[str, Any]:
        """Geometric check first, then the VLM for the relations it defers.

        Depth and part-whole relations are marked `deferred` by geometry and
        answered by the VLM channel.
        """
        result = self.spatial_validator.main_evaluation(
            json_data, image_path, bboxes_dict, visualize=False)

        verification = result.get('verification') or {}
        if not verification.get('deferred_to_vlm'):
            return result

        if self.vlm_spatial_validator is None:
            result['vlm_channel'] = {
                'status': 'not_checked',
                'reason': 'no VLM wrapper; depth relations stay unverified',
            }
            return result

        self.vlm_spatial_validator.work_dir = os.path.join(
            self.annotation_dir, artwork_id)
        vlm_result = self.vlm_spatial_validator.resolve_deferred(
            json_data, image_path, measurement_block, verification)
        result['vlm_channel'] = vlm_result

        # Merge the VLM answers and rescore across both channels.
        merged = sp.merge_vlm_results(verification, vlm_result.get('relations', []))
        result['verification'] = merged
        result['overall_score'] = merged['score']
        result['dimension_scores']['declared_relations'] = merged['score']
        result['dimension_scores']['geometric_channel'] = merged['geometric_score']
        result['dimension_scores']['vlm_channel'] = merged['vlm_score']
        result['evaluation_summary'].update({
            'checked': merged['checked'],
            'verified': merged['verified'],
            'deferred_to_vlm': merged['deferred_to_vlm'],
            'geometric_checked': merged['geometric_checked'],
            'vlm_checked': merged['vlm_checked'],
        })
        return result

    def _run_state_validation(self, json_data: Dict[str, Any], image_path: str,
                              bboxes_dict: Dict[str, Any],
                              measurement_block: Dict[str, Any],
                              artwork_id: str) -> Dict[str, Any]:
        """Per-instance state, with the object marked on the image."""
        if self.vlm_state_validator is None:
            # State validation requires a VLM.
            return {'status': 'not_checked', 'score': None, 'objects': [],
                    'reason': 'state validation needs a VLM wrapper'}

        # Scratch space for annotated images, one subdirectory per artwork.
        self.vlm_state_validator.work_dir = os.path.join(
            self.annotation_dir, artwork_id)
        return self.vlm_state_validator.main_evaluation(
            json_data, image_path, measurement_block)

    def _run_semantic_validation(self, json_data: Dict[str, Any], image_path: str,
                                 measurement_block: Dict[str, Any],
                                 artwork_id: str) -> Dict[str, Any]:
        """Verify the declared semantic relations with the VLM."""
        relations = (json_data.get('composition') or {}).get('semantic_relations') or []

        if not relations:
            # Semantic relations are optional; none declared is not a failure.
            return {'status': 'not_applicable', 'relations': [], 'score': None,
                    'reason': 'specification declares no semantic relations'}

        if self.semantic_validator is None:
            return {'status': 'not_checked', 'relations': relations, 'score': None,
                    'reason': 'no VLM wrapper; semantic relations need one'}

        self.semantic_validator.work_dir = os.path.join(self.annotation_dir, artwork_id)
        return self.semantic_validator.main_evaluation(
            json_data, image_path, measurement_block)

    # --- helpers ------------------------------------------------------------

    def _safe_palette(self, image_path: str, bbox: List[float]):
        """K-means palette for one box, or None if extraction fails."""
        try:
            return self.color_validator.extract_colors_from_region(image_path, bbox)
        except Exception as exc:
            print(f"  ! colour extraction failed for bbox {bbox}: {exc}")
            return None

    def _build_provenance(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Record the detection source and the VLM that produced this result."""
        provenance: Dict[str, Any] = {'bboxes_source': data.get('od_path')}
        if self.vlm is not None and hasattr(self.vlm, 'model_info'):
            provenance.update(self.vlm.model_info())
        return provenance

    def _save(self, full_result: Dict[str, Any], output_dir: str, artwork_id: str) -> bool:
        # A run without a VLM leaves the VLM checks out, so its result is
        # marked incomplete (see result_envelope.is_complete).
        if self.vlm is None:
            full_result["incomplete"] = ("run without a VLM: style, state, semantic relations, "
                                         "alignment and the VLM spatial channel are not checked")
        output_path_dir = os.path.join(output_dir, artwork_id)
        os.makedirs(output_path_dir, exist_ok=True)
        output_path = os.path.join(output_path_dir, f"{artwork_id}_full_validation.json")
        return save_validation_result(full_result, output_path)
    
    def process_file_list(self, 
                         file_list: List[str],
                         output_dir: str,
                         od_results_dir: str,
                         image_base_dir: str) -> Dict[str, Any]:
        """
        Process file list
        
        Args:
            file_list: List of GT file paths
            output_dir: Output directory
            od_results_dir: OD results directory
            image_base_dir: Image base directory
            
        Returns:
            Processing statistics
        """
        total_files = len(file_list)
        processed = 0
        skipped = 0
        failed = 0
        
        print(f"Starting batch processing: {total_files} files")
        print(f"Batch size: {self.batch_size}")
        print(f"Output dir: {output_dir}")
        
        start_time = time.time()
        
        # Process by batch
        for i in range(0, total_files, self.batch_size):
            batch_files = file_list[i:i+self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (total_files + self.batch_size - 1) // self.batch_size
            
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
            try:
                batch_results = self._process_batch(
                    batch_files,
                    output_dir,
                    od_results_dir,
                    image_base_dir
                )
                
                processed += batch_results['processed']
                skipped += batch_results['skipped']
                failed += batch_results['failed']
                
            except Exception as e:
                print(f"Batch {batch_num} failed: {e}")
                failed += len(batch_files)
        
        elapsed = time.time() - start_time
        
        print(f"Batch processing completed.")
        print(f"Total: {total_files}")
        print(f"Processed: {processed}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Time: {elapsed:.1f}s ({elapsed/total_files:.2f}s/file)")
        
        return {
            'total': total_files,
            'processed': processed,
            'skipped': skipped,
            'failed': failed,
            'elapsed_time': elapsed,
            'avg_time_per_file': elapsed / total_files if total_files > 0 else 0
        }
    
    def _process_batch(self,
                      batch_files: List[str],
                      output_dir: str,
                      od_results_dir: str,
                      image_base_dir: str) -> Dict[str, int]:
        """Process a batch"""
        
        # Prepare batch data
        batch_data = []
        skipped_count = 0
        # A file that cannot be prepared (spec, image or detections missing or
        # unreadable) is counted as a failure.
        failed_count = 0

        for gt_path in batch_files:
            try:
                data = self._prepare_single_file(gt_path, od_results_dir, image_base_dir)
                if not data:
                    failed_count += 1
                    continue
                
                # Check if output file exists (supports resuming)
                artwork_id = data['artwork_id']
                output_path = os.path.join(output_dir, artwork_id, f"{artwork_id}_full_validation.json")
                
                if os.path.exists(output_path) and env.is_complete(load_gt_data(output_path)):
                    skipped_count += 1
                    continue
                
                batch_data.append(data)
                
            except Exception as e:
                print(f"Failed to prepare {os.path.basename(gt_path)}: {e}")
                failed_count += 1

        if not batch_data:
            return {'processed': 0, 'skipped': skipped_count, 'failed': failed_count}

        processed = 0
        failed = failed_count
        
        for data in batch_data:
            try:
                json_data = data['json_data']
                image_path = data['image_path']
                artwork_id = data['artwork_id']

                provenance = self._build_provenance(data)
                steps: Dict[str, Any] = {}

                # --- Gate 1: style -------------------------------------------
                # A wrong-style image is discarded before any other step runs.
                style_result = self._run_style_gate(json_data, image_path)
                steps[env.STEP_STYLE] = style_result
                if style_result.get('gate') == 'error':
                    print(f"  ! style classification for {artwork_id} did not complete "
                          f"({style_result.get('status')}): no result saved")
                    failed += 1
                    continue
                if style_result.get('gate') == 'failed':
                    full_result = env.discarded(
                        artwork_id, image_path, env.GATE_STYLE, steps, provenance)
                    if self._save(full_result, output_dir, artwork_id):
                        processed += 1
                    else:
                        failed += 1
                    continue

                # --- Gate 2: object presence ---------------------------------
                # Boxes are bound to instances by instance_assignment.
                instances = ia.instances_from_spec(json_data)
                palette_of = None
                if self.color_validator is not None:
                    palette_of = functools.partial(
                        self._safe_palette, image_path)

                measurement_block = measurements.build(
                    instances, data.get('detections', []), palette_of)

                steps[env.STEP_OBJECT] = {
                    'passed': measurement_block['all_present'],
                    'summary': measurement_block['summary'],
                    'objects': measurement_block['objects'],
                    'ambiguous_keys': data.get('ambiguous_keys', {}),
                }
                if not measurement_block['all_present']:
                    full_result = env.discarded(
                        artwork_id, image_path, env.GATE_OBJECT, steps,
                        provenance, measurement_block)
                    if self._save(full_result, output_dir, artwork_id):
                        processed += 1
                    else:
                        failed += 1
                    continue

                # --- Recorded steps ------------------------------------------
                # Run one after another; each is recorded and none discards
                # the artwork.
                # Boxes are keyed by instance_id.
                bboxes_dict = measurements.bboxes_by_instance(measurement_block)

                duplicates = measurements.duplicate_names(instances)
                if duplicates:
                    print(f"  ! duplicate object names: {duplicates}")
                    steps[env.STEP_OBJECT]['duplicate_names'] = duplicates

                print(f"  - Running color validation...")
                colour_result = self.color_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )
                self._add_scene_colour(colour_result, json_data, image_path,
                                       measurement_block)
                steps[env.STEP_COLOUR] = colour_result

                print(f"  - Running size validation...")
                steps[env.STEP_SIZE] = self.size_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )

                print(f"  - Running state validation...")
                steps[env.STEP_STATE] = self._run_state_validation(
                    json_data, image_path, bboxes_dict, measurement_block, artwork_id)

                print(f"  - Running spatial validation...")
                steps[env.STEP_SPATIAL] = self._run_spatial_validation(
                    json_data, image_path, bboxes_dict, measurement_block, artwork_id)

                print(f"  - Running alignment validation...")
                steps[env.STEP_ALIGNMENT] = self.alignment_validator.main_evaluation(
                    json_data, image_path, bboxes_dict, visualize=False
                )

                steps[env.STEP_SEMANTIC] = self._run_semantic_validation(
                    json_data, image_path, measurement_block, artwork_id)

                full_result = env.retained(
                    artwork_id, image_path, steps, measurement_block, provenance)

                if self._save(full_result, output_dir, artwork_id):
                    processed += 1
                else:
                    failed += 1

            except Exception as e:
                print(f"Failed to process {data.get('artwork_id', 'unknown')}: {e}")
                traceback.print_exc()
                failed += 1
        
        return {
            'processed': processed,
            'skipped': skipped_count,
            'failed': failed
        }

    def _prepare_single_file(self,
                            gt_path: str,
                            od_results_dir: str,
                            image_base_dir: str) -> Optional[Dict[str, Any]]:
        """Load the spec, locate its image and OD result, and extract detections."""

        json_data = load_gt_data(gt_path)
        if not json_data:
            print(f"Failed to load JSON data from {gt_path}")
            return None
        
        artwork_id = json_data.get('artwork_id', '')
        if not artwork_id:
            print(f"No artwork_id found in {gt_path}")
            return None
        
        # The generation stage writes <image_dir>/<artwork_id>.png.
        image_path = os.path.join(image_base_dir, f"{artwork_id}.png")
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
        
        # OD result: <id>_fused.json, then <id>_combined.json, else run detection.
        od_path = os.path.join(od_results_dir, f"{artwork_id}_fused.json")
        if not os.path.exists(od_path):
            od_path_combined = os.path.join(od_results_dir, f"{artwork_id}_combined.json")
            if os.path.exists(od_path_combined):
                od_path = od_path_combined
            else:
                if self.enable_od and self.object_detector:
                    print(f"OD result not found, running automatic detection for {artwork_id}...")

                    expected_objects = self._extract_object_names(json_data)

                    if expected_objects:
                        try:
                            detection_result = self.object_detector.main_evaluation(
                                json_data, image_path
                            )
                            # The detector reports its own failures as {'error': ...}.
                            # That is not a detection result: it is not cached, so
                            # the next run detects again.
                            if 'error' in detection_result:
                                print(f"Auto-detection failed for {artwork_id}: "
                                      f"{detection_result['error']}")
                                return None

                            write_json_atomic(detection_result, od_path)

                            print(f"Auto-detection completed for {artwork_id}")
                        except Exception as e:
                            print(f"Auto-detection failed for {artwork_id}: {e}")
                            return None
                    else:
                        print(f"No expected objects found for {artwork_id}")
                        return None
                else:
                    print(f"OD result not found: {od_path} (or _combined.json)")
                    return None


        with open(od_path, 'r') as f:
            try:
                od_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Failed to decode OD result: {od_path}")
                return None
        if 'error' in od_data:
            print(f"OD result is a failure record, not detections: {od_path} "
                  f"({od_data['error']}). Delete it to detect again.")
            return None

        bboxes_dict, detections = extract_detections(od_data)
        if not bboxes_dict:
            print(f"[BatchProcessor] Warning: no usable detections in {od_path}")

        print(f"[BatchProcessor] Loaded {len(bboxes_dict)} object bboxes from {os.path.basename(od_path)}")
        
        return {
            'artwork_id': artwork_id,
            'gt_path': gt_path,
            'image_path': image_path,
            'od_path': od_path,
            'json_data': json_data,
            'bboxes_dict': bboxes_dict,
            # Every detection; bboxes_dict keeps only the first per key.
            'detections': detections,
            'ambiguous_keys': duplicate_keys(detections)
        }

    def _extract_object_names(self, json_data: Dict[str, Any]) -> List[str]:
        """Extract object names from JSON data"""
        objects = []
        
        # Read from enhanced_objects, then objects.object_names, then a plain objects list.
        enhanced_objects = json_data.get('objects', {}).get('enhanced_objects', [])
        for obj in enhanced_objects:
            if isinstance(obj, dict) and 'name' in obj:
                objects.append(obj['name'])

        if not objects:
            object_names = json_data.get('objects', {}).get('object_names', [])
            if isinstance(object_names, list):
                objects.extend(object_names)

        if not objects:
            objects_list = json_data.get('objects', [])
            if isinstance(objects_list, list):
                for obj in objects_list:
                    if isinstance(obj, dict) and 'name' in obj:
                        objects.append(obj['name'])
                    elif isinstance(obj, str):
                        objects.append(obj)
        
        return objects


def discover_artwork_json_files(input_dir: str) -> List[str]:
    """Discover artwork JSON files recursively."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = [
        path for path in input_path.rglob("*.json")
        if path.name.lower().startswith("artwork_") or path.name.lower().startswith("batch_artwork_")
    ]
    json_files.sort(key=lambda path: str(path))

    if not json_files:
        raise ValueError(f"No artwork JSON files found in: {input_dir}")

    return [str(path) for path in json_files]


def build_cli_parser() -> argparse.ArgumentParser:
    """Build validation CLI parser."""
    parser = argparse.ArgumentParser(description="ARTO validation pipeline")
    parser.add_argument("--input-dir", "--input_dir", dest="input_dir", required=True,
                        help="Input JSON directory")
    parser.add_argument("--image-dir", "--image_dir", dest="image_dir", required=True,
                        help="Input image directory")
    parser.add_argument("--od-dir", "--od_dir", dest="od_dir", required=True,
                        help="Object detection result directory")
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True,
                        help="Validation output directory")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=10,
                        help="Batch size")
    parser.add_argument("--model-path", "--model_path", "--vlm-model-path", "--vlm_model_path",
                        dest="model_path", default=None, help="Validation VLM model path")
    parser.add_argument("--tensor-parallel-size", "--tensor_parallel_size", dest="tensor_parallel_size",
                        type=int, default=1, help="Number of GPUs for vLLM tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", "--gpu_memory_utilization", dest="gpu_memory_utilization",
                        type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--disable-od", action="store_true",
                        help="Disable automatic object detection fallback")
    parser.add_argument("--no-vlm", action="store_true",
                        help="Run only the parts that need no VLM: detection, "
                             "measurements, colour, size and the geometric spatial "
                             "channel. Style, state, semantic relations, alignment "
                             "and the VLM spatial channel are not checked, and the "
                             "result is marked incomplete: a run with a VLM "
                             "validates it again, and Stage 4 does not assemble it.")
    parser.add_argument("--annotation-dir", "--annotation_dir", dest="annotation_dir",
                        default=None,
                        help="Scratch directory for the box-marked images shown to the "
                             "VLM. Defaults to <output-dir>/annotations. One file per "
                             "question, overwritten as the run proceeds.")
    return parser


def main() -> None:
    """Run validation CLI."""
    args = build_cli_parser().parse_args()

    try:
        file_list = discover_artwork_json_files(args.input_dir)
        print(f"Discovered {len(file_list)} artwork JSON files")

        if args.no_vlm:
            # No wrapper at all: the VLM steps are not run.
            print("[Validation] --no-vlm: style, state, semantic relations, "
                  "alignment and the VLM spatial channel are not checked; "
                  "results are marked incomplete.")
            vlm_wrapper = None
        else:
            vlm_wrapper = create_vllm_wrapper(
                model_path=args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )

        processor = BatchProcessor(
            vlm_wrapper=vlm_wrapper,
            batch_size=args.batch_size,
            enable_od=not args.disable_od,
            annotation_dir=args.annotation_dir or os.path.join(args.output_dir, "annotations"),
        )

        stats = processor.process_file_list(
            file_list=file_list,
            output_dir=args.output_dir,
            od_results_dir=args.od_dir,
            image_base_dir=args.image_dir,
        )

        if stats["failed"] == 0:
            raise SystemExit(0)
        if stats["processed"] > 0:
            raise SystemExit(1)
        raise SystemExit(2)

    except SystemExit:
        raise
    except Exception as exc:
        print(f"[FATAL] Validation failed: {exc}")
        raise SystemExit(3)


if __name__ == "__main__":
    main()


__all__ = ['BatchProcessor', 'build_cli_parser', 'discover_artwork_json_files', 'main']
