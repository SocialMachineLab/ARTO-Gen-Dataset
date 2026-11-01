"""
Result Validator: runs the ARTO validation (arto_kg.validation) on each model's images.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from arto_kg.validation.batch_processor import BatchProcessor
from arto_kg.validation.vllm_wrapper import create_vllm_wrapper

# Results go to <image folder>/validation/<artwork_id>/<artwork_id>_full_validation.json,
# detections to <image folder>/validation/od_results/.
VALIDATION_DIR = "validation"


class ResultValidator:
    """Result Validator - Validate generated images"""

    MODEL_FOLDERS = {
        "qwen": "qwen",
        "flux": "black-forest-labs_FLUX.1-dev",
        "sd35": "stabilityai_stable-diffusion-3.5-large",
        "sdxl": "stabilityai_stable-diffusion-xl-base-1.0"
    }

    def __init__(self, dataset_dir: str, results_dir: str,
                 tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.85):
        """
        Initialize validator

        Args:
            dataset_dir: Dataset directory (<style>/<bucket>/<artwork_id>.json)
            results_dir: Results directory (<model>/<style>/<bucket>/<artwork_id>.png)
            tensor_parallel_size: GPUs for the validation VLM
            gpu_memory_utilization: VLM share of GPU memory; the detectors use the rest
        """
        self.dataset_dir = Path(dataset_dir)
        self.results_dir = Path(results_dir)
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.processor = None

    def initialize_evaluator(self, image_dir: Optional[str] = None):
        """Load the validation VLM and the detectors."""
        # vLLM reads local images (and the box-marked copies) only under this path.
        roots = [str(REPO_ROOT), str(self.results_dir.resolve())]
        if image_dir:
            roots.append(str(Path(image_dir).resolve()))
        os.environ.setdefault("VLLM_MEDIA_PATH", os.path.commonpath(roots))
        print("Initializing validation VLM...")
        vlm_wrapper = create_vllm_wrapper(
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
        )
        self.processor = BatchProcessor(
            vlm_wrapper=vlm_wrapper,
            batch_size=1,
            annotation_dir=str(self.results_dir / "validation_annotations"),
        )

    def validate_model(self, model_key: str, force: bool = False, limit: Optional[int] = None,
                       image_dir: Optional[str] = None):
        """
        Validate results for specified model

        Args:
            model_key: Model key
            force: Re-validate artworks that already have a result
            limit: Validate at most this many artworks
            image_dir: Flat folder of <artwork_id>.png to use instead of the model's
                result folders
        """
        if model_key not in self.MODEL_FOLDERS:
            raise ValueError(f"Unknown model: {model_key}")

        print(f"Starting validation for model: {model_key}")
        model_path = self.results_dir / self.MODEL_FOLDERS[model_key]

        # One batch per <style>/<bucket> folder of the dataset: (image folder, spec paths).
        pending: Dict[Path, tuple] = {}
        missing_images = 0
        done = 0
        for spec_path in sorted(self.dataset_dir.rglob("artwork_*.json")):
            rel_path = spec_path.parent.relative_to(self.dataset_dir)
            artwork_id = spec_path.stem
            image_base = Path(image_dir) if image_dir else model_path / rel_path
            if not (image_base / f"{artwork_id}.png").exists():
                missing_images += 1
                continue

            output_dir = model_path / rel_path / VALIDATION_DIR
            result_path = output_dir / artwork_id / f"{artwork_id}_full_validation.json"
            if result_path.exists():
                if not force:
                    done += 1
                    continue
                result_path.unlink()

            pending.setdefault(rel_path, (image_base, []))[1].append(str(spec_path))
            if limit and sum(len(v[1]) for v in pending.values()) >= limit:
                break

        total_pending = sum(len(v[1]) for v in pending.values())
        print(f"To validate: {total_pending}, already validated: {done}, "
              f"no image: {missing_images}")
        if not total_pending:
            return

        if self.processor is None:
            self.initialize_evaluator(image_dir)

        processed = failed = 0
        for rel_path, (image_base, spec_paths) in pending.items():
            output_dir = model_path / rel_path / VALIDATION_DIR
            stats = self.processor.process_file_list(
                file_list=spec_paths,
                output_dir=str(output_dir),
                od_results_dir=str(output_dir / "od_results"),
                image_base_dir=str(image_base),
            )
            processed += stats["processed"]
            failed += stats["failed"]

        print(f"\nValidation Complete for {model_key}")
        print(f"Processed: {processed}")
        print(f"Failed: {failed}")
