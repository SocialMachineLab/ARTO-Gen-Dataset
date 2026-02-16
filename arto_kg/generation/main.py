"""
Batch Image Generation Main Program
Supports reading detailed information from JSON files and generating corresponding images
"""

import os
import json
import math
import torch
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler

from .prompt_processor import PromptProcessor
from .style_handler import StyleHandler
from .utils import setup_logging, save_generation_info, cleanup_memory


def build_scheduler():
    """Build scheduler"""
    cfg = {
        "base_image_seq_len": 256,
        "base_shift": math.log(3),
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": math.log(3),
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    }
    return FlowMatchEulerDiscreteScheduler.from_config(cfg)


class BatchImageGenerator:
    """Batch Image Generator"""
    
    def __init__(self, output_dir: str, cache_dir: str = None, batch_size: int = 1, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.batch_size = batch_size
        self.seed = seed
        
        self.pipe = None
        self.prompt_processor = PromptProcessor()
        self.style_handler = StyleHandler()
        
        # Set cache directory
        if self.cache_dir:
            self._setup_cache_dirs()
        
        # Set memory optimization
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch.cuda.empty_cache()
        
        print(f"[INFO] Detected {torch.cuda.device_count()} GPUs")
    
    def _setup_cache_dirs(self):
        """Set cache directory"""
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        cache_vars = {
            "HF_HOME": self.cache_dir / "hf",
            "HUGGINGFACE_HUB_CACHE": self.cache_dir / "hf",
            "TORCH_HOME": self.cache_dir / "torch",
            "PYTORCH_KERNEL_CACHE_PATH": self.cache_dir / "torch" / "kernel"
        }
        
        for var, path in cache_vars.items():
            path.mkdir(exist_ok=True, parents=True)
            os.environ[var] = str(path)
    
    def load_model(self):
        """Load model"""
        if self.pipe is not None:
            print("[INFO] Model already loaded")
            return

        print("[INFO] Loading Qwen-Image model...")

        # Use same configuration as test code
        scheduler = build_scheduler()

        print(f"[INFO] CUDA available: {torch.cuda.is_available()}")
        device_count = torch.cuda.device_count()
        print(f"[INFO] Detected {device_count} GPUs")
        
        if device_count < 2:
            print("[WARNING] 'balanced' device_map requires multiple GPUs to split the model effectively.")
            print("[WARNING] Only 1 GPU detected. This may lead to OOM on GPU 0.")
            print("[WARNING] Please ensure you requested 2 GPUs in your SLURM script (e.g., #SBATCH --gpus=2).")
        else:
            print(f"[INFO] Multi-GPU mode enabled. Splitting model across {device_count} GPUs.")
            for i in range(device_count):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)} (Mem: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB)")

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                "Qwen/Qwen-Image",
                scheduler=scheduler,
                torch_dtype=torch.bfloat16,
                device_map="balanced",
                low_cpu_mem_usage=True,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[ERROR] CUDA Out of Memory during model loading.")
                print(f"[TIP] If you intended to use 2 GPUs, check if they are both visible via 'nvidia-smi' inside the job.")
                print(f"[TIP] Current visible devices: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
            raise e
        
        # Enable memory optimizations
        print("[INFO] Enabling memory optimizations...")
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        
        try:
            self.pipe.enable_attention_slicing(1)
            print("[INFO] Attention slicing enabled")
        except:
            print("[INFO] Attention slicing not available")
        
        print("[INFO] Model loaded successfully!")
    
    def load_json_files(self, input_dir: str, skip_files: int = 0, max_files: int = None) -> List[Dict[str, Any]]:
        """Load all JSON files from input directory"""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        json_files = list(input_path.rglob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in: {input_dir}")

        # Filter out non-artwork files
        artwork_files = []
        for json_file in json_files:
            filename = json_file.name.lower()
            # Only process artwork files, skip batch_summary, batch_report, etc.
            if filename.startswith('artwork_') or filename.startswith('batch_artwork_'):
                artwork_files.append(json_file)
            else:
                print(f"[INFO] Skipping non-artwork file: {json_file.name}")

        if not artwork_files:
            raise ValueError(f"No artwork JSON files found in: {input_dir}")

        # Sort to ensure consistency
        artwork_files.sort(key=lambda x: str(x))
        
        total_files = len(artwork_files)
        print(f"[INFO] Found {total_files} artwork JSON files (filtered from {len(json_files)} total)")
        
        # Apply chunking parameters
        if skip_files > 0:
            print(f"[INFO] Skipping first {skip_files} files")
            artwork_files = artwork_files[skip_files:]
        
        if max_files is not None:
            print(f"[INFO] Limiting to {max_files} files")
            artwork_files = artwork_files[:max_files]
        
        print(f"[INFO] Processing {len(artwork_files)} files (from index {skip_files} to {skip_files + len(artwork_files) - 1})")

        json_data = []
        for json_file in artwork_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    data['_source_file'] = str(json_file)
                    data['_filename'] = json_file.stem
                    json_data.append(data)
                    print(f"[INFO] Loaded: {json_file.name}")
            except Exception as e:
                print(f"[ERROR] Failed to load {json_file}: {e}")
                print(f"[ERROR] JSON parse error details: {type(e).__name__}: {str(e)}")
        
        return json_data
    
    def process_artwork_data(self, artwork_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process single artwork data, extract generation parameters"""
        try:
            # Extract basic info
            artwork_id = artwork_data.get('artwork_id', artwork_data.get('_filename', 'unknown'))
            
            # Process prompts
            prompts = self.prompt_processor.extract_prompts(artwork_data)
            
            # Process style info
            style_info = self.style_handler.extract_style_info(artwork_data)
            
            # Build final prompt
            final_prompt = self.prompt_processor.build_final_prompt(
                prompts, style_info, artwork_data
            )
            
            # Extract generation parameters
            generation_params = self._extract_generation_params(artwork_data, style_info)
            
            return {
                'artwork_id': artwork_id,
                'final_prompt': final_prompt,
                'prompts': prompts,
                'style_info': style_info,
                'generation_params': generation_params,
                'source_data': artwork_data
            }
            
        except Exception as e:
            print(f"[ERROR] Failed to process artwork data for {artwork_data.get('artwork_id', 'unknown')}: {e}")
            print(f"[ERROR] Processing error details: {type(e).__name__}: {str(e)}")
            if 'artwork_id' in artwork_data:
                print(f"[ERROR] Source file: {artwork_data.get('_source_file', 'unknown')}")
            return None
    
    def _extract_generation_params(self, artwork_data: Dict[str, Any], 
                                  style_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract generation parameters"""
        # Default parameters
        params = {
            'width': 1024,
            'height': 1024,
            'num_inference_steps': 80,
            'true_cfg_scale': 1.0,  # Default value, will be overwritten below
            'seed': self.seed
        }
        
        # Adaptively adjust CFG Scale based on object count (Key optimization!)
        # Qwen-Image v1 doesn't support CFG without negative prompt, and negative prompt is not supported yet
        # So we must use 1.0 to avoid artifacts/noise
        params['true_cfg_scale'] = 1.0
        
        # Override parameters from artwork data (if custom params exist)
        if 'generation_params' in artwork_data:
            gen_params = artwork_data['generation_params']
            if isinstance(gen_params, dict):
                params.update(gen_params)
        
        return params
    
    def generate_single_image(self, processed_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate single image"""
        # Model should be loaded at start of batch, this is just a safety check
        if self.pipe is None:
            print("[WARNING] Model not loaded, loading now...")
            self.load_model()
        
        artwork_id = processed_data['artwork_id']
        final_prompt = processed_data['final_prompt']
        generation_params = processed_data['generation_params']
        
        print(f"\n[INFO] Generating image for: {artwork_id}")
        print(f"[INFO] Prompt: {final_prompt[:100]}...")
        print(f"[INFO] Params: {generation_params}")
        
        try:
            cleanup_memory()
            
            # Show memory usage
            print("\n[INFO] GPU memory before generation:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                print(f"  GPU {i}: {allocated:.1f}GB")
            
            # Generate image
            with torch.inference_mode():
                generator = torch.Generator().manual_seed(generation_params['seed'])

                # Check GPU memory sufficiency
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        if allocated > 20:  # If GPU memory > 20GB, force cleanup
                            print(f"[WARNING] High GPU memory usage ({allocated:.1f}GB), forcing cleanup")
                            cleanup_memory()
                            break

                # Use same parameters as test code
                images = self.pipe(
                    prompt=final_prompt,
                    width=generation_params['width'],
                    height=generation_params['height'],
                    num_inference_steps=generation_params['num_inference_steps'],
                    true_cfg_scale=generation_params['true_cfg_scale'],
                    generator=generator,
                ).images
            
            print("\n[INFO] GPU memory after generation:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                print(f"  GPU {i}: {allocated:.1f}GB")
            
            # Save image
            output_filename = f"{artwork_id}.png"
            output_path = self.output_dir / output_filename
            images[0].save(output_path)
            
            # Save generation info
            info_path = self.output_dir / f"{artwork_id}_info.json"
            save_generation_info(processed_data, output_path, info_path)
            
            print(f"[SUCCESS] Image saved: {output_path}")
            
            return {
                'artwork_id': artwork_id,
                'output_path': str(output_path),
                'info_path': str(info_path),
                'success': True,
                'generation_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"[ERROR] Generation failed for {artwork_id}: {e}")
            traceback.print_exc()

            # Clean GPU memory
            cleanup_memory()

            return {
                'artwork_id': artwork_id,
                'success': False,
                'error': str(e),
                'generation_time': datetime.now().isoformat()
            }
        finally:
            # Ensure memory is cleaned up after each generation
            cleanup_memory()
    
    def batch_generate(self, input_dir: str, skip_files: int = 0, max_files: int = None) -> Dict[str, Any]:
        """Batch generate images"""
        print(f"[INFO] Starting batch generation from: {input_dir}")
        print(f"[INFO] Output directory: {self.output_dir}")
        
        # Load all JSON files
        json_data_list = self.load_json_files(input_dir, skip_files=skip_files, max_files=max_files)
        total_files = len(json_data_list)
        
        # Generate result statistics
        results = {
            'total_files': total_files,
            'successful': 0,
            'failed': 0,
            'results': [],
            'start_time': datetime.now().isoformat()
        }
        
        # Load model (load once to avoid repeated checks)
        print("[INFO] Loading model once for batch processing...")
        self.load_model()

        # Process one by one
        for i, artwork_data in enumerate(json_data_list, 1):
            # Cleanup memory every 5 images
            if i % 5 == 0:
                print("[INFO] Performing periodic memory cleanup...")
                cleanup_memory()
            print(f"\n{'='*80}")
            print(f"[INFO] Processing {i}/{total_files}: {artwork_data.get('_filename', 'unknown')}")
            print(f"{'='*80}")
            
            # Check if output file already exists (skip logic)
            artwork_id = artwork_data.get('_filename', f"artwork_{i}")
            # Remove _v2 suffix (if exists) to match save logic
            if artwork_id.endswith('_v2'):
                artwork_id = artwork_id[:-3]
            
            output_filename = f"{artwork_id}.png"
            output_path = self.output_dir / output_filename
            
            if output_path.exists():
                print(f"[INFO] ✓ Skipping {artwork_id} - output already exists")
                results['successful'] += 1  # Count as success (since file already exists)
                continue
            
            try:
                # Process data
                processed_data = self.process_artwork_data(artwork_data)
                if processed_data is None:
                    results['failed'] += 1
                    continue
                
                # Generate image
                generation_result = self.generate_single_image(processed_data)
                
                if generation_result and generation_result.get('success'):
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                results['results'].append(generation_result)
                
            except Exception as e:
                print(f"[ERROR] Failed to process file {i}: {e}")
                print(f"[ERROR] File processing error details: {type(e).__name__}: {str(e)}")
                traceback.print_exc()
                results['failed'] += 1
                results['results'].append({
                    'artwork_id': artwork_data.get('_filename', 'unknown'),
                    'success': False,
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'source_file': artwork_data.get('_source_file', 'unknown'),
                    'generation_time': datetime.now().isoformat()
                })
        
        results['end_time'] = datetime.now().isoformat()
        
        # Save batch results
        batch_results_path = self.output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"[INFO] Batch generation completed!")
        print(f"[INFO] Total: {total_files}, Successful: {results['successful']}, Failed: {results['failed']}")
        print(f"[INFO] Results saved to: {batch_results_path}")
        print(f"{'='*80}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Batch Image Generator")
    parser.add_argument("--input-dir", required=True, help="Input JSON directory")
    parser.add_argument("--output-dir", required=True, help="Output image directory")
    parser.add_argument("--cache-dir", help="Cache directory")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--skip-files", type=int, default=0, help="Skip first N files (for chunked processing)")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N files (for chunked processing)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create generator
    generator = BatchImageGenerator(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        seed=args.seed
    )
    
    # Start batch generation
    try:
        results = generator.batch_generate(
            args.input_dir,
            skip_files=args.skip_files,
            max_files=args.max_files
        )
        
        # Exit code
        if results['failed'] == 0:
            exit(0)
        elif results['successful'] > 0:
            exit(1)  # Partial success
        else:
            exit(2)  # All failed
            
    except Exception as e:
        print(f"[FATAL] Batch generation failed: {e}")
        traceback.print_exc()
        exit(3)


if __name__ == "__main__":
    main()


    