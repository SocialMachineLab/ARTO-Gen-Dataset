#!/usr/bin/env python3
"""
Iterative Image Generator
Generate 5 iterations for each artwork using different seeds
Does not contain validation logic, only responsible for image generation
"""

import os
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import existing generator (without modifying original code)
from main import BatchImageGenerator


class IterativeImageGenerator:
    """Iterative Image Generator Wrapper Class"""

    def __init__(self, output_dir: str, cache_dir: str = None,
                 base_seed: int = 42, num_iterations: int = 5):
        """
        Initialize iterative generator

        Args:
            output_dir: Output root directory
            cache_dir: Cache directory
            base_seed: Base random seed
            num_iterations: Number of iterations (default 5)
        """
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.base_seed = base_seed
        self.num_iterations = num_iterations

        # Base generator (reuse existing code)
        self.base_generator = None

        print(f"[INFO] Iterative Generator initialized")
        print(f"  Output: {self.output_dir}")
        print(f"  Base seed: {self.base_seed}")
        print(f"  Iterations: {self.num_iterations}")

    def generate_iterations_for_artwork(self, json_file_path: str) -> Dict[str, Any]:
        """
        Generate all iterations for a single artwork

        Args:
            json_file_path: Path to artwork JSON file

        Returns:
            Dictionary containing all iteration results
        """
        # Load JSON data
        with open(json_file_path, 'r', encoding='utf-8') as f:
            artwork_data = json.load(f)

        artwork_id = artwork_data.get('artwork_id', Path(json_file_path).stem)

        # Extract batch and style info from file path
        json_path = Path(json_file_path)

        # Detect directory structure
        # Possible structures:
        # 1. batch_xxx/final_results/style/artwork.json
        # 2. batch_xxx/final_results/artwork.json

        style_name = None
        if json_path.parent.name not in ['final_results', 'batch_xxx']:
            # Has style subdirectory
            style_name = json_path.parent.name
            batch_name = json_path.parent.parent.parent.name
        else:
            # Flat structure
            batch_name = json_path.parent.parent.name

        # Create artwork-specific directory
        if style_name:
            artwork_output_dir = self.output_dir / batch_name / style_name / artwork_id
        else:
            artwork_output_dir = self.output_dir / batch_name / artwork_id

        artwork_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"🎨 Processing Artwork: {artwork_id}")
        print(f"   Batch: {batch_name}")
        if style_name:
            print(f"   Style: {style_name}")
        print(f"   Output: {artwork_output_dir}")
        print(f"{'='*80}")

        # Save original JSON to artwork directory
        original_json_path = artwork_output_dir / "original_design.json"
        with open(original_json_path, 'w', encoding='utf-8') as f:
            json.dump(artwork_data, f, indent=2, ensure_ascii=False)

        # Iteration results statistics
        iterations_results = {
            'artwork_id': artwork_id,
            'batch_name': batch_name,
            'style_name': style_name,
            'source_json': str(json_file_path),
            'output_directory': str(artwork_output_dir),
            'num_iterations': self.num_iterations,
            'iterations': [],
            'generation_timestamp': datetime.now().isoformat()
        }

        successful_count = 0
        failed_count = 0

        # Generate each iteration
        for iteration in range(1, self.num_iterations + 1):
            print(f"\n--- Iteration {iteration}/{self.num_iterations} ---")

            # Create iteration directory
            iter_dir = artwork_output_dir / f"iteration_{iteration}"
            iter_dir.mkdir(exist_ok=True)

            # Calculate seed for this iteration
            iteration_seed = self.base_seed + (iteration - 1) * 1000

            try:
                # Create temporary generator (use seed for this iteration)
                temp_generator = BatchImageGenerator(
                    output_dir=str(iter_dir),
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                    batch_size=1,
                    seed=iteration_seed
                )

                # Process artwork data
                processed_data = temp_generator.process_artwork_data(artwork_data)

                if processed_data is None:
                    raise Exception("Failed to process artwork data")

                # Generate image
                generation_result = temp_generator.generate_single_image(processed_data)

                if generation_result and generation_result.get('success'):
                    print(f"✅ Iteration {iteration} completed successfully")

                    # Rename output file to standard name
                    generated_image = Path(generation_result['output_path'])
                    info_file = Path(generation_result['info_path'])

                    standard_image_path = iter_dir / "image.png"
                    standard_info_path = iter_dir / "generation_info.json"

                    if generated_image.exists():
                        generated_image.rename(standard_image_path)
                    if info_file.exists():
                        info_file.rename(standard_info_path)

                    iterations_results['iterations'].append({
                        'iteration': iteration,
                        'success': True,
                        'seed': iteration_seed,
                        'image_path': str(standard_image_path),
                        'info_path': str(standard_info_path),
                        'prompt_used': processed_data['final_prompt'][:100] + "...",
                        'generation_time': generation_result.get('generation_time', '')
                    })

                    successful_count += 1
                else:
                    raise Exception(generation_result.get('error', 'Unknown error'))

            except Exception as e:
                print(f"❌ Iteration {iteration} failed: {e}")
                traceback.print_exc()

                iterations_results['iterations'].append({
                    'iteration': iteration,
                    'success': False,
                    'seed': iteration_seed,
                    'error': str(e),
                    'generation_time': datetime.now().isoformat()
                })

                failed_count += 1

        # Save iteration summary
        iterations_results['summary'] = {
            'total_iterations': self.num_iterations,
            'successful': successful_count,
            'failed': failed_count,
            'success_rate': successful_count / self.num_iterations if self.num_iterations > 0 else 0
        }

        summary_path = artwork_output_dir / "iterations_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(iterations_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"📊 Artwork {artwork_id} - Generation Summary:")
        print(f"   Successful: {successful_count}/{self.num_iterations}")
        print(f"   Failed: {failed_count}/{self.num_iterations}")
        print(f"   Summary saved: {summary_path}")
        print(f"{'='*80}")

        return iterations_results

    def batch_generate_iterations(self, input_dir: str) -> Dict[str, Any]:
        """
        Batch process all artwork JSON files in directory

        Args:
            input_dir: Input directory containing JSON files

        Returns:
            Batch processing results
        """
        input_path = Path(input_dir)

        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        # Find all artwork JSON files
        json_files = list(input_path.glob("**/*artwork_*.json"))

        if not json_files:
            raise ValueError(f"No artwork JSON files found in: {input_dir}")

        print(f"\n{'='*80}")
        print(f"🚀 BATCH ITERATIVE GENERATION")
        print(f"{'='*80}")
        print(f"Input directory: {input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Found {len(json_files)} artwork files")
        print(f"Iterations per artwork: {self.num_iterations}")
        print(f"Total images to generate: {len(json_files) * self.num_iterations}")
        print(f"{'='*80}\n")

        batch_results = {
            'input_directory': str(input_dir),
            'output_directory': str(self.output_dir),
            'total_artworks': len(json_files),
            'iterations_per_artwork': self.num_iterations,
            'artworks': [],
            'start_time': datetime.now().isoformat()
        }

        successful_artworks = 0

        for i, json_file in enumerate(json_files, 1):
            print(f"\n{'#'*80}")
            print(f"# Processing Artwork {i}/{len(json_files)}")
            print(f"{'#'*80}")

            try:
                artwork_result = self.generate_iterations_for_artwork(str(json_file))
                batch_results['artworks'].append(artwork_result)

                if artwork_result['summary']['successful'] > 0:
                    successful_artworks += 1

            except Exception as e:
                print(f"❌ Failed to process {json_file}: {e}")
                traceback.print_exc()

                batch_results['artworks'].append({
                    'source_json': str(json_file),
                    'error': str(e)
                })

        batch_results['end_time'] = datetime.now().isoformat()
        batch_results['summary'] = {
            'total_artworks_processed': len(json_files),
            'successful_artworks': successful_artworks,
            'failed_artworks': len(json_files) - successful_artworks
        }

        # Save batch results
        batch_summary_path = self.output_dir / f"batch_generation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(batch_summary_path, 'w', encoding='utf-8') as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"🏁 BATCH GENERATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total artworks: {len(json_files)}")
        print(f"Successful: {successful_artworks}")
        print(f"Failed: {len(json_files) - successful_artworks}")
        print(f"Summary: {batch_summary_path}")
        print(f"{'='*80}\n")

        return batch_results


def main():
    parser = argparse.ArgumentParser(description="Iterative Image Generator")
    parser.add_argument("--input-dir", required=True, help="Input JSON directory")
    parser.add_argument("--output-dir", required=True, help="Output image root directory")
    parser.add_argument("--cache-dir", help="Model cache directory")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per artwork")

    args = parser.parse_args()

    # Create iterative generator
    generator = IterativeImageGenerator(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        base_seed=args.base_seed,
        num_iterations=args.iterations
    )

    # Execute batch generation
    try:
        results = generator.batch_generate_iterations(args.input_dir)

        # Determine exit code based on results
        if results['summary']['failed_artworks'] == 0:
            exit(0)
        elif results['summary']['successful_artworks'] > 0:
            exit(1)  # Partial success
        else:
            exit(2)  # All failed

    except Exception as e:
        print(f"[FATAL] Batch generation failed: {e}")
        traceback.print_exc()
        exit(3)


if __name__ == "__main__":
    main()
