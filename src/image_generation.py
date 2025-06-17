import json
import os
import glob
import time
import requests
from typing import Dict, Tuple, List, Optional
from datetime import datetime
from collections import Counter


class FluxImageGenerator:
    """Optimized Flux Image Generator for ARTO-based artwork datasets"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.bfl.ai/v1"
        self.headers = {
            "x-key": api_key,
            "Content-Type": "application/json"
        }
        
        # Standard aspect ratios for artwork generation
        self.aspect_ratios = {
            '4:3': (1024, 768),   # Landscape classic ratio
            '3:4': (768, 1024),   # Portrait classic ratio  
            '1:1': (1024, 1024)   # Square ratio
        }
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'successful_generations': 0,
            'failed_generations': 0,
            'skipped_existing': 0,
            'total_api_calls': 0,
            'aspect_ratio_distribution': Counter(),
            'style_distribution': Counter(),
            'object_count_distribution': Counter(),
            'generation_times': [],
            'prompt_lengths': [],
            'error_types': Counter()
        }
    
    def determine_aspect_ratio(self, scene_data: Dict) -> Tuple[str, Tuple[int, int], str]:
        """Intelligently determine optimal aspect ratio based on scene content"""
        objects = scene_data.get('selected_objects', {}).get('object_names', [])
        scene_design = scene_data.get('scene_design', {})
        art_style = scene_data.get('style', '')
        
        # Extract composition information
        stage1 = scene_design.get('stage1_scene_framework', {})
        stage2 = scene_design.get('stage2_spatial_layout', {})
        
        composition_type = stage1.get('composition_type', 'genre_scene')
        compositional_arrangement = stage2.get('compositional_arrangement', '')
        
        # Analyze object characteristics
        has_person = any('person' in obj.lower() for obj in objects)
        object_count = len(objects)
        
        # Aspect ratio selection logic
        if composition_type == 'portrait':
            if object_count <= 2 and has_person:
                ratio = '1:1'
                reason = "Portrait close-up, square composition"
            else:
                ratio = '3:4'
                reason = "Portrait painting, vertical orientation"
                
        elif composition_type == 'landscape':
            ratio = '4:3'
            reason = "Landscape painting, horizontal display"
            
        elif composition_type == 'still_life':
            if object_count <= 3:
                ratio = '1:1'
                reason = "Simple still life, square composition"
            else:
                ratio = '4:3'
                reason = "Complex still life, horizontal display"
                
        else:  # genre_scene, historical_scene, etc.
            if 'vertical' in compositional_arrangement.lower():
                ratio = '3:4'
                reason = "Vertical compositional scene"
            else:
                ratio = '4:3'
                reason = "Standard scene composition"
        
        # Style-specific adjustments
        if art_style == 'Chinese Ink Painting' and ratio == '4:3':
            ratio = '3:4'
            reason += " + Traditional Chinese vertical format"
        elif art_style == 'Sketch' and ratio == '4:3':
            ratio = '1:1'
            reason += " + Sketch square composition"
        
        return ratio, self.aspect_ratios[ratio], reason
    
    def analyze_scene_content(self, scene_data: Dict) -> Dict:
        """Analyze scene data for statistical purposes"""
        objects = scene_data.get('selected_objects', {}).get('object_names', [])
        object_count = len(objects)
        
        # Update object count distribution
        self.stats['object_count_distribution'][object_count] += 1
        
        # Categorize object types
        object_categories = {
            'has_person': any('person' in obj.lower() for obj in objects),
            'has_animal': any(obj.lower() in ['cat', 'dog', 'horse', 'bird', 'elephant', 'bear', 'sheep', 'cow'] for obj in objects),
            'has_vehicle': any(obj.lower() in ['car', 'truck', 'bicycle', 'motorcycle', 'bus', 'train'] for obj in objects),
            'has_furniture': any(obj.lower() in ['chair', 'table', 'bed', 'couch', 'dining table'] for obj in objects),
            'has_food': any(obj.lower() in ['apple', 'banana', 'cake', 'pizza', 'sandwich'] for obj in objects)
        }
        
        return {
            'object_count': object_count,
            'object_categories': object_categories,
            'objects': objects
        }
    
    def generate_optimized_prompt(self, scene_data: Dict) -> str:
        """Generate optimized prompt for Flux API"""
        scene_design = scene_data.get('scene_design', {})
        stage4 = scene_design.get('stage4_artistic_expression', {})
        final_prompt = stage4.get('final_prompt', '')
        
        if final_prompt:
            # Clean and optimize the prompt
            prompt = final_prompt.strip()
            # Keep prompt at reasonable length for better results
            if len(prompt) > 200:
                prompt = prompt[:200] + "..."
        else:
            # Generate basic prompt if none exists
            style = scene_data.get('style', 'painting')
            objects = ', '.join(scene_data.get('selected_objects', {}).get('object_names', []))
            prompt = f"A {style.lower()} featuring {objects}"
        
        # Add quality modifiers
        prompt += ", high quality, detailed, masterpiece"
        
        return prompt
    
    def call_flux_api(self, prompt: str, width: int, height: int) -> Dict:
        """Make API call to Flux image generation service"""
        start_time = time.time()
        self.stats['total_api_calls'] += 1
        
        url = f"{self.base_url}/flux-pro-1.1"
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "prompt_upsampling": False,
            "safety_tolerance": 2,
            "output_format": "jpeg"
        }
        
        try:
            # Submit generation task
            response = requests.post(url, json=payload, headers=self.headers)
            response.raise_for_status()
            
            result = response.json()
            task_id = result.get('id')
            polling_url = result.get('polling_url')
            
            print(f"  📤 Task submitted: {task_id[:8]}...")
            
            # Poll for results
            generation_result = self._poll_generation_result(polling_url, task_id)
            
            # Record generation time
            generation_time = time.time() - start_time
            self.stats['generation_times'].append(generation_time)
            
            return generation_result
            
        except Exception as e:
            error_type = type(e).__name__
            self.stats['error_types'][error_type] += 1
            print(f"  ❌ API Error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def _poll_generation_result(self, polling_url: str, task_id: str, max_wait: int = 120) -> Dict:
        """Poll for generation completion"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(polling_url, headers=self.headers)
                response.raise_for_status()
                
                result = response.json()
                status = result.get('status')
                
                if status == 'Ready':
                    elapsed = time.time() - start_time
                    print(f"  ✅ Generation completed ({elapsed:.1f}s)")
                    return {
                        "status": "success",
                        "result": result.get('result', {}),
                        "image_url": result.get('result', {}).get('sample')
                    }
                elif status == 'Error':
                    error_msg = result.get('error', 'Unknown error')
                    print(f"  ❌ Generation failed: {error_msg}")
                    return {"status": "error", "error": error_msg}
                else:
                    print(f"  ⏳ Generating... ({status})")
                    time.sleep(3)
                    
            except Exception as e:
                print(f"  ⚠️ Polling error: {str(e)}")
                time.sleep(3)
        
        print(f"  ⏰ Generation timeout")
        return {"status": "timeout", "error": "Generation timeout"}
    
    def download_image(self, image_url: str, save_path: str) -> bool:
        """Download generated image to local storage"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  💾 Saved: {os.path.basename(save_path)}")
            return True
            
        except Exception as e:
            print(f"  ❌ Download failed: {str(e)}")
            return False
    
    def process_single_artwork(self, json_file_path: str, output_dir: str, skip_existing: bool = True) -> Dict:
        """Process a single artwork JSON file"""
        file_start_time = time.time()
        
        try:
            # Load scene data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            artwork_id = scene_data.get('artwork_id', os.path.splitext(os.path.basename(json_file_path))[0])
            style = scene_data.get('style', 'Unknown')
            
            # Check if image already exists
            image_filename = f"{artwork_id}.jpg"
            image_path = os.path.join(output_dir, image_filename)
            
            if skip_existing and os.path.exists(image_path):
                print(f"  ⏭️  Skipped: {artwork_id} (image exists)")
                
                # Analyze for statistics
                scene_info = self.analyze_scene_content(scene_data)
                aspect_ratio, dimensions, reason = self.determine_aspect_ratio(scene_data)
                prompt = self.generate_optimized_prompt(scene_data)
                
                # Update statistics
                self._update_stats_for_processed(style, aspect_ratio, prompt, 'skipped')
                
                return self._create_result_dict(artwork_id, 'skipped', style, aspect_ratio, 
                                              scene_info['object_count'], 0.0, len(prompt))
            
            # Analyze scene content
            scene_info = self.analyze_scene_content(scene_data)
            
            # Determine aspect ratio
            aspect_ratio, dimensions, reason = self.determine_aspect_ratio(scene_data)
            width, height = dimensions
            
            # Generate prompt
            prompt = self.generate_optimized_prompt(scene_data)
            
            # Update statistics
            self._update_stats_for_processed(style, aspect_ratio, prompt, 'processing')
            
            print(f"  🎨 {style} | 📐 {aspect_ratio} | 🔢 {scene_info['object_count']} objects | 📝 {len(prompt)} chars")
            
            # Generate image
            generation_result = self.call_flux_api(prompt, width, height)
            
            if generation_result['status'] == 'success':
                # Download image
                image_url = generation_result['image_url']
                if image_url and self.download_image(image_url, image_path):
                    self.stats['successful_generations'] += 1
                    processing_time = time.time() - file_start_time
                    
                    return self._create_result_dict(artwork_id, 'success', style, aspect_ratio,
                                                  scene_info['object_count'], processing_time, len(prompt))
            
            # Generation failed
            self.stats['failed_generations'] += 1
            return self._create_result_dict(artwork_id, 'failed', style, aspect_ratio,
                                          scene_info['object_count'], 0.0, len(prompt),
                                          generation_result.get('error', 'Unknown error'))
            
        except Exception as e:
            self.stats['failed_generations'] += 1
            print(f"  ❌ Processing failed: {str(e)}")
            return self._create_result_dict('unknown', 'failed', 'Unknown', '1:1', 0, 0.0, 0, str(e))
    
    def _update_stats_for_processed(self, style: str, aspect_ratio: str, prompt: str, status: str):
        """Update statistics for processed artwork"""
        self.stats['aspect_ratio_distribution'][aspect_ratio] += 1
        self.stats['style_distribution'][style] += 1
        self.stats['prompt_lengths'].append(len(prompt))
        
        if status == 'skipped':
            self.stats['skipped_existing'] += 1
    
    def _create_result_dict(self, artwork_id: str, status: str, style: str, aspect_ratio: str,
                           object_count: int, processing_time: float, prompt_length: int, 
                           error: Optional[str] = None) -> Dict:
        """Create standardized result dictionary"""
        result = {
            'artwork_id': artwork_id,
            'status': status,
            'style': style,
            'aspect_ratio': aspect_ratio,
            'object_count': object_count,
            'processing_time': processing_time,
            'prompt_length': prompt_length
        }
        
        if error:
            result['error'] = error
            
        return result
    
    def process_batch(self, input_dir: str, output_dir: str, max_files: Optional[int] = None, 
                     skip_existing: bool = True) -> List[Dict]:
        """Process entire directory of artwork JSON files"""
        start_time = time.time()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find JSON files
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        if max_files:
            json_files = json_files[:max_files]
        
        if not json_files:
            print(f"❌ No JSON files found in {input_dir}")
            return []
        
        # Check existing images
        existing_count = 0
        if skip_existing:
            existing_images = {f.replace('.jpg', '') for f in os.listdir(output_dir) 
                             if f.endswith('.jpg')}
            existing_count = len(existing_images)
            
            if existing_count > 0:
                print(f"🔍 Found {existing_count} existing images")
        
        print(f"🎨 Processing {len(json_files)} files (Skip existing: {'Yes' if skip_existing else 'No'})")
        print("=" * 60)
        
        # Process files
        results = []
        self.stats['total_processed'] = len(json_files)
        
        for i, json_file in enumerate(json_files, 1):
            print(f"\n[{i}/{len(json_files)}] {os.path.basename(json_file)}")
            
            result = self.process_single_artwork(json_file, output_dir, skip_existing)
            results.append(result)
            
            # Rate limiting for actual generations
            if result.get('status') == 'success' and i < len(json_files):
                time.sleep(2)  # Prevent API rate limiting
        
        # Generate comprehensive report
        total_time = time.time() - start_time
        self.generate_analysis_report(results, output_dir, total_time)
        
        return results
    
    def generate_analysis_report(self, results: List[Dict], output_dir: str, total_time: float):
        """Generate comprehensive analysis report"""
        # Categorize results
        successful = [r for r in results if r.get('status') == 'success']
        skipped = [r for r in results if r.get('status') == 'skipped']
        failed = [r for r in results if r.get('status') == 'failed']
        
        # Calculate metrics
        success_rate = len(successful) / len(results) * 100 if results else 0
        avg_generation_time = (sum(self.stats['generation_times']) / len(self.stats['generation_times']) 
                             if self.stats['generation_times'] else 0)
        
        # Processing time analysis (only for newly generated)
        processing_times = [r.get('processing_time', 0) for r in successful if r.get('processing_time', 0) > 0]
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Content analysis
        all_processed = successful + skipped
        avg_prompt_length = (sum(self.stats['prompt_lengths']) / len(self.stats['prompt_lengths']) 
                           if self.stats['prompt_lengths'] else 0)
        object_counts = [r.get('object_count', 0) for r in all_processed]
        avg_object_count = sum(object_counts) / len(object_counts) if object_counts else 0
        
        # Generate detailed report
        report = {
            'report_generated': datetime.now().isoformat(),
            'execution_summary': {
                'total_runtime_seconds': round(total_time, 1),
                'total_files_processed': len(results),
                'newly_generated_images': len(successful),
                'skipped_existing_images': len(skipped),
                'failed_generations': len(failed),
                'overall_success_rate_percent': round(success_rate, 1)
            },
            'performance_metrics': {
                'total_api_calls': self.stats['total_api_calls'],
                'avg_generation_time_seconds': round(avg_generation_time, 1),
                'avg_processing_time_seconds': round(avg_processing_time, 1),
                'images_per_minute': round(len(successful) / (total_time / 60), 2) if total_time > 0 else 0,
                'api_efficiency_percent': round((len(successful) / max(1, self.stats['total_api_calls'])) * 100, 1)
            },
            'content_analysis': {
                'aspect_ratio_distribution': dict(self.stats['aspect_ratio_distribution']),
                'style_distribution': dict(self.stats['style_distribution']),
                'object_count_distribution': dict(self.stats['object_count_distribution']),
                'avg_objects_per_artwork': round(avg_object_count, 1),
                'avg_prompt_length_characters': round(avg_prompt_length, 1),
                'total_unique_styles': len(self.stats['style_distribution'])
            },
            'error_analysis': {
                'error_types_distribution': dict(self.stats['error_types']),
                'failed_artworks': [{'id': r['artwork_id'], 'error': r.get('error', 'Unknown')} 
                                  for r in failed]
            }
        }
        
        # Save detailed report
        report_file = os.path.join(output_dir, "flux_generation_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Save CSV summary
        self._save_csv_summary(all_processed, output_dir)
        
        # Print summary
        self._print_summary_stats(report, len(successful), len(skipped), len(failed))
        
        return report
    
    def _save_csv_summary(self, processed_results: List[Dict], output_dir: str):
        """Save CSV summary of all processed results"""
        if not processed_results:
            return
            
        import csv
        csv_file = os.path.join(output_dir, "generation_summary.csv")
        
        fieldnames = ['artwork_id', 'status', 'style', 'aspect_ratio', 'object_count', 
                     'processing_time', 'prompt_length']
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in processed_results:
                writer.writerow({k: result.get(k, '') for k in fieldnames})
    
    def _print_summary_stats(self, report: Dict, successful: int, skipped: int, failed: int):
        """Print key statistics summary"""
        print("\n" + "=" * 60)
        print("📊 GENERATION COMPLETED - KEY STATISTICS:")
        print(f"✅ Newly generated: {successful} images")
        print(f"⏭️  Skipped existing: {skipped} images")
        print(f"❌ Failed: {failed} generations")
        print(f"🎯 Success rate: {report['execution_summary']['overall_success_rate_percent']:.1f}%")
        
        if self.stats['generation_times']:
            print(f"⏱️  Avg generation time: {report['performance_metrics']['avg_generation_time_seconds']:.1f}s")
            print(f"🚀 Generation efficiency: {report['performance_metrics']['images_per_minute']:.1f} images/min")
        
        print(f"📐 Aspect ratio distribution: {dict(self.stats['aspect_ratio_distribution'])}")
        print(f"🎨 Style variety: {report['content_analysis']['total_unique_styles']} different styles")
        print(f"🔢 Avg objects per artwork: {report['content_analysis']['avg_objects_per_artwork']:.1f}")
        print(f"📋 Detailed report: flux_generation_report.json")
        print(f"📊 CSV summary: generation_summary.csv")


def main():
    """Main execution function"""
    # Configuration
    API_KEY = "your_flux_api_key_here"  # Replace with your actual API key
    INPUT_DIRECTORY = "./batch_artworks"
    OUTPUT_DIRECTORY = "./generated_images"
    MAX_FILES = 100  # Set to None for unlimited
    SKIP_EXISTING = True  # Skip files that already have generated images
    
    # Create generator instance
    generator = FluxImageGenerator(API_KEY)
    
    print("🎨 FLUX IMAGE GENERATOR FOR ARTO DATASET")
    print("=" * 60)
    print(f"📁 Input directory: {INPUT_DIRECTORY}")
    print(f"📁 Output directory: {OUTPUT_DIRECTORY}")
    print(f"🔢 Max files: {MAX_FILES if MAX_FILES else 'Unlimited'}")
    print(f"⏭️  Skip existing: {'Yes' if SKIP_EXISTING else 'No'}")
    print("=" * 60)
    
    # Process artworks
    results = generator.process_batch(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        max_files=MAX_FILES,
        skip_existing=SKIP_EXISTING
    )
    
    # Final summary
    successful = len([r for r in results if r.get('status') == 'success'])
    skipped = len([r for r in results if r.get('status') == 'skipped'])
    total_images = successful + skipped
    
    print(f"\n🎉 BATCH PROCESSING COMPLETED!")
    print(f"📊 Newly generated: {successful} images")
    print(f"⏭️  Skipped existing: {skipped} images")
    print(f"📁 Total images available: {total_images} images")


if __name__ == "__main__":
    main()