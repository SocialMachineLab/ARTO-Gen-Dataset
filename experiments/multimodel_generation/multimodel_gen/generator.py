"""
Multi-Model Generator
Unified multi-model image generator
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from .models import QwenModel, FluxModel, SD35Model, SDXLModel


class MultiModelGenerator:
    """Multi-model Image Generator"""
    
    # Model class mapping
    MODEL_CLASSES = {
        'qwen': QwenModel,
        'flux': FluxModel,
        'sd35': SD35Model,
        'sdxl': SDXLModel,
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize generator
        
        Args:
            config_path: Config file path, defaults to config/models.json
        """
        if config_path is None:
            # Default config path
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / "config" / "models.json"
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.current_model = None
        self.model_key = None
    
    def load_model(self, model_key: str):
        """
        Load specified model
        
        Args:
            model_key: Model key (qwen/flux/sd35/sdxl)
        """
        if model_key not in self.MODEL_CLASSES:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(self.MODEL_CLASSES.keys())}")
        
        if model_key not in self.config:
            raise ValueError(f"Model {model_key} not found in config")
        
        # Unload current model
        if self.current_model is not None:
            print(f"[INFO] Unloading current model...")
            self.current_model.unload()
        
        # Load new model
        model_class = self.MODEL_CLASSES[model_key]
        self.current_model = model_class(self.config[model_key])
        self.current_model.load()
        self.model_key = model_key
    
    def generate_batch(self, dataset_dir: str, output_dir: str, model_key: str):
        """
        Generate batch images
        
        Args:
            dataset_dir: Dataset directory
            output_dir: Output directory
            model_key: Model key
        """
        dataset_path = Path(dataset_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.load_model(model_key)
        
        # Find all JSON files
        json_files = list(dataset_path.rglob("*.json"))
        json_files = [f for f in json_files if not f.name.startswith('.') and 'manifest' not in f.name.lower()]
        
        print(f"[INFO] Found {len(json_files)} JSON files")
        
        success_count = 0
        fail_count = 0
        
        for i, json_file in enumerate(json_files, 1):
            try:
                # Load JSON
                with open(json_file, 'r') as f:
                    artwork_data = json.load(f)
                
                artwork_id = artwork_data.get('artwork_id', json_file.stem)
                
                # Determine output path (maintain directory structure)
                rel_path = json_file.parent.relative_to(dataset_path)
                target_dir = output_path / rel_path
                target_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = target_dir / f"{artwork_id}.png"
                
                # Skip existing files
                if output_file.exists():
                    print(f"[{i}/{len(json_files)}] Skipping {artwork_id} (exists)")
                    success_count += 1
                    continue
                
                print(f"[{i}/{len(json_files)}] Generating {artwork_id}...")
                
                # Generate image
                if model_key == 'qwen':
                    # Qwen needs special handling
                    processed = self.current_model.process_artwork_data(artwork_data)
                    image = self.current_model.generate(processed['prompt'], processed['params'])
                else:
                    # Other models use simple prompt
                    prompt = artwork_data.get('final_prompts', {}).get('main_prompt', '')
                    params = self.current_model.get_params(seed=42)
                    image = self.current_model.generate(prompt, params)
                
                # Save image
                image.save(output_file)
                
                # Save metadata
                meta_file = target_dir / f"{artwork_id}_info.json"
                with open(meta_file, 'w') as f:
                    json.dump({
                        'artwork_id': artwork_id,
                        'model': model_key,
                        'timestamp': datetime.now().isoformat(),
                        'source_file': str(json_file)
                    }, f, indent=2)
                
                print(f"  ✓ Saved to {output_file}")
                success_count += 1
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                fail_count += 1
        
        print(f"\n[DONE] Success: {success_count}, Failed: {fail_count}")
