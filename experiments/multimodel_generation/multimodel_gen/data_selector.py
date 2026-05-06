"""
Data Selector
Select high-quality artwork data from object_detection experiments
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class DataSelector:
    """Data Selector - Select balanced experiment dataset"""
    
    # Style mapping
    STYLES = {
        "baroque": "Baroque",
        "neoclassicism": "Neoclassicism",
        "impressionism": "Impressionism",
        "post_impressionism": "Post-Impressionism",
        "chinese_ink_painting": "Chinese Ink Painting"
    }
    
    STYLE_MAPPING = {
        "Baroque": "baroque",
        "baroque": "baroque",
        "Neoclassicism": "neoclassicism",
        "neoclassicism": "neoclassicism",
        "Impressionism": "impressionism",
        "impressionism": "impressionism",
        "Post-Impressionism": "post_impressionism",
        "post-impressionism": "post_impressionism",
        "Chinese Ink Painting": "chinese_ink_painting",
        "chinese ink painting": "chinese_ink_painting",
    }
    
    # Complexity bins
    COMPLEXITY_BINS = {
        "1-3_objects": (1, 3),
        "4-6_objects": (4, 6),
        "7-9_objects": (7, 9)
    }
    
    # Target count per style
    TARGET_PER_STYLE = {
        "1-3_objects": 3,
        "4-6_objects": 4,
        "7-9_objects": 3
    }
    
    def __init__(self, source_dir: str, output_dir: str, seed: int = 42):
        """
        Initialize data selector
        
        Args:
            source_dir: Source directory (experiments/object_detection)
            output_dir: Output directory (experiment_dataset)
            seed: Random seed
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        random.seed(seed)
    
    def get_complexity_bin(self, obj_count: int) -> str:
        """Determine complexity bin based on object count"""
        if 1 <= obj_count <= 3:
            return "1-3_objects"
        elif 4 <= obj_count <= 6:
            return "4-6_objects"
        elif 7 <= obj_count <= 9:
            return "7-9_objects"
        return None
    
    def is_valid_artwork(self, data: Dict[str, Any]) -> bool:
        """Check if artwork meets quality standards"""
        # Must have objects
        if not data.get('objects'):
            return False
        
        # Must have object names
        object_names = data.get('objects', {}).get('object_names', [])
        if not object_names:
            return False
        
        # Object count should be reasonable
        obj_count = len(object_names)
        if obj_count < 1 or obj_count > 9:
            return False
        
        # Must have style
        if not data.get('style'):
            return False
        
        # Must have prompt
        if not data.get('final_prompts', {}).get('main_prompt'):
            return False
        
        return True
    
    def scan_and_categorize(self) -> Dict[str, Dict[str, List[Dict]]]:
        """Scan source directory and categorize by style and complexity"""
        artworks = defaultdict(lambda: defaultdict(list))
        
        print("Scanning source directory...")
        
        if not self.source_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.source_dir}")
        
        # Iterate style directories
        for style_dir in self.source_dir.iterdir():
            if not style_dir.is_dir() or style_dir.name == "scripts" or style_dir.name.startswith("."):
                continue
            
            print(f"Scanning {style_dir.name}...")
            json_files = list(style_dir.glob("*.json"))
            # Filter out result files
            json_files = [f for f in json_files if not f.name.endswith("_result.json") 
                         and not f.name.endswith("_groundtruth.json")]
            
            print(f"  Found {len(json_files)} potential files")
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate
                    if not self.is_valid_artwork(data):
                        continue
                    
                    # Map style
                    raw_style = data.get('style', '')
                    canonical_style = self.STYLE_MAPPING.get(raw_style)
                    if not canonical_style:
                        continue
                    
                    # Get complexity
                    obj_count = len(data.get('objects', {}).get('object_names', []))
                    complexity = self.get_complexity_bin(obj_count)
                    if not complexity:
                        continue
                    
                    # Store
                    artworks[canonical_style][complexity].append({
                        'path': json_file,
                        'data': data,
                        'obj_count': obj_count
                    })
                    
                except Exception as e:
                    print(f"  Error reading {json_file.name}: {e}")
                    continue
        
        return artworks
    
    def select_balanced(self, artworks: Dict) -> Dict:
        """Select balanced dataset"""
        selected = defaultdict(lambda: defaultdict(list))
        
        print("\nSelecting artworks...")
        print(f"{'Style':<25} | {'1-3 obj':<8} {'4-6 obj':<8} {'7-9 obj':<8} | Total")
        print("-" * 70)
        
        for style in self.STYLES.keys():
            style_total = 0
            
            for complexity, target_count in self.TARGET_PER_STYLE.items():
                available = artworks[style][complexity]
                
                if len(available) < target_count:
                    print(f"WARNING: {style}/{complexity} has only {len(available)} artworks (need {target_count})")
                    selected_items = available
                else:
                    # Random sampling
                    selected_items = random.sample(available, target_count)
                
                selected[style][complexity] = selected_items
                style_total += len(selected_items)
            
            counts = [len(selected[style][c]) for c in ["1-3_objects", "4-6_objects", "7-9_objects"]]
            print(f"{style:<25} | {counts[0]:<8} {counts[1]:<8} {counts[2]:<8} | {style_total}")
        
        # Calculate total
        total = sum(len(selected[s][c]) for s in self.STYLES.keys() for c in self.COMPLEXITY_BINS.keys())
        print("-" * 70)
        print(f"{'TOTAL':<25} | {'':<26} | {total}")
        
        return selected
    
    def copy_selected(self, selected: Dict):
        """Copy selected files to output directory"""
        print(f"\nCopying to {self.output_dir}...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        manifest = {
            'total_artworks': 0,
            'styles': {}
        }
        
        for style in self.STYLES.keys():
            style_manifest = {
                'total': 0,
                'complexity': {}
            }
            
            for complexity in self.COMPLEXITY_BINS.keys():
                # Create directory
                output_subdir = self.output_dir / style / complexity
                output_subdir.mkdir(parents=True, exist_ok=True)
                
                artworks_list = []
                
                for item in selected[style][complexity]:
                    src_path = item['path']
                    dst_path = output_subdir / src_path.name
                    
                    # Copy file
                    shutil.copy2(src_path, dst_path)
                    
                    artworks_list.append({
                        'artwork_id': item['data']['artwork_id'],
                        'filename': src_path.name,
                        'object_count': item['obj_count'],
                        'source': str(src_path)
                    })
                
                style_manifest['complexity'][complexity] = {
                    'count': len(artworks_list),
                    'artworks': artworks_list
                }
                style_manifest['total'] += len(artworks_list)
                
                print(f"  ✓ {style}/{complexity}: {len(artworks_list)} files")
            
            manifest['styles'][style] = style_manifest
            manifest['total_artworks'] += style_manifest['total']
        
        # Save manifest
        manifest_path = self.output_dir / "selection_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\n✅ Copied {manifest['total_artworks']} artworks")
        print(f"✅ Manifest saved to {manifest_path}")
    
    def run(self):
        """Execute full data selection process"""
        print("=" * 70)
        print("Multi-Model Experiment: Data Selection")
        print("=" * 70)
        
        # Scan
        artworks = self.scan_and_categorize()
        
        # Show availability
        print("\nAvailability:")
        print(f"{'Style':<25} | {'1-3 obj':<8} {'4-6 obj':<8} {'7-9 obj':<8}")
        print("-" * 70)
        for style in self.STYLES.keys():
            counts = [len(artworks[style][c]) for c in ["1-3_objects", "4-6_objects", "7-9_objects"]]
            print(f"{style:<25} | {counts[0]:<8} {counts[1]:<8} {counts[2]:<8}")
        
        # Select
        selected = self.select_balanced(artworks)
        
        # Copy
        self.copy_selected(selected)
        
        print("\n" + "=" * 70)
        print("Data selection complete!")
        print("=" * 70)
