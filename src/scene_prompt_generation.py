import json
import random
import os
from typing import List, Dict, Any, Optional
from ollama import chat
import re
from datetime import datetime
from collections import Counter


class ARTOArtworkPipeline:
    """ARTO-Guided Artwork Generation Pipeline: Object Selection + Scene Design"""
    
    def __init__(self, model_name: str = 'deepseek-r1:70b'):
        self.model_name = model_name
        
        # COCO object mapping (80 categories)
        self.coco_objects = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 12: 'stop sign', 13: 'parking meter', 14: 'bench',
            15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep',
            20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe',
            25: 'backpack', 26: 'umbrella', 27: 'handbag', 28: 'tie', 29: 'suitcase',
            30: 'frisbee', 31: 'skis', 32: 'snowboard', 33: 'sports ball', 34: 'kite',
            35: 'baseball bat', 36: 'baseball glove', 37: 'skateboard', 38: 'surfboard', 39: 'tennis racket',
            40: 'bottle', 41: 'wine glass', 42: 'cup', 43: 'fork', 44: 'knife',
            45: 'spoon', 46: 'bowl', 47: 'banana', 48: 'apple', 49: 'sandwich',
            50: 'orange', 51: 'broccoli', 52: 'carrot', 53: 'hot dog', 54: 'pizza',
            55: 'donut', 56: 'cake', 57: 'chair', 58: 'couch', 59: 'potted plant',
            60: 'bed', 61: 'dining table', 62: 'toilet', 63: 'tv', 64: 'laptop',
            65: 'mouse', 66: 'remote', 67: 'keyboard', 68: 'cell phone', 69: 'microwave',
            70: 'oven', 71: 'toaster', 72: 'sink', 73: 'refrigerator', 74: 'book',
            75: 'clock', 76: 'vase', 77: 'scissors', 78: 'teddy bear', 79: 'hair drier',
            80: 'toothbrush'
        }
        
        # Art styles for selection
        self.art_styles = [
            "Oil Painting", "Post-Impressionism", "Photorealistic", 
            "Sketch", "Chinese Ink Painting", "Renaissance", "Baroque",
            "Impressionism", "Romantic", "Realism", "Abstract", "Surrealism"
        ]
        
        # System prompt for object compatibility checking
        self.compatibility_system_prompt = """
You are an expert composition analyzer. Evaluate whether objects can realistically coexist in a single artwork scene.

Focus on ABSOLUTE CONFLICTS - situations where objects cannot physically or logically appear together.

INPUT FORMAT:
main object: [main_object_name]
secondary objects: [list_of_secondary_objects]

CONFLICT TYPES:
- Physical impossibility
- Universal incompatibility  
- Fundamental logical conflicts

RESPONSE FORMAT (JSON only):
Compatible:
{
"compatible": true,
"example_scenario": "Brief description of realistic scenario"
}

Incompatible:
{
"compatible": false,
"main_object_conflicts": [
    {"object1": "main_object", "object2": "conflicting_object", "reason": "explanation"}
],
"secondary_object_conflicts": [
    {"object1": "object1", "object2": "object2", "reason": "explanation"}
]
}

Rules:
- Only identify ABSOLUTE conflicts
- Be very conservative
- Always respond with valid JSON only
"""

    def generate_artwork(self, 
                        style: Optional[str] = None, 
                        max_secondary_objects: int = 8,
                        output_dir: str = "artworks") -> Dict[str, Any]:
        """Complete artwork generation workflow"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate unique ID
        artwork_id = f"artwork_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(100, 999)}"
        
        print(f"🎨 Starting artwork generation: {artwork_id}")
        print("=" * 60)
        
        # Step 0: Select style if not specified
        if style is None:
            style = random.choice(self.art_styles)
        print(f"🎭 Selected style: {style}")
        
        # Step 1: Object selection and compatibility check
        print(f"🎯 Phase 1: Object Selection and Compatibility Check")
        selected_objects = self._select_compatible_objects(max_secondary_objects)
        object_names = [self.coco_objects[obj_id] for obj_id in selected_objects]
        print(f"✅ Final objects: {object_names}")
        
        # Step 2: Four-stage scene design
        print(f"🏗️  Phase 2: Four-Stage Scene Design")
        scene_data = self._generate_scene_design(selected_objects, style)
        
        # Combine final data
        final_data = {
            "artwork_id": artwork_id,
            "generation_timestamp": datetime.now().isoformat(),
            "style": style,
            "selected_objects": {
                "object_ids": selected_objects,
                "object_names": object_names
            },
            "scene_design": scene_data,
            "metadata": {
                "max_secondary_objects": max_secondary_objects,
                "model_used": self.model_name
            }
        }
        
        # Save data
        filename = os.path.join(output_dir, f"{artwork_id}.json")
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        print("=" * 60)
        print(f"🎉 Artwork generation completed!")
        print(f"💾 Saved to: {filename}")
        print(f"🖼️  Final prompt: {scene_data.get('stage4_artistic_expression', {}).get('final_prompt', 'N/A')}")
        
        return final_data

    def _select_compatible_objects(self, max_secondary_objects: int) -> List[int]:
        """Select compatible object combinations"""
        print("🔍 Selecting and validating object compatibility...")
        
        # Randomly select main object
        main_object = random.choice(list(range(1, 81)))
        print(f"   🎯 Main object: {self.coco_objects[main_object]}")
        
        # Randomly select secondary objects
        secondary_count = random.randint(0, max_secondary_objects)
        available_objects = [i for i in range(1, 81) if i != main_object]
        secondary_objects = random.sample(available_objects, min(secondary_count, len(available_objects)))
        
        all_objects = [main_object] + secondary_objects
        print(f"   📝 Initial selection: {[self.coco_objects[obj] for obj in all_objects]}")
        
        # Compatibility check and conflict resolution
        final_objects = self._resolve_conflicts(main_object, all_objects)
        print(f"   ✅ After compatibility check: {[self.coco_objects[obj] for obj in final_objects]}")
        
        return final_objects

    def _resolve_conflicts(self, main_object: int, all_objects: List[int]) -> List[int]:
        """Resolve object conflicts through iterative checking"""
        current_objects = all_objects.copy()
        max_iterations = 5
        
        for iteration in range(max_iterations):
            compatibility_result = self._check_object_compatibility(current_objects)
            
            if compatibility_result.get("compatible", True):
                print(f"   ✅ All objects are compatible")
                break
            
            print(f"   ⚠️  Found incompatibilities, resolving...")
            
            # Extract conflict information and decide removal strategy
            main_conflicts = compatibility_result.get("main_object_conflicts", [])
            secondary_conflicts = compatibility_result.get("secondary_object_conflicts", [])
            
            to_remove = self._decide_removal_strategy(main_object, current_objects, main_conflicts, secondary_conflicts)
            
            if to_remove:
                for obj in to_remove:
                    if obj in current_objects:
                        current_objects.remove(obj)
                print(f"   🗑️  Removed: {[self.coco_objects[obj] for obj in to_remove]}")
            else:
                break
        
        return current_objects

    def _check_object_compatibility(self, objects: List[int]) -> Dict[str, Any]:
        """Check object compatibility using LLM"""
        object_names = [self.coco_objects[obj] for obj in objects]
        
        user_prompt = f"Evaluate compatibility of these objects in a single artwork:\nMain Object: {object_names[0]}\nSecondary Objects: [{', '.join(object_names[1:])}]"
        
        try:
            response = chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': self.compatibility_system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            
            # Handle different response formats
            response_text = self._extract_response_content(response)
            
            # Remove thinking tags if present
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            # Parse JSON response
            return self._parse_json_response(response_text)
            
        except Exception as e:
            print(f"   ⚠️  Compatibility check error: {e}")
            return {"compatible": True}

    def _decide_removal_strategy(self, main_object: int, current_objects: List[int], 
                               main_conflicts: List[Dict], secondary_conflicts: List[Dict]) -> List[int]:
        """Decide which objects to remove based on conflicts"""
        to_remove = set()
        main_object_name = self.coco_objects[main_object]
        
        # Handle main object conflicts
        if main_conflicts:
            for conflict in main_conflicts:
                conflicting_name = None
                if conflict["object1"] == main_object_name:
                    conflicting_name = conflict["object2"]
                elif conflict["object2"] == main_object_name:
                    conflicting_name = conflict["object1"]
                
                if conflicting_name:
                    for obj_id in current_objects:
                        if self.coco_objects[obj_id] == conflicting_name and obj_id != main_object:
                            to_remove.add(obj_id)
        
        # Handle secondary object conflicts
        if secondary_conflicts and not to_remove:
            conflict_counter = Counter()
            
            for conflict in secondary_conflicts:
                obj1_name = conflict["object1"]
                obj2_name = conflict["object2"]
                
                for obj_id in current_objects:
                    if obj_id != main_object:
                        if self.coco_objects[obj_id] == obj1_name:
                            conflict_counter[obj_id] += 1
                        elif self.coco_objects[obj_id] == obj2_name:
                            conflict_counter[obj_id] += 1
            
            if conflict_counter:
                max_conflicts = max(conflict_counter.values())
                most_conflicted = [obj_id for obj_id, count in conflict_counter.items() 
                                 if count == max_conflicts]
                to_remove.add(random.choice(most_conflicted))
        
        return list(to_remove)

    def _generate_scene_design(self, object_ids: List[int], style: str) -> Dict[str, Any]:
        """Generate four-stage scene design"""
        object_names = [self.coco_objects[obj_id] for obj_id in object_ids]
        
        print(f"   🎬 Stage 1: Scene Framework")
        stage1_data = self._stage1_scene_framework(object_names, style)
        
        print(f"   🏗️  Stage 2: Spatial Layout")
        stage2_data = self._stage2_spatial_layout(object_names, style, stage1_data)
        
        print(f"   🌍 Stage 3: Environment Details")
        stage3_data = self._stage3_environment_details(object_names, style, stage1_data, stage2_data)
        
        print(f"   🎨 Stage 4: Artistic Expression")
        stage4_data = self._stage4_artistic_expression(object_names, style, stage1_data, stage2_data, stage3_data)
        
        return {
            "stage1_scene_framework": stage1_data,
            "stage2_spatial_layout": stage2_data,
            "stage3_environment_details": stage3_data,
            "stage4_artistic_expression": stage4_data
        }

    def _stage1_scene_framework(self, objects: List[str], style: str) -> Dict[str, Any]:
        """Stage 1: Determine scene framework"""
        system_prompt = """
You are an art composition expert. Determine the basic scene framework based on objects and art style.

Analyze and determine:
1. Composition type (portrait, landscape, still life, genre scene, historical scene)
2. Environment setting (indoor/outdoor)
3. Scene nature (intimate, grand, dramatic, peaceful, etc.)

Respond ONLY with valid JSON:
{
    "composition_type": "portrait/landscape/still_life/genre_scene/historical_scene",
    "environment": "indoor/outdoor/mixed",
    "scene_nature": "description of overall mood/nature",
    "reasoning": "brief explanation"
}
"""
        
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\n\nDetermine appropriate scene framework."
        
        return self._query_llm_json(system_prompt, user_prompt)

    def _stage2_spatial_layout(self, objects: List[str], style: str, stage1_data: Dict) -> Dict[str, Any]:
        """Stage 2: Design spatial layout and relationships"""
        system_prompt = """
You are an art composition expert. Design spatial layout and relationships based on objects, style, and scene framework.

Determine:
1. Spatial relationships between objects
2. Viewing perspective/angle
3. Object states and quantities
4. Compositional arrangement

Respond ONLY with valid JSON:
{
    "spatial_relationships": "description of spatial relations",
    "viewing_perspective": "frontal/three-quarter/profile/bird's eye/worm's eye view",
    "object_states": {
        "object_name": {"quantity": number, "state": "description", "position": "foreground/midground/background"}
    },
    "compositional_arrangement": "composition description (triangular, circular, diagonal, etc.)"
}
"""
        
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\nScene Framework: {json.dumps(stage1_data)}\n\nDesign spatial layout."
        
        return self._query_llm_json(system_prompt, user_prompt)

    def _stage3_environment_details(self, objects: List[str], style: str, stage1_data: Dict, stage2_data: Dict) -> Dict[str, Any]:
        """Stage 3: Specify environmental details"""
        system_prompt = """
You are an art environment designer. Specify detailed environmental conditions based on previous stages.

Determine:
1. Specific location/setting
2. Time period/time of day
3. Weather conditions (if applicable)
4. Lighting conditions and quality

Respond ONLY with valid JSON:
{
    "specific_location": "detailed setting description",
    "time_period": "historical period or contemporary",
    "time_of_day": "morning/afternoon/evening/night",
    "weather": "sunny/cloudy/rainy/stormy/calm (if outdoor)",
    "lighting": {
        "type": "natural/artificial/mixed",
        "quality": "soft/harsh/dramatic/even",
        "direction": "front-lit/back-lit/side-lit",
        "mood": "warm/cool/neutral"
    }
}
"""
        
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\nPrevious stages: {json.dumps({'stage1': stage1_data, 'stage2': stage2_data})}\n\nSpecify environmental conditions."
        
        return self._query_llm_json(system_prompt, user_prompt)

    def _stage4_artistic_expression(self, objects: List[str], style: str, stage1_data: Dict, stage2_data: Dict, stage3_data: Dict) -> Dict[str, Any]:
        """Stage 4: Determine artistic expression elements"""
        system_prompt = """
You are an art expression specialist. Determine artistic expression elements based on all previous stages.

Determine:
1. Color palette and tones
2. Emotional theme and mood
3. Symbolic relationships and meanings
4. Final artistic prompt for image generation

Respond ONLY with valid JSON:
{
    "color_palette": {
        "primary_colors": ["color1", "color2", "color3"],
        "secondary_colors": ["color1", "color2"],
        "overall_tone": "warm/cool/neutral",
        "saturation": "high/medium/low",
        "brightness": "bright/medium/dark"
    },
    "emotional_theme": "main emotional expression",
    "symbolic_meanings": {
        "object_name": "symbolic meaning or role"
    },
    "artistic_techniques": "specific techniques relevant to style",
    "final_prompt": "comprehensive prompt for image generation, NO MORE THAN 150 WORDS"
}
"""
        
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\nAll previous stages: {json.dumps({'stage1': stage1_data, 'stage2': stage2_data, 'stage3': stage3_data})}\n\nDetermine artistic expression and create final prompt."
        
        return self._query_llm_json(system_prompt, user_prompt)

    def _extract_response_content(self, response) -> str:
        """Extract content from different response formats"""
        if hasattr(response, 'message'):
            return response.message.content.strip()
        elif isinstance(response, dict) and 'message' in response:
            return response['message']['content'].strip()
        elif isinstance(response, dict) and 'content' in response:
            return response['content'].strip()
        else:
            return str(response)

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from response text using multiple patterns"""
        patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'```\s*(\{.*?\})\s*```',
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, dict):
                        return data
                except:
                    continue
        
        return {"compatible": True}  # Default to compatible if parsing fails

    def _query_llm_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """Query LLM and parse JSON response"""
        try:
            response = chat(
                model=self.model_name,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}
                ]
            )
            
            response_text = self._extract_response_content(response)
            response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            return self._parse_json_response(response_text)
            
        except Exception as e:
            return {"error": str(e)}

    def batch_generate(self, count: int, output_dir: str = "artworks", 
                      styles: Optional[List[str]] = None, 
                      max_secondary_objects: int = 8) -> List[str]:
        """Generate multiple artworks in batch"""
        print(f"🚀 Starting batch generation of {count} artworks")
        
        generated_files = []
        for i in range(count):
            style = random.choice(styles) if styles else None
            print(f"\n📊 Progress: {i+1}/{count}")
            
            try:
                artwork_data = self.generate_artwork(
                    style=style,
                    max_secondary_objects=max_secondary_objects,
                    output_dir=output_dir
                )
                generated_files.append(f"{output_dir}/{artwork_data['artwork_id']}.json")
            except Exception as e:
                print(f"❌ Error generating artwork {i+1}: {e}")
                continue
        
        print(f"\n🎉 Batch generation completed!")
        print(f"📁 Generated {len(generated_files)} artworks in '{output_dir}' directory")
        return generated_files


def main():
    """Main function for demonstration"""
    # Create pipeline instance
    pipeline = ARTOArtworkPipeline(model_name='deepseek-r1:70b')
    
    print("🎨 ARTO-Guided Artwork Generation Pipeline")
    print("=" * 60)
    
    # Single artwork generation example
    print("📝 Single artwork generation:")
    artwork = pipeline.generate_artwork(
        style="Post-Impressionism",
        max_secondary_objects=6,
        output_dir="generated_artworks"
    )
    
    print("\n" + "=" * 60)
    
    # Batch generation example
    print("📦 Batch generation example:")
    generated_files = pipeline.batch_generate(
        count=5,
        output_dir="batch_artworks",
        styles=["Post-Impressionism", "Sketch", "Chinese Ink Painting", "Photorealistic", "Oil Painting"],
        max_secondary_objects=8
    )
    
    print(f"\n📋 Generated files: {generated_files}")


if __name__ == "__main__":
    main()
