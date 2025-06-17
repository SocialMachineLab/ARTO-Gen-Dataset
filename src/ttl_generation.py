import json
import os
import glob
import uuid
import re
from datetime import datetime
from ollama import chat
from typing import Dict, List, Optional


class ARTOTurtleGenerator:
    """Generator for ARTO-compliant TTL (Turtle) RDF files from artwork JSON data"""
    
    def __init__(self, model_name: str = "deepseek-r1:70b"):
        self.model_name = model_name
        
        # Default color mapping for common colors
        self.default_colors = {
            "red": {"rgb": "255,0,0", "hsv": "0,100,100"},
            "blue": {"rgb": "0,0,255", "hsv": "240,100,100"},
            "green": {"rgb": "0,255,0", "hsv": "120,100,100"},
            "yellow": {"rgb": "255,255,0", "hsv": "60,100,100"},
            "brown": {"rgb": "139,69,19", "hsv": "25,86,55"},
            "white": {"rgb": "255,255,255", "hsv": "0,0,100"},
            "black": {"rgb": "0,0,0", "hsv": "0,0,0"},
            "orange": {"rgb": "255,165,0", "hsv": "39,100,100"},
            "purple": {"rgb": "128,0,128", "hsv": "300,100,50"},
            "pink": {"rgb": "255,192,203", "hsv": "350,25,100"}
        }
    
    def query_llm(self, prompt: str) -> str:
        """Query LLM with error handling"""
        try:
            response = chat(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Handle different response formats
            if hasattr(response, 'message'):
                return response.message.content
            elif isinstance(response, dict) and 'message' in response:
                return response['message']['content']
            else:
                return str(response)
                
        except Exception as e:
            print(f"  ⚠️ LLM query failed: {str(e)}")
            return ""
    
    def generate_artwork_metadata(self, scene_data: Dict) -> Dict:
        """Generate artwork title, description, and medium using LLM"""
        style = scene_data.get('style', 'Unknown')
        objects = scene_data.get('selected_objects', {}).get('object_names', [])
        
        # Get scene description for context
        final_prompt = ""
        if 'scene_design' in scene_data:
            final_prompt = scene_data['scene_design'].get('stage4_artistic_expression', {}).get('final_prompt', '')
        
        # Create LLM prompt for artwork metadata
        llm_prompt = f"""Generate artwork metadata based on the following information:
Style: {style}
Objects: {', '.join(objects)}
Scene Description: {final_prompt[:300] if final_prompt else 'No description available'}

Return ONLY a valid JSON object with this exact format:
{{"title": "descriptive artwork title", "description": "brief artwork description", "medium": "appropriate medium for the style"}}"""
        
        response = self.query_llm(llm_prompt)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response using regex
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                artwork_info = json.loads(json_match.group())
                return artwork_info
        except:
            pass
        
        # Fallback if LLM fails
        return {
            "title": f"{style} Composition with {len(objects)} Elements",
            "description": f"A {style.lower()} artwork featuring {', '.join(objects[:3])}{'...' if len(objects) > 3 else ''}",
            "medium": self._get_default_medium(style)
        }
    
    def generate_object_attributes(self, object_name: str, style: str) -> Dict:
        """Generate detailed attributes for each object using LLM"""
        llm_prompt = f"""Generate detailed attributes for the object "{object_name}" in a {style} artwork.

Return ONLY a valid JSON object with this exact format:
{{"descriptor": "detailed visual description", "state": "object's condition/state", "material": "likely material"}}"""
        
        response = self.query_llm(llm_prompt)
        
        # Try to parse JSON response
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                obj_info = json.loads(json_match.group())
                return obj_info
        except:
            pass
        
        # Fallback attributes
        return {
            "descriptor": f"A {object_name} depicted in {style.lower()} style",
            "state": "positioned within the scene",
            "material": "unknown"
        }
    
    def extract_color_palette(self, scene_data: Dict) -> List[str]:
        """Extract color palette from scene design data"""
        colors = []
        
        if 'scene_design' in scene_data:
            stage4 = scene_data['scene_design'].get('stage4_artistic_expression', {})
            color_palette = stage4.get('color_palette', {})
            
            if color_palette:
                primary_colors = color_palette.get('primary_colors', [])
                secondary_colors = color_palette.get('secondary_colors', [])
                colors = primary_colors + secondary_colors
        
        # Use default colors if none found
        if not colors:
            colors = ["red", "blue", "green", "yellow", "brown"]
        
        # Limit to 5 colors and clean color names
        return [self._clean_color_name(color) for color in colors[:5]]
    
    def _clean_color_name(self, color: str) -> str:
        """Clean and normalize color names"""
        if not color:
            return "unknown"
        
        # Convert to lowercase and remove extra spaces
        clean_color = color.lower().strip()
        
        # Handle compound color names
        if ' ' in clean_color:
            clean_color = clean_color.replace(' ', '_')
        
        return clean_color
    
    def _get_default_medium(self, style: str) -> str:
        """Get appropriate medium based on art style"""
        medium_mapping = {
            'oil painting': 'Oil on canvas',
            'chinese ink painting': 'Ink on paper',
            'sketch': 'Pencil on paper',
            'photorealistic': 'Digital art',
            'post-impressionism': 'Oil on canvas',
            'watercolor': 'Watercolor on paper',
            'acrylic': 'Acrylic on canvas'
        }
        
        style_lower = style.lower()
        for key, medium in medium_mapping.items():
            if key in style_lower:
                return medium
        
        return "Mixed media"
    
    def _get_color_values(self, color_name: str) -> Dict[str, str]:
        """Get RGB and HSV values for a color"""
        clean_color = color_name.lower().replace('_', ' ').strip()
        
        # Check if we have predefined values
        if clean_color in self.default_colors:
            return self.default_colors[clean_color]
        
        # Default fallback values
        return {"rgb": "128,128,128", "hsv": "0,0,50"}
    
    def generate_ttl_content(self, scene_data: Dict) -> str:
        """Generate complete TTL content for the artwork"""
        # Extract basic information
        artwork_id = scene_data.get('artwork_id', f"artwork_{uuid.uuid4().hex[:8]}")
        style = scene_data.get('style', 'Unknown')
        objects = scene_data.get('selected_objects', {}).get('object_names', [])
        timestamp = scene_data.get('generation_timestamp', datetime.now().isoformat())
        
        # Get scene description
        final_prompt = ""
        if 'scene_design' in scene_data:
            final_prompt = scene_data['scene_design'].get('stage4_artistic_expression', {}).get('final_prompt', '')
        
        # Generate artwork metadata
        artwork_info = self.generate_artwork_metadata(scene_data)
        
        # Generate object details
        objects_info = []
        for obj_name in objects:
            obj_attributes = self.generate_object_attributes(obj_name, style)
            obj_attributes['id'] = f"{obj_name.replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            obj_attributes['label'] = obj_name
            objects_info.append(obj_attributes)
        
        # Extract colors
        colors = self.extract_color_palette(scene_data)
        
        # Build TTL content
        ttl_content = self._build_ttl_header(timestamp)
        ttl_content += self._build_artwork_section(artwork_id, artwork_info, style, timestamp)
        ttl_content += self._build_scene_section(artwork_id, final_prompt)
        ttl_content += self._build_objects_section(objects_info)
        ttl_content += self._build_colors_section(colors)
        
        return ttl_content
    
    def _build_ttl_header(self, timestamp: str) -> str:
        """Build TTL file header with prefixes"""
        return f"""@prefix : <http://w3id.org/artwork/> .
@prefix arto: <http://w3id.org/arto#> .
@prefix dc: <http://purl.org/dc/terms/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix sdo: <https://schema.org/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

# Generated on {timestamp[:10]} using ARTO Ontology
# Source: ARTO-Guided Artwork Generation Pipeline

"""
    
    def _build_artwork_section(self, artwork_id: str, artwork_info: Dict, style: str, timestamp: str) -> str:
        """Build artwork RDF section"""
        return f"""# Artwork Definition
:{artwork_id} a arto:Painting ;
    dc:title "{self._escape_string(artwork_info['title'])}" ;
    dc:creator "AI Generated" ;
    arto:style "{style}" ;
    dc:medium "{artwork_info['medium']}" ;
    dc:description "{self._escape_string(artwork_info['description'])}" ;
    dc:created "{timestamp[:10]}"^^xsd:date ;
    sdo:height "1024px" ;
    sdo:width "1024px" .

"""
    
    def _build_scene_section(self, artwork_id: str, final_prompt: str) -> str:
        """Build scene RDF section"""
        scene_description = final_prompt[:200] + "..." if len(final_prompt) > 200 else final_prompt
        
        return f"""# Scene Definition
:main_scene a arto:Scene ;
    rdfs:label "Primary Scene" ;
    arto:descriptor "{self._escape_string(scene_description)}" ;
    arto:coordinates "0,0,1024,1024" ;
    arto:size "full" .

:{artwork_id} arto:containsScene :main_scene .

"""
    
    def _build_objects_section(self, objects_info: List[Dict]) -> str:
        """Build objects RDF section"""
        section = "# Object Definitions\n"
        
        for obj in objects_info:
            section += f""":{obj['id']} a arto:Object ;
    rdfs:label "{self._escape_string(obj['label'])}" ;
    arto:descriptor "{self._escape_string(obj['descriptor'])}" ;
    arto:state "{self._escape_string(obj['state'])}" ;
    arto:material "{self._escape_string(obj['material'])}" ;
    arto:coordinates "auto" ;
    arto:size "medium" .

:main_scene arto:containsObject :{obj['id']} .

"""
        
        return section
    
    def _build_colors_section(self, colors: List[str]) -> str:
        """Build colors RDF section"""
        section = "# Color Definitions\n"
        
        for color in colors:
            color_id = f"{color.replace(' ', '_')}_color"
            color_values = self._get_color_values(color)
            
            section += f""":{color_id} a arto:Color ;
    rdfs:label "{color}" ;
    arto:RGBValue "{color_values['rgb']}" ;
    arto:HSVValue "{color_values['hsv']}" .

:main_scene arto:containsElement :{color_id} .

"""
        
        return section
    
    def _escape_string(self, text: str) -> str:
        """Escape special characters in strings for TTL format"""
        if not text:
            return ""
        
        # Replace quotes and other special characters
        escaped = text.replace('"', '\\"').replace('\n', ' ').replace('\r', ' ')
        # Remove multiple spaces
        escaped = re.sub(r'\s+', ' ', escaped).strip()
        
        return escaped
    
    def process_single_file(self, json_file_path: str, output_dir: str, skip_existing: bool = True) -> Dict:
        """Process a single JSON file and generate TTL"""
        filename = os.path.basename(json_file_path)
        
        try:
            # Load scene data
            with open(json_file_path, 'r', encoding='utf-8') as f:
                scene_data = json.load(f)
            
            # Get artwork ID
            artwork_id = scene_data.get('artwork_id', f"artwork_{uuid.uuid4().hex[:8]}")
            output_file = os.path.join(output_dir, f"{artwork_id}.ttl")
            
            # Check if file already exists
            if skip_existing and os.path.exists(output_file):
                return {
                    'status': 'skipped',
                    'artwork_id': artwork_id,
                    'message': 'TTL file already exists'
                }
            
            # Generate TTL content
            ttl_content = self.generate_ttl_content(scene_data)
            
            # Save TTL file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(ttl_content)
            
            return {
                'status': 'success',
                'artwork_id': artwork_id,
                'output_file': output_file,
                'objects_count': len(scene_data.get('selected_objects', {}).get('object_names', [])),
                'style': scene_data.get('style', 'Unknown')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'artwork_id': 'unknown',
                'error': str(e),
                'file': filename
            }
    
    def process_batch(self, input_dir: str, output_dir: str, skip_existing: bool = True) -> Dict:
        """Process all JSON files in input directory"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        
        if not json_files:
            print(f"❌ No JSON files found in {input_dir}")
            return {'status': 'error', 'message': 'No JSON files found'}
        
        print(f"🔍 Found {len(json_files)} JSON files")
        print(f"📂 Input directory: {input_dir}")
        print(f"📂 Output directory: {output_dir}")
        print(f"⏭️ Skip existing: {'Yes' if skip_existing else 'No'}")
        print("=" * 60)
        
        # Process files
        results = {
            'total_files': len(json_files),
            'successful': 0,
            'skipped': 0,
            'failed': 0,
            'details': [],
            'style_distribution': {},
            'object_count_distribution': {}
        }
        
        for i, json_file in enumerate(json_files, 1):
            filename = os.path.basename(json_file)
            print(f"[{i}/{len(json_files)}] Processing: {filename}")
            
            result = self.process_single_file(json_file, output_dir, skip_existing)
            results['details'].append(result)
            
            if result['status'] == 'success':
                results['successful'] += 1
                print(f"  ✅ Generated: {result['artwork_id']}.ttl")
                
                # Update statistics
                style = result.get('style', 'Unknown')
                obj_count = result.get('objects_count', 0)
                
                results['style_distribution'][style] = results['style_distribution'].get(style, 0) + 1
                results['object_count_distribution'][obj_count] = results['object_count_distribution'].get(obj_count, 0) + 1
                
            elif result['status'] == 'skipped':
                results['skipped'] += 1
                print(f"  ⏭️ Skipped: {result['artwork_id']}.ttl (already exists)")
                
            else:  # error
                results['failed'] += 1
                print(f"  ❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Print summary
        print("=" * 60)
        print("📊 PROCESSING SUMMARY:")
        print(f"✅ Successfully generated: {results['successful']} TTL files")
        print(f"⏭️ Skipped existing: {results['skipped']} files")
        print(f"❌ Failed: {results['failed']} files")
        print(f"📈 Success rate: {(results['successful'] / len(json_files) * 100):.1f}%")
        
        if results['style_distribution']:
            print(f"🎨 Style distribution: {dict(results['style_distribution'])}")
        
        # Save processing report
        report_file = os.path.join(output_dir, "ttl_generation_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Detailed report saved: {report_file}")
        
        return results


def main():
    """Main execution function"""
    # Configuration
    INPUT_DIRECTORY = "./batch_artworks"    # Directory containing JSON files
    OUTPUT_DIRECTORY = "./output_ttl"       # Directory for generated TTL files
    MODEL_NAME = "deepseek-r1:70b"          # LLM model for metadata generation
    SKIP_EXISTING = True                    # Skip files that already have TTL output
    
    print("🐢 ARTO TTL GENERATOR")
    print("=" * 60)
    print("Generating RDF/Turtle files from ARTO-structured artwork data")
    print("=" * 60)
    
    # Create generator instance
    generator = ARTOTurtleGenerator(model_name=MODEL_NAME)
    
    # Process files
    results = generator.process_batch(
        input_dir=INPUT_DIRECTORY,
        output_dir=OUTPUT_DIRECTORY,
        skip_existing=SKIP_EXISTING
    )
    
    # Final summary
    if results.get('status') != 'error':
        print(f"\n🎉 TTL GENERATION COMPLETED!")
        print(f"📊 Total TTL files: {results['successful'] + results['skipped']}")
        print(f"📁 Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()