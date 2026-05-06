"""
ARTO Mapper Module
Convert generated JSON artwork data to ARTO (Artwork Object Ontology) TTL format
Focus on basic mapping of Scene, Object, VisualElement
"""

import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import quote
from arto_kg.conceptualization.utils import setup_logger
import os

class ARTOMapper:
    """Map JSON artwork data to ARTO ontology TTL format"""
    
    def __init__(self):
        self.logger = setup_logger("arto_mapper")
        
        # ARTO namespace and prefixes
        self.namespaces = {
            "arto": "http://w3id.org/arto#",
            "dc": "http://purl.org/dc/terms/",
            "sdo": "https://schema.org/",
            "edm": "http://www.europeana.eu/schemas/edm/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "qudt": "http://qudt.org/schema/qudt/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "": "http://w3id.org/arto/resource/"
        }
        
        # Mapping from art style to ARTO StyleType
        self.style_mappings = {
            "Renaissance": "arto:RenaissanceStyle",
            "Baroque": "arto:BaroqueStyle", 
            "Impressionism": "arto:ImpressionismStyle",
            "Impressionist": "arto:ImpressionismStyle",
            "Romantic": "arto:RomanticismStyle",
            "Realism": "arto:RealismStyle",
            "Abstract": "arto:AbstractStyle",
            "Surrealism": "arto:SurrealismStyle",
            "Cubism": "arto:CubismStyle",
            "Pop Art": "arto:PopArtStyle",
            "Minimalism": "arto:MinimalismStyle",
            "Classical": "arto:ClassicalStyle",
            "Gothic": "arto:GothicStyle",
            "Art Nouveau": "arto:ArtNouveauStyle",
            "Expressionism": "arto:ExpressionismStyle",
            "Fauvism": "arto:FauvismStyle",
            "Post-Impressionism": "arto:PostImpressionismStyle",
            "Chinese Ink Painting": "arto:ChineseInkPaintingStyle",
            "Photorealistic": "arto:PhotorealisticStyle"
        }
        
        # Composition type mapping
        self.composition_type_mappings = {
            "portrait": "arto:PortraitGenre",
            "landscape": "arto:LandscapeGenre", 
            "still_life": "arto:StillLifeGenre",
            "genre_scene": "arto:GenreSceneGenre",
            "historical": "arto:HistoricalGenre"
        }
        
        # Object category mapping (based on COCO categories)
        self.object_category_mappings = {
            "person": "arto:Character",
            "animal": "arto:Animal", 
            "vehicle": "arto:Vehicle",
            "furniture": "arto:Furniture",
            "food": "arto:Food",
            "plant": "arto:Plant",
            "utility": "arto:UtilityObject",
            "decorative": "arto:DecorativeObject",
            "technology": "arto:TechnologicalObject"
        }

    def convert_artwork_to_ttl(self, json_data: Dict[str, Any], od_data: Optional[Dict[str, Any]] = None, validation_data: Optional[Dict[str, Any]] = None, output_path: Optional[str] = None) -> str:
        """
        Convert single JSON artwork data to ARTO TTL format, integrating OD and validation data
        
        Args:
            json_data: Archive JSON data
            od_data: Object detection data (optional)
            validation_data: Validation data (optional)
            output_path: Output TTL file path (optional)
            
        Returns:
            TTL format string
        """
        artwork_id = json_data.get('artwork_id', 'unknown')
        self.logger.info(f"Converting artwork {artwork_id} to ARTO TTL")
        
        try:
            ttl_content = self._generate_ttl_content(json_data, od_data, validation_data)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(ttl_content)
                self.logger.info(f"ARTO TTL saved to {output_path}")
            
            return ttl_content
            
        except Exception as e:
            self.logger.error(f"Failed to convert {artwork_id} to TTL: {e}")
            raise

    def _generate_ttl_content(self, json_data: Dict[str, Any], od_data: Optional[Dict[str, Any]] = None, validation_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate complete TTL content"""
        ttl_lines = []
        
        # TTL Header - Namespace and Metadata
        ttl_lines.extend(self._generate_header(json_data))
        
        # Generate artwork entity
        artwork_uri = self._generate_artwork_uri(json_data)
        ttl_lines.extend(self._generate_artwork_triples(json_data, artwork_uri))
        
        # Generate scene entity
        main_scene_uri = self._generate_scene_triples(json_data, artwork_uri, ttl_lines)
        
        # Generate object entities
        self._generate_objects_triples(json_data, main_scene_uri, ttl_lines, od_data)
        
        # Generate visual element entities
        self._generate_visual_elements_triples(json_data, main_scene_uri, ttl_lines, validation_data, od_data)
        
        # Generate spatial relation/connectivity entities
        self._generate_spatial_connectivity(json_data, main_scene_uri, ttl_lines, validation_data)
        
        return "\n".join(ttl_lines)

    def _generate_header(self, json_data: Dict[str, Any]) -> List[str]:
        """Generate TTL file header"""
        lines = []
        
        lines.append("# ARTO TTL - Generated from AI Artwork JSON")
        lines.append(f"# Artwork ID: {json_data.get('artwork_id', 'unknown')}")
        lines.append(f"# Generated: {datetime.now().isoformat()}")
        lines.append("")
        
        # Namespace declaration
        for prefix, namespace in self.namespaces.items():
            lines.append(f"@prefix {prefix}: <{namespace}> .")
        lines.append("")

        # Declare all spatial/semantic predicates as subproperties of arto:relatedTo
        for pred in ("nextTo", "beside", "behind", "above", "below", "under", "on",
                     "near", "inside", "between", "contains", "surrounding",
                     "leftOf", "rightOf", "inFrontOf", "partOf", "watching", "holding"):
            lines.append(f"arto:{pred} rdfs:subPropertyOf arto:relatedTo .")
        lines.append("")

        return lines

    def _generate_artwork_uri(self, json_data: Dict[str, Any]) -> str:
        """Generate artwork URI"""
        artwork_id = json_data.get("artwork_id", f"artwork_{uuid.uuid4().hex[:8]}")
        safe_id = self._safe_uri_name(artwork_id)
        return f":artwork_{safe_id}"

    def _generate_artwork_triples(self, json_data: Dict[str, Any], artwork_uri: str) -> List[str]:
        """Generate artwork RDF triples"""
        triples = []
        
        # Basic type declaration
        triples.append(f"{artwork_uri} rdf:type arto:Artwork ;")
        
        # Title
        artwork_id = json_data.get("artwork_id", "Untitled Artwork")
        title = artwork_id.replace("_", " ").title()
        triples.append(f'    dc:title "{self._escape_literal(title)}"@en ;')
        
        # Description - Prioritize Main Prompt as dc:description
        description = None
        if json_data.get("final_prompts") and json_data["final_prompts"].get("main_prompt"):
            description = json_data["final_prompts"]["main_prompt"]
        else:
            description = self._build_artwork_description(json_data)
        
        if description:
            # Simple quote handling to avoid syntax errors
            safe_desc = self._escape_literal(description)
            # For long descriptions, use multi-line string format
            if len(safe_desc) > 50:
                 triples.append(f'    dc:description """{safe_desc}"""@en ;')
            else:
                 triples.append(f'    dc:description "{safe_desc}"@en ;')
        
        # Art style
        style = json_data.get("style")
        if style:
            style_uri = self.style_mappings.get(style, f"arto:{self._safe_uri_name(style)}Style")
            triples.append(f"    arto:style {style_uri} ;")
        
        # Creation info
        if json_data.get("generation_timestamp"):
            triples.append(f'    dc:created "{json_data["generation_timestamp"]}"^^xsd:dateTime ;')
        
        # Creator info
        triples.append('    dc:creator "AI Artwork Generation System" ;')
        
        # Genre - Inferred from composition info
        genre = self._infer_genre_from_composition(json_data)
        if genre:
            triples.append(f"    arto:genre {genre} ;")
        
        # Medium type - Digital art
        triples.append("    arto:medium arto:DigitalMedium ;")
        
        # Remove last semicolon, add period
        if triples and triples[-1].endswith(' ;'):
            triples[-1] = triples[-1].rstrip(' ;') + ' .'
        
        triples.append("")
        return triples

    def _generate_scene_triples(self, json_data: Dict[str, Any], artwork_uri: str, ttl_lines: List[str]) -> str:
        """Generate scene RDF triples"""
        main_scene_uri = f"{artwork_uri}_scene"
        
        # Basic scene declaration
        ttl_lines.append(f"{main_scene_uri} rdf:type arto:Scene ;")
        ttl_lines.append(f'    rdfs:label "Main Scene"@en ;')
        
        # Scene description
        scene_description = self._build_scene_description(json_data)
        if scene_description:
            ttl_lines.append(f'    dc:description "{self._escape_literal(scene_description)}"@en ;')
        
        # Composition info
        composition = json_data.get("composition", {})
        
        # Spatial relationship info
        spatial_relationships = composition.get("spatial_relationships", {})
        if spatial_relationships:
            # Depth hierarchy
            depth_arrangement = spatial_relationships.get("depth_arrangement", {})
            if depth_arrangement:
                ttl_lines.append(f'    arto:hasDepthArrangement "{self._describe_depth_arrangement(depth_arrangement)}"@en ;')
            
            # Scale hierarchy
            scale_hierarchy = spatial_relationships.get("scale_hierarchy", {})
            if scale_hierarchy:
                ttl_lines.append(f'    arto:hasScaleHierarchy "{self._describe_scale_hierarchy(scale_hierarchy)}"@en ;')
        
        # Composition structure info
        composition_structure = composition.get("composition_structure", {})
        if composition_structure:
            # Visual balance
            visual_balance = composition_structure.get("visual_balance", {})
            if visual_balance.get("balance_type"):
                ttl_lines.append(f'    arto:hasBalanceType "{visual_balance["balance_type"]}"@en ;')
            
            # Focal hierarchy
            focal_hierarchy = composition_structure.get("focal_hierarchy", {})
            if focal_hierarchy.get("primary_focus"):
                ttl_lines.append(f'    arto:hasPrimaryFocus "{self._escape_literal(focal_hierarchy["primary_focus"])}"@en ;')
        
        # Environment info
        environment = json_data.get("environment", {}).get("environment_details", {})
        if environment:
            # Lighting conditions
            lighting = environment.get("lighting", {})
            if lighting:
                lighting_desc = f"{lighting.get('quality', 'natural')} {lighting.get('type', 'light')}"
                ttl_lines.append(f'    arto:hasLighting "{lighting_desc}"@en ;')
            
            # Time setting
            if environment.get("time_of_day"):
                ttl_lines.append(f'    arto:hasTimeOfDay "{environment["time_of_day"]}"@en ;')
            
            # Weather conditions
            if environment.get("weather") and environment["weather"] != "not applicable":
                ttl_lines.append(f'    arto:hasWeather "{environment["weather"]}"@en ;')
        
        # Remove last semicolon
        if ttl_lines and ttl_lines[-1].endswith(' ;'):
            ttl_lines[-1] = ttl_lines[-1].rstrip(' ;') + ' .'
        
        ttl_lines.append("")
        
        # Connect artwork to scene
        ttl_lines.append(f"{artwork_uri} arto:containsScene {main_scene_uri} .")
        ttl_lines.append("")
        
        return main_scene_uri

    def _generate_objects_triples(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], od_data: Optional[Dict[str, Any]] = None):
        """Generate object RDF triples"""
        enhanced_objects = json_data.get("objects", {}).get("enhanced_objects", [])
        
        # Build OD data index (Label -> BBox)
        od_map = {}
        if od_data and "detected_objects" in od_data:
            for obj in od_data["detected_objects"]:
                label = obj.get("label", "").lower()
                box = obj.get("box")
                if label and box:
                    # If multiple labels, simply take the last one, or list all bboxes
                    od_map[label] = box

        for i, obj_data in enumerate(enhanced_objects):
            obj_name = obj_data.get("name", f"object_{i}")
            obj_uri = f"{scene_uri}_obj_{self._safe_uri_name(obj_name)}"
            
            # Basic object declaration
            ttl_lines.append(f"{obj_uri} rdf:type arto:Object ;")
            ttl_lines.append(f'    rdfs:label "{self._escape_literal(obj_name.title())}"@en ;')
            
            # Artistic description
            if obj_data.get("artistic_description"):
                desc = obj_data["artistic_description"]
                ttl_lines.append(f'    dc:description "{self._escape_literal(desc)}"@en ;')
            
            # Material info - support flat string (actual JSON) and nested dict (extended format)
            material = obj_data.get("material") or obj_data.get("material_texture", {}).get("primary_material")
            if material:
                ttl_lines.append(f'    arto:material [ a arto:MaterialType ; rdfs:label "{self._escape_literal(str(material))}"@en ] ;')

            # Physical state - support flat string (actual JSON) and nested dict (extended format)
            state = obj_data.get("state") or obj_data.get("physical_condition", {}).get("overall_state")
            if state:
                ttl_lines.append(f'    arto:state [ a skos:Concept ; rdfs:label "{self._escape_literal(str(state))}"@en ] ;')
            
            # Symbolic meaning
            symbolic_meaning = obj_data.get("symbolic_meaning", {})
            if symbolic_meaning.get("symbolic_interpretation"):
                symbol = symbolic_meaning["symbolic_interpretation"]
                ttl_lines.append(f'    arto:hasSymbolicMeaning "{self._escape_literal(symbol)}"@en ;')
            
            # Integrate BBox coordinates
            if obj_name.lower() in od_map:
                bbox = od_map[obj_name.lower()]
                # bbox format: [x1, y1, x2, y2]
                vector_str = f"({bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]})"
                ttl_lines.append(f"    arto:hasCoordinates [")
                ttl_lines.append(f"        a qudt:Vector ;")
                ttl_lines.append(f"        qudt:vector {vector_str}")
                ttl_lines.append(f"    ] ;")

            # Remove last semicolon
            if ttl_lines and ttl_lines[-1].endswith(' ;'):
                ttl_lines[-1] = ttl_lines[-1].rstrip(' ;') + ' .'
            
            ttl_lines.append("")
            
            # Connect scene to objects
            ttl_lines.append(f"{scene_uri} arto:containsObject {obj_uri} .")
            ttl_lines.append("")

    def _generate_visual_elements_triples(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], validation_data: Optional[Dict[str, Any]] = None, od_data: Optional[Dict[str, Any]] = None):
        """Generate visual element RDF triples"""

        # 1. Generate color elements
        self._generate_colour_elements(json_data, scene_uri, ttl_lines, validation_data, od_data)
        
        # 2. Generate composition elements
        self._generate_composition_elements(json_data, scene_uri, ttl_lines)
        
        # 3. Generate texture elements
        self._generate_texture_elements(json_data, scene_uri, ttl_lines)

    def _generate_colour_elements(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], validation_data: Optional[Dict[str, Any]] = None, od_data: Optional[Dict[str, Any]] = None):
        """Generate color visual elements"""

        # Build object name → URI map
        obj_name_uri = {}
        enhanced_objects = json_data.get("objects", {}).get("enhanced_objects", [])
        for i, obj_data in enumerate(enhanced_objects):
            obj_name = obj_data.get("name", "").lower()
            if obj_name:
                obj_name_uri[obj_name] = f"{scene_uri}_obj_{self._safe_uri_name(obj_data.get('name', f'object_{i}'))}"

        # Path 1 FIRST: validation step3_color — RGB + VLM-verified name, highest quality
        val_covered: set = set()
        if validation_data:
            try:
                root = validation_data.get("verification_result", {}) or validation_data.get("steps", {})
                details_list = root.get("step3_color", {}).get("details", [])
                for item in details_list:
                    if not isinstance(item, dict):
                        continue
                    obj_label = item.get("object", "").lower()
                    obj_uri = obj_name_uri.get(obj_label, scene_uri)
                    cols = item.get("detailed_colors", [])
                    if not cols:
                        continue
                    val_covered.add(obj_label)
                    total_prop = sum(c.get("proportion", 0) for c in cols if isinstance(c, dict))
                    safe_obj = self._safe_uri_name(item.get("object", "item"))
                    for idx, color_info in enumerate(cols):
                        if not isinstance(color_info, dict):
                            continue
                        rgb = color_info.get("rgb")
                        proportion = color_info.get("proportion", 0)
                        color_name = color_info.get("name", "unknown").lower().replace(" ", "_")
                        if not (rgb and len(rgb) == 3):
                            continue
                        norm_prop = (proportion / total_prop) if total_prop > 0 else proportion
                        if color_name and color_name != "unknown":
                            color_uri = f"{scene_uri}_{safe_obj}_colour_{color_name}_{idx}"
                        else:
                            color_uri = f"{scene_uri}_{safe_obj}_colour_{idx}"
                        label_name = color_name.replace("_", " ").title() if color_name != "unknown" else f"Colour {idx}"
                        ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                        ttl_lines.append(f'    rdfs:label "{label_name}"@en ;')
                        ttl_lines.append(f"    arto:hasColourValue [")
                        ttl_lines.append(f"        a arto:ColourValue ;")
                        ttl_lines.append(f'        arto:colourSystem "RGB" ;')
                        ttl_lines.append(f"        qudt:vector ({rgb[0]} {rgb[1]} {rgb[2]})")
                        ttl_lines.append(f"    ] ;")
                        ttl_lines.append(f'    arto:hasMetric [ a arto:Metric ; qudt:numericValue "{norm_prop:.4f}" ] .')
                        ttl_lines.append("")
                        ttl_lines.append(f"{obj_uri} arto:containsElement {color_uri} .")
                        ttl_lines.append("")
            except Exception as e:
                self.logger.warning(f"Error extracting validation colors: {e}")

        # Path 0 SECOND: image-extracted RGB — only for objects not covered by validation.
        # Each color item is a (rgb, proportion, color_name) triple from run_kg_assembly.py.
        image_covered: set = set()
        if od_data:
            for obj_name_lower, colors in od_data.get("image_colors", {}).items():
                if obj_name_lower in val_covered:
                    continue
                obj_uri = obj_name_uri.get(obj_name_lower)
                if not obj_uri or not colors:
                    continue
                image_covered.add(obj_name_lower)
                for c_idx, color_item in enumerate(colors):
                    rgb, proportion = color_item[0], color_item[1]
                    color_name = color_item[2] if len(color_item) > 2 else "Color"
                    color_uri = f"{obj_uri}_imgcolor_{c_idx}"
                    safe_name = self._safe_uri_name(color_name)
                    ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                    ttl_lines.append(f'    rdfs:label "{self._escape_literal(color_name)}"@en ;')
                    ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color_name)}"@en ;')
                    ttl_lines.append(f"    arto:hasColourValue [")
                    ttl_lines.append(f"        a arto:ColourValue ;")
                    ttl_lines.append(f'        arto:colourSystem "RGB" ;')
                    ttl_lines.append(f"        qudt:vector ({rgb[0]} {rgb[1]} {rgb[2]})")
                    ttl_lines.append(f"    ] ;")
                    ttl_lines.append(f'    arto:hasMetric [ a arto:Metric ; qudt:numericValue "{proportion:.4f}" ] .')
                    ttl_lines.append("")
                    ttl_lines.append(f"{obj_uri} arto:containsElement {color_uri} .")
                    ttl_lines.append("")

        # Path 2: Environment color scheme → colours linked to scene
        color_scheme = json_data.get("environment", {}).get("color_scheme", {})
        main_palette = color_scheme.get("main_palette", {})
        if main_palette.get("primary_colors"):
            for i, color in enumerate(main_palette["primary_colors"]):
                color_uri = f"{scene_uri}_envcolor_{i}_{self._safe_uri_name(color)}"
                ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                ttl_lines.append(f'    rdfs:label "{self._escape_literal(color)}"@en ;')
                ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color)}"@en .')
                ttl_lines.append("")
                ttl_lines.append(f"{scene_uri} arto:containsElement {color_uri} .")
                ttl_lines.append("")

        # Path 3: Text-only primary_colors — skipped for val/image-covered objects
        for obj_idx, obj_data in enumerate(enhanced_objects):
            obj_name = obj_data.get("name", f"object_{obj_idx}")
            if obj_name.lower() in val_covered or obj_name.lower() in image_covered:
                continue
            primary_colors = obj_data.get("primary_colors") or obj_data.get("color_palette", {}).get("primary_colors", [])
            obj_uri = obj_name_uri.get(obj_name.lower(), f"{scene_uri}_obj_{self._safe_uri_name(obj_name)}")
            for color_idx, color in enumerate(primary_colors):
                color_uri = f"{obj_uri}_color_{self._safe_uri_name(color)}"
                ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                ttl_lines.append(f'    rdfs:label "{self._escape_literal(color)}"@en ;')
                ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color)}"@en .')
                ttl_lines.append("")
                ttl_lines.append(f"{obj_uri} arto:containsElement {color_uri} .")
                ttl_lines.append("")

    def _generate_composition_elements(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str]):
        """Generate composition visual elements"""
        
        composition_techniques = json_data.get("composition", {}).get("composition_techniques", {})
        
        if composition_techniques:
            comp_uri = f"{scene_uri}_composition"
            
            ttl_lines.append(f"{comp_uri} rdf:type arto:Composition ;")
            ttl_lines.append(f'    rdfs:label "Main Composition"@en ;')
            
            # Composition rules
            comp_rules = composition_techniques.get("composition_rules", {})
            if comp_rules.get("primary_rule"):
                rule = comp_rules["primary_rule"]
                ttl_lines.append(f'    arto:hasCompositionRule "{self._escape_literal(rule)}"@en ;')
            
            # Perspective info
            viewing_perspective = composition_techniques.get("viewing_perspective", {})
            if viewing_perspective.get("viewpoint"):
                viewpoint = viewing_perspective["viewpoint"]
                ttl_lines.append(f'    arto:hasViewpoint "{self._escape_literal(viewpoint)}"@en ;')
            
            # Depth techniques
            depth_techniques = composition_techniques.get("depth_techniques", {})
            if depth_techniques.get("primary_depth_cues"):
                depth_cues = ", ".join(depth_techniques["primary_depth_cues"])
                ttl_lines.append(f'    arto:hasDepthCues "{self._escape_literal(depth_cues)}"@en ;')
            
            # Remove last semicolon
            if ttl_lines and ttl_lines[-1].endswith(' ;'):
                ttl_lines[-1] = ttl_lines[-1].rstrip(' ;') + ' .'
            
            ttl_lines.append("")
            
            # Connect to scene
            ttl_lines.append(f"{scene_uri} arto:containsElement {comp_uri} .")
            ttl_lines.append("")

    def _generate_texture_elements(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str]):
        """Generate material texture visual elements"""
        
        enhanced_objects = json_data.get("objects", {}).get("enhanced_objects", [])
        
        for obj_idx, obj_data in enumerate(enhanced_objects):
            material_texture = obj_data.get("material_texture", {})
            
            if material_texture.get("surface_texture"):
                texture_uri = f"{scene_uri}_obj{obj_idx}_texture"
                
                ttl_lines.append(f"{texture_uri} rdf:type arto:Texture ;")
                ttl_lines.append(f'    rdfs:label "Object Texture"@en ;')
                
                surface_texture = material_texture["surface_texture"]
                ttl_lines.append(f'    arto:hasTextureDescription "{self._escape_literal(surface_texture)}"@en ;')
                
                if material_texture.get("tactile_quality"):
                    tactile = material_texture["tactile_quality"]
                    ttl_lines.append(f'    arto:hasTactileQuality "{self._escape_literal(tactile)}"@en ;')
                
                ttl_lines.append(".")
                ttl_lines.append("")
                
                # Connect to scene
                ttl_lines.append(f"{scene_uri} arto:containsElement {texture_uri} .")
                ttl_lines.append("")

    def _generate_spatial_connectivity(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], validation_data: Optional[Dict[str, Any]] = None):
        """Generate Spatial Connectivity Triples - handles all JSON formats"""

        # Build object name → URI map from enhanced_objects
        obj_name_uri = {}
        enhanced_objects = json_data.get("objects", {}).get("enhanced_objects", [])
        for i, obj_data in enumerate(enhanced_objects):
            obj_name = obj_data.get("name", "").lower()
            if obj_name:
                obj_uri = f"{scene_uri}_obj_{self._safe_uri_name(obj_data.get('name', f'object_{i}'))}"
                obj_name_uri[obj_name] = obj_uri

        # Build object ID → ALL names from enhanced_objects — same source as obj_name_uri,
        # so id→name→URI is always consistent (object_ids/object_names may use coarser
        # COCO labels while enhanced_objects carries the fine-grained artistic names).
        obj_id_to_all_names: Dict[int, List[str]] = {}
        for obj_data in enhanced_objects:
            oid = obj_data.get("object_id")
            oname = obj_data.get("name", "").lower()
            if oid is not None and oname:
                obj_id_to_all_names.setdefault(oid, []).append(oname)
        # First-name lookup for the common case (different subject/object ids)
        obj_id_to_name = {oid: names[0] for oid, names in obj_id_to_all_names.items()}

        # Relation string → ARTO predicate
        PREDICATE_MAP = {
            "on": "on", "beside": "beside", "behind": "behind", "above": "above",
            "below": "below", "under": "under", "near": "near", "between": "between",
            "inside": "inside", "contains": "contains", "surrounding": "surrounding",
            "left of": "leftOf", "left_of": "leftOf",
            "right of": "rightOf", "right_of": "rightOf",
            "in front of": "inFrontOf", "in_front_of": "inFrontOf",
            "next to": "nextTo", "next_to": "nextTo",
            "part of": "partOf", "part_of": "partOf",
            "watching": "watching", "holding": "holding",
        }

        def get_predicate(rel_str: str) -> str:
            rel_lower = rel_str.lower().strip().replace("_", " ")
            if rel_lower in PREDICATE_MAP:
                return PREDICATE_MAP[rel_lower]
            # Normalize unmapped multi-word predicates to camelCase
            parts = [w for w in rel_lower.split() if w]
            return (parts[0] + "".join(w.capitalize() for w in parts[1:])) if parts else "spatiallyRelatedTo"

        emitted = set()

        def emit_triple(subj_uri: str, predicate: str, obj_uri: str):
            if subj_uri and obj_uri and subj_uri != obj_uri:
                triple = f"{subj_uri} arto:{predicate} {obj_uri} ."
                if triple not in emitted:
                    ttl_lines.append(triple)
                    emitted.add(triple)

        composition = json_data.get("composition", {})

        # --- Path 1: composition.spatial_relations / semantic_relations (from JSON) ---
        # Handles both old int-ID list format [[id, "rel", id], ...] and
        # new name-dict format [{"subject": "name", "relation": "rel", "object": "name"}, ...]
        for rel_key in ("spatial_relations", "semantic_relations"):
            for rel in composition.get(rel_key, []):
                if isinstance(rel, list) and len(rel) == 3:
                    # Old format: [subject_id, "relation", object_id]
                    subj_id, rel_str, obj_id = rel[0], str(rel[1]), rel[2]
                    predicate = get_predicate(rel_str)
                    if subj_id == obj_id:
                        # Duplicate COCO ID: the relation connects distinct objects
                        # sharing the same id (e.g. [20,"part_of",20] → cow partOf herd).
                        # Emit ordered pairs between all unique names for that id.
                        names = list(dict.fromkeys(obj_id_to_all_names.get(subj_id, [])))
                        for i in range(len(names)):
                            for j in range(i + 1, len(names)):
                                emit_triple(obj_name_uri.get(names[i]), predicate,
                                            obj_name_uri.get(names[j]))
                    else:
                        subj_name = obj_id_to_name.get(subj_id)
                        obj_name = obj_id_to_name.get(obj_id)
                        if subj_name and obj_name:
                            emit_triple(
                                obj_name_uri.get(subj_name),
                                predicate,
                                obj_name_uri.get(obj_name),
                            )
                elif isinstance(rel, dict):
                    # New format: {"subject": "name", "relation": "rel", "object": "name"}
                    subj_name = rel.get("subject", "").lower()
                    rel_str = rel.get("relation", "")
                    obj_name = rel.get("object", "").lower()
                    emit_triple(
                        obj_name_uri.get(subj_name),
                        get_predicate(rel_str),
                        obj_name_uri.get(obj_name),
                    )

        # --- Path 2: Old dict-of-dicts format (composition.spatial_relationships.geometric_relations) ---
        if "spatial_relationships" in composition:
            for rel_data in composition["spatial_relationships"].get("geometric_relations", []):
                if isinstance(rel_data, dict) and "subject" in rel_data and "object" in rel_data:
                    emit_triple(
                        obj_name_uri.get(rel_data["subject"].lower()),
                        get_predicate(rel_data.get("relation", "")),
                        obj_name_uri.get(rel_data["object"].lower()),
                    )

        # --- Path 3: Validation data (steps.step6_spatial_geometric.details) ---
        if validation_data:
            try:
                root = validation_data.get("verification_result", {}) or validation_data.get("steps", {})
                val_rels = root.get("step6_spatial_geometric", {}).get("details", [])
                for rel_data in val_rels:
                    if not rel_data.get("verified", False):
                        continue
                    relation_text = rel_data.get("relation", "").lower()
                    if not relation_text:
                        continue
                    # Determine predicate via keyword scan
                    predicate = "spatiallyRelatedTo"
                    for phrase, pred in PREDICATE_MAP.items():
                        padded = f" {relation_text} "
                        if f" {phrase} " in padded or padded.endswith(f" {phrase} "):
                            predicate = pred
                            break
                    # Locate subject and object by scanning known names in relation text
                    found = []
                    for name, uri in obj_name_uri.items():
                        if name in relation_text:
                            found.append((relation_text.find(name), uri))
                    found.sort()
                    if len(found) >= 2:
                        emit_triple(found[0][1], predicate, found[-1][1])
            except Exception:
                pass

        if emitted:
            ttl_lines.append("")

    # Helper methods
    def _build_artwork_description(self, json_data: Dict[str, Any]) -> str:
        """Build artwork comprehensive description"""
        parts = []
        
        style = json_data.get("style")
        if style:
            parts.append(f"An artwork created in {style} style")
        
        objects = json_data.get("objects", {}).get("object_names", [])
        if objects:
            if len(objects) <= 3:
                obj_list = ", ".join(objects)
                parts.append(f"featuring {obj_list}")
            else:
                parts.append(f"featuring {len(objects)} objects including {objects[0]} and {objects[1]}")
        
        env_details = json_data.get("environment", {}).get("environment_details", {})
        scene_brief = env_details.get("scene_brief", "")
        if scene_brief:
            parts.append(f"set in {scene_brief}")
        
        return ". ".join(parts) if parts else ""

    def _build_scene_description(self, json_data: Dict[str, Any]) -> str:
        """Build scene description"""
        parts = []
        
        # Composition info
        composition = json_data.get("composition", {})
        spatial_relationships = composition.get("spatial_relationships", {})
        
        if spatial_relationships.get("depth_arrangement"):
            depth = spatial_relationships["depth_arrangement"]
            fg_count = len(depth.get("foreground", []))
            parts.append(f"Multi-layered scene with {fg_count} foreground elements")
        
        # Environment info
        env_details = json_data.get("environment", {}).get("environment_details", {})
        if env_details.get("time_of_day"):
            parts.append(f"set during {env_details['time_of_day']}")
        
        return "; ".join(parts) if parts else "Main artwork scene"

    def _infer_genre_from_composition(self, json_data: Dict[str, Any]) -> Optional[str]:
        """Infer art genre from composition info"""
        
        # Infer from spatial relations
        spatial = json_data.get("composition", {}).get("spatial_relationships", {})
        if spatial.get("composition_type"):
            comp_type = spatial["composition_type"]
            return self.composition_type_mappings.get(comp_type)
        
        # Infer from object types
        objects = json_data.get("objects", {}).get("object_names", [])
        if "person" in objects:
            return "arto:PortraitGenre"
        elif any(obj in ["tree", "mountain", "sky"] for obj in objects):
            return "arto:LandscapeGenre"
        elif len(objects) > 0 and all(obj in ["vase", "fruit", "book", "flower"] for obj in objects):
            return "arto:StillLifeGenre"
        
        return "arto:GenreSceneGenre"  # Default to Genre Scene

    def _describe_depth_arrangement(self, depth_arrangement: Dict[str, Any]) -> str:
        """Describe depth arrangement"""
        parts = []
        
        if depth_arrangement.get("foreground"):
            fg_count = len(depth_arrangement["foreground"])
            parts.append(f"{fg_count} foreground elements")
        
        if depth_arrangement.get("midground"):
            mg_count = len(depth_arrangement["midground"])
            parts.append(f"{mg_count} midground elements")
        
        if depth_arrangement.get("background"):
            bg_count = len(depth_arrangement["background"])
            parts.append(f"{bg_count} background elements")
        
        return ", ".join(parts) if parts else "layered composition"

    def _describe_scale_hierarchy(self, scale_hierarchy: Dict[str, Any]) -> str:
        """Describe scale hierarchy"""
        parts = []
        
        if scale_hierarchy.get("primary_objects"):
            primary_count = len(scale_hierarchy["primary_objects"])
            parts.append(f"{primary_count} primary objects")
        
        if scale_hierarchy.get("secondary_objects"):
            secondary_count = len(scale_hierarchy["secondary_objects"])
            parts.append(f"{secondary_count} secondary objects")
        
        return ", ".join(parts) if parts else "hierarchical scaling"

    def _safe_uri_name(self, name: str) -> str:
        """Generate safe URI name"""
        if not name:
            return "unnamed"
        
        # Replace spaces and special chars
        safe_name = name.replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric chars (except underscore)
        safe_name = "".join(c for c in safe_name if c.isalnum() or c == "_")
        # Ensure starts with letter
        if safe_name and not safe_name[0].isalpha():
            safe_name = "obj_" + safe_name
        
        return safe_name or "unnamed"

    def _escape_literal(self, text: str) -> str:
        """Escape special chars in TTL literal"""
        if not text:
            return ""
        
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        return text

    def batch_convert_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Batch convert all JSON files in directory to TTL format
        """
        self.logger.info(f"Starting batch ARTO conversion: {input_dir} -> {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        stats = {
            "total_files": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "errors": []
        }
        
        # Find all JSON files
        json_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.json') and not file.startswith('batch_'):
                    json_files.append(os.path.join(root, file))
        
        stats["total_files"] = len(json_files)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                base_name = os.path.splitext(os.path.basename(json_file))[0]
                ttl_file = os.path.join(output_dir, f"{base_name}.ttl")
                
                self.convert_artwork_to_ttl(json_data, output_path=ttl_file)
                
                stats["successful_conversions"] += 1
                
            except Exception as e:
                stats["failed_conversions"] += 1
                error_msg = f"Failed to convert {json_file}: {str(e)}"
                stats["errors"].append(error_msg)
                self.logger.error(error_msg)
        
        # Save conversion report
        report = {
            "conversion_timestamp": datetime.now().isoformat(),
            "input_directory": input_dir,
            "output_directory": output_dir,
            "statistics": stats,
            "success_rate": stats["successful_conversions"] / stats["total_files"] if stats["total_files"] > 0 else 0
        }
        
        report_file = os.path.join(output_dir, "arto_conversion_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"ARTO batch conversion completed: {stats['successful_conversions']}/{stats['total_files']} successful")
        
        return report


def create_arto_mapper() -> ARTOMapper:
    """Create ARTO Mapper instance"""
    return ARTOMapper()