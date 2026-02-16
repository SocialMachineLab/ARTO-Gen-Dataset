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
            "": "http://w3id.org/arto#"
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
        self._generate_visual_elements_triples(json_data, main_scene_uri, ttl_lines, validation_data)
        
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
        
        # Spatial relationship info (Refactored to check flat composition first, fallback to nested)
        spatial_relationships = composition 
        # spatial_relationships = composition.get("spatial_relationships", {})
        
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
            
            # Material info
            material_texture = obj_data.get("material_texture", {})
            if material_texture.get("primary_material"):
                material = material_texture["primary_material"]
                ttl_lines.append(f'    arto:hasMaterial "{self._escape_literal(material)}"@en ;')
            
            # Physical state
            physical_condition = obj_data.get("physical_condition", {})
            if physical_condition.get("overall_state"):
                state = physical_condition["overall_state"]
                ttl_lines.append(f'    arto:hasCondition "{self._escape_literal(state)}"@en ;')
            
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

    def _generate_visual_elements_triples(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], validation_data: Optional[Dict[str, Any]] = None):
        """Generate visual element RDF triples"""
        
        # 1. Generate color elements
        self._generate_colour_elements(json_data, scene_uri, ttl_lines, validation_data)
        
        # 2. Generate composition elements
        self._generate_composition_elements(json_data, scene_uri, ttl_lines)
        
        # 3. Generate texture elements
        self._generate_texture_elements(json_data, scene_uri, ttl_lines)

    def _generate_colour_elements(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], validation_data: Optional[Dict[str, Any]] = None):
        """Generate color visual elements"""
        
        # Get detailed colors from validation data
        detailed_colors = []
        if validation_data:
            try:
                # Structure: steps -> step3_color -> details
                # Handle both 'verification_result' (old format?) and 'steps' (new format)
                root = validation_data.get("verification_result", {})
                if not root:
                    root = validation_data.get("steps", {})
                
                details_list = root.get("step3_color", {}).get("details", [])
                
                if isinstance(details_list, list):
                    for item in details_list:
                        if isinstance(item, dict):
                             cols = item.get("detailed_colors", [])
                             if cols:
                                 detailed_colors.extend(cols)
            except Exception as e:
                self.logger.warning(f"Error extracting validation colors: {e}")

        # Extract from environment color scheme
        color_scheme = json_data.get("environment", {}).get("color_scheme", {})
        main_palette = color_scheme.get("main_palette", {})
        
        if main_palette.get("primary_colors"):
            for i, color in enumerate(main_palette["primary_colors"]):
                color_uri = f"{scene_uri}_color_{i}_{self._safe_uri_name(color)}"
                
                ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                ttl_lines.append(f'    rdfs:label "{self._escape_literal(color)}"@en ;')
                ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color)}"@en ;')
                ttl_lines.append(f'    arto:isAttributeOf {scene_uri} .')
                ttl_lines.append("")
                
                # Connect to scene
                ttl_lines.append(f"{scene_uri} arto:containsElement {color_uri} .")
                ttl_lines.append("")
        
        # If validation data exists, add verified precise colors
        if detailed_colors:
             for idx, color_info in enumerate(detailed_colors):
                rgb = color_info.get("rgb")
                proportion = color_info.get("proportion", 0)
                color_name = color_info.get("name", "unknown").lower().replace(" ", "_")
                
                if rgb and len(rgb) == 3:
                     # Prefer named URI if available, fallback to index if name is missing/generic
                     if color_name and color_name != "unknown":
                         color_uri = f"{scene_uri}_colour_{color_name}_{idx}" # Include idx to allow duplicates (e.g. 2 salmon areas)
                     else:
                         color_uri = f"{scene_uri}_colour_{idx}"
                     
                     ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                     # Use specific name in label if available
                     label_name = color_name.replace("_", " ").title() if color_name != "unknown" else f"Colour {idx}"
                     ttl_lines.append(f'    rdfs:label "{label_name}"@en ;')
                     
                     ttl_lines.append(f"    arto:hasColourValue [")
                     ttl_lines.append(f"        a arto:ColourValue ;")
                     ttl_lines.append(f'        arto:colourSystem "RGB" ;')
                     ttl_lines.append(f"        qudt:vector ({rgb[0]} {rgb[1]} {rgb[2]})")
                     ttl_lines.append(f"    ] ;")
                     ttl_lines.append(f'    arto:hasMetric [ a arto:Size ; qudt:numericValue "{proportion:.2f}" ] .') 
                     
                     ttl_lines.append("")
                     ttl_lines.append(f"{scene_uri} arto:containsElement {color_uri} .")
                     ttl_lines.append("")
        
        # Extract from object color info
        enhanced_objects = json_data.get("objects", {}).get("enhanced_objects", [])
        for obj_idx, obj_data in enumerate(enhanced_objects):
            color_palette = obj_data.get("color_palette", {})
            if color_palette.get("primary_colors"):
                for color_idx, color in enumerate(color_palette["primary_colors"]):
                    color_uri = f"{scene_uri}_obj{obj_idx}_color_{self._safe_uri_name(color)}"
                    
                    ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                    ttl_lines.append(f'    rdfs:label "{self._escape_literal(color)}"@en ;')
                    ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color)}"@en ;')
                    ttl_lines.append(f'    arto:isAttributeOf {scene_uri}_obj_{self._safe_uri_name(obj_data.get("name", "object"))} .')
                    ttl_lines.append("")
                    
                    # Connect to scene
                    ttl_lines.append(f"{scene_uri} arto:containsElement {color_uri} .")
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
        """Generate Spatial Connectivity Triple - Direct Predicates"""
        
        geometric_relations = []
        
        # 1. Try to get from Original JSON (Step 6 equivalent?)
        composition = json_data.get("composition", {})
        
        # Check for geometric relations (spatial_relations in flat structure)
        rels = composition.get("spatial_relations", [])
        if not rels:
             # Fallback to nested check or old key
             if "spatial_relationships" in composition:
                 rels = composition["spatial_relationships"].get("geometric_relations", [])

        if rels:
            geometric_relations.extend(rels)
                
        # 2. Try to get from Validation Data (steps -> step6_spatial_geometric -> details)
        if validation_data:
            try:
                # Handle both 'verification_result' and 'steps'
                root = validation_data.get("verification_result", {})
                if not root:
                    root = validation_data.get("steps", {})
                    
                val_rels = root.get("step6_spatial_geometric", {}).get("details", [])
                if val_rels:
                    geometric_relations.extend(val_rels)
            except:
                pass
        
        if not geometric_relations:
            return

        # Build Object Name -> URI mapping
        obj_map = {}
        enhanced_objects = json_data.get("objects", {}).get("enhanced_objects", [])
        for i, obj_data in enumerate(enhanced_objects):
            obj_name = obj_data.get("name", "").lower()
            if obj_name:
                # Use naming logic consistent with _generate_objects_triples
                obj_uri = f"{scene_uri}_obj_{self._safe_uri_name(obj_data.get('name', f'object_{i}'))}"
                obj_map[obj_name] = obj_uri

        for rel_data in geometric_relations:
            relation_text = rel_data.get("relation", "").lower()
            if not relation_text:
                continue
                
            # Parse relation text "subject predicate object"
            # Here relation_text is usually "cat next to dog", "tree behind house", etc.
            
            # Simple keyword matching for predicate
            predicate = "spatiallyRelatedTo"
            if "next to" in relation_text:
                predicate = "nextTo"
            elif "left of" in relation_text:
                predicate = "leftOf"
            elif "right of" in relation_text:
                predicate = "rightOf"
            elif "above" in relation_text:
                predicate = "above"
            elif "below" in relation_text:
                predicate = "below"
            elif "front of" in relation_text:
                predicate = "inFrontOf"
            elif "behind" in relation_text:
                predicate = "behind"
            elif " on " in f" {relation_text} ": # Avoid "on" in "front"
                predicate = "on"
            elif "under" in relation_text:
                predicate = "under"
                
            # Identify subject and object from the text
            # This is tricky without strict parsing, but we can try to find known object names in the string
            subj_uri = None
            obj_uri = None
            
            # Simple strategy: Iterate all known objects, check if they appear at start/end of text
            # Note: Prone to errors (e.g. "cat" matches "caterpillar"), but may suffice in limited domain
            
            found_objects = []
            for name, uri in obj_map.items():
                if name in relation_text:
                    found_objects.append((name, uri, relation_text.find(name)))
            
            # Sort by position
            found_objects.sort(key=lambda x: x[2])
            
            if len(found_objects) >= 2:
                # Assume first is subject, last is object (covers "A is next to B")
                # Ignore intermediate noise words
                subj_uri = found_objects[0][1]
                obj_uri = found_objects[-1][1]
            
            if subj_uri and obj_uri:
                # Generate Direct Triple
                ttl_lines.append(f"{subj_uri} arto:{predicate} {obj_uri} .")
                
                # If confidence score exists, we can't add directly to this triple (lack RDF-star support)
                # User requested "relation is just predicate", so ignore reification
                
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
        spatial_relationships = composition
        # spatial_relationships = composition.get("spatial_relationships", {})
        
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
        spatial = json_data.get("composition", {})
        # spatial = json_data.get("composition", {}).get("spatial_relationships", {})
        
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
                
                self.convert_artwork_to_ttl(json_data, ttl_file)
                
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
    """Create ARTO mapper instance"""
    return ARTOMapper()