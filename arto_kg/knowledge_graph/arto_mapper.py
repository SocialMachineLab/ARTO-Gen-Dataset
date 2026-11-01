"""
Maps artwork specifications to ARTO (Artwork Object Ontology) Turtle: the
artwork, its scene, objects, visual elements and relations.
"""

import re
import uuid
from typing import Dict, Any, List, Optional
from urllib.parse import quote
from rdflib import Graph
from arto_kg.conceptualization.utils import setup_logger
from arto_kg.conceptualization.object_instance import colours_of, materials_of
from arto_kg.config import colour_vocabulary, material_vocabulary
from arto_kg.config import grounding_vocabulary as gv
from arto_kg.config.model_config import RELATION_CONFIG, STYLE_AAT_CONCEPTS
from arto_kg.validation import instance_assignment as ia
from arto_kg.validation.detections import detection_list

def relation_predicate(relation: Any) -> Optional[str]:
    """The ARTO predicate name for a relation string, or None if it has none.

    The words in camelCase: "left_of" -> "leftOf", "looking at" -> "lookingAt".
    Anything other than letters and digits separates words, so the result is
    always a valid Turtle local name.
    """
    words = re.sub(r"[^a-z0-9]+", " ", str(relation or "").lower()).split()
    if not words:
        return None
    return words[0] + "".join(w.capitalize() for w in words[1:])


def _vocabulary_relations() -> List[str]:
    semantic = RELATION_CONFIG.get("semantic_relations", {})
    names = (list(RELATION_CONFIG.get("spatial_relations", []))
             + list(semantic.get("actions_and_interactions", []))
             + list(semantic.get("pose_and_states", []))
             + ["next to", "between", "left of", "right of"])
    return sorted({relation_predicate(name) for name in names} - {None})


# Every predicate of the Stage 1 relation vocabulary, plus "next to" and
# "between". Each TTL file declares the ones it uses as subproperties of
# arto:relatedTo.
RELATION_PROPERTIES = tuple(_vocabulary_relations())

class ARTOMapper:
    """Map JSON artwork data to ARTO ontology TTL format"""
    
    def __init__(self):
        self.logger = setup_logger("arto_mapper")
        
        # ARTO namespace and prefixes
        self.namespaces = {
            "arto": "http://w3id.org/arto#",
            "dcterms": "http://purl.org/dc/terms/",
            "sdo": "https://schema.org/",
            "edm": "http://www.europeana.eu/schemas/edm/",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "qudt": "http://qudt.org/schema/qudt/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "wd": "http://www.wikidata.org/entity/",
            # Getty AAT, used for style and material concepts.
            "aat": "http://vocab.getty.edu/aat/",
            "": "http://w3id.org/arto/resource/"
        }
        
        # Art style -> Getty AAT concept, used as the `arto:style` value.
        self.style_mappings = {
            style: f"aat:{concept['aat_id']}"
            for style, concept in STYLE_AAT_CONCEPTS.items()
        }

    def convert_artwork_to_ttl(self, json_data: Dict[str, Any], od_data: Optional[Dict[str, Any]] = None, validation_data: Optional[Dict[str, Any]] = None, output_path: Optional[str] = None) -> str:
        """Convert one specification, with optional detection and validation
        results, to ARTO TTL. Writes it to output_path if given and returns it.
        """
        artwork_id = json_data.get('artwork_id', 'unknown')
        self.logger.info(f"Converting artwork {artwork_id} to ARTO TTL")
        
        try:
            ttl_content = self._generate_ttl_content(json_data, od_data, validation_data)

            # The Turtle is assembled as text; it is parsed before anything is
            # written, so a malformed file raises instead of being saved.
            try:
                Graph().parse(data=ttl_content, format="turtle")
            except Exception as exc:
                raise ValueError(f"generated Turtle does not parse: {exc}") from exc

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
        relation_declarations_at = len(ttl_lines) - 2
        self._used_relations = set()

        # Generate artwork entity
        artwork_uri = self._generate_artwork_uri(json_data)
        ttl_lines.extend(self._generate_artwork_triples(json_data, artwork_uri))
        
        # Concepts used by this artwork are collected as they are referenced and
        # declared at the end, so each artwork's file stands on its own.
        self._used_concepts = {}
        self._pending_scheme = None

        # Generate scene entity
        main_scene_uri = self._generate_scene_triples(json_data, artwork_uri, ttl_lines)

        # The colour scheme node, referenced from inside the scene block above.
        self._flush_colour_scheme(ttl_lines)

        # Generate object entities
        self._generate_objects_triples(json_data, main_scene_uri, ttl_lines, od_data, validation_data)

        # Generate visual element entities
        self._generate_visual_elements_triples(json_data, main_scene_uri, ttl_lines, validation_data, od_data)

        # Generate spatial relation/connectivity entities
        self._generate_spatial_connectivity(json_data, main_scene_uri, ttl_lines)

        # Concept declarations last: everything that referenced one has run.
        self._emit_concept_declarations(ttl_lines)

        # The relation predicates used above, declared in the header.
        ttl_lines[relation_declarations_at:relation_declarations_at] = [
            f"arto:{pred} rdfs:subPropertyOf arto:relatedTo ."
            for pred in sorted(self._used_relations)]

        return "\n".join(ttl_lines)

    def _generate_header(self, json_data: Dict[str, Any]) -> List[str]:
        """Generate TTL file header"""
        lines = []
        
        lines.append("# ARTO TTL - Generated from AI Artwork JSON")
        lines.append(f"# Artwork ID: {json_data.get('artwork_id', 'unknown')}")
        lines.append("")
        
        # Namespace declaration
        for prefix, namespace in self.namespaces.items():
            lines.append(f"@prefix {prefix}: <{namespace}> .")
        lines.append("")

        # An object's relative size, one of the size concepts. The relation
        # predicates the file uses are inserted above this line once the
        # relations are written (see _generate_ttl_content).
        lines.append("arto:hasSize rdf:type owl:ObjectProperty ; rdfs:range skos:Concept .")
        lines.append("")

        return lines

    def _generate_artwork_uri(self, json_data: Dict[str, Any]) -> str:
        """Generate artwork URI"""
        artwork_id = json_data.get("artwork_id", f"artwork_{uuid.uuid4().hex[:8]}")
        safe_id = self._safe_uri_name(artwork_id)
        # Artwork ids already start with "artwork_"; do not repeat it.
        return f":{safe_id}" if safe_id.startswith("artwork_") else f":artwork_{safe_id}"

    def _generate_artwork_triples(self, json_data: Dict[str, Any], artwork_uri: str) -> List[str]:
        """Generate artwork RDF triples"""
        triples = []
        
        # Basic type declaration
        triples.append(f"{artwork_uri} rdf:type arto:Artwork ;")
        
        # Title
        artwork_id = json_data.get("artwork_id", "Untitled Artwork")
        title = artwork_id.replace("_", " ").title()
        triples.append(f'    dcterms:title "{self._escape_literal(title)}"@en ;')
        
        # Description - Prioritize Main Prompt as dcterms:description
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
                 triples.append(f'    dcterms:description """{safe_desc}"""@en ;')
            else:
                 triples.append(f'    dcterms:description "{safe_desc}"@en ;')
        
        # Art style
        style = json_data.get("style")
        if style:
            style_uri = self.style_mappings.get(style)
            if style_uri:
                triples.append(f"    arto:style {style_uri} ;")
            else:
                # An unmapped style gets no invented URI; the triple is omitted.
                self.logger.warning(
                    f"No Getty AAT concept for style '{style}'; omitting arto:style. "
                    f"Known styles: {sorted(self.style_mappings)}"
                )
        
        # Creation info
        if json_data.get("generation_timestamp"):
            triples.append(f'    dcterms:created "{json_data["generation_timestamp"]}"^^xsd:dateTime ;')
        
        # Creator info
        triples.append('    dcterms:creator "AI Artwork Generation System" ;')

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
            ttl_lines.append(f'    dcterms:description "{self._escape_literal(scene_description)}"@en ;')
        
        # Environment dimensions. Time, weather, place and period are concept
        # IRIs where the grounding vocabulary has one.
        environment = json_data.get("environment", {}).get("environment_details", {})
        if environment:
            # Lighting conditions
            lighting = environment.get("lighting", {})
            if lighting:
                lighting_desc = f"{lighting.get('quality', 'natural')} {lighting.get('type', 'light')}"
                ttl_lines.append(f'    arto:hasLighting "{lighting_desc}"@en ;')

            # Time and weather: concept if one is registered, literal if not.
            self._emit_scene_concept(ttl_lines, "arto:hasTimeOfDay", "time",
                                     environment.get("time_of_day"))
            weather = environment.get("weather")
            if weather and weather != "not applicable":
                self._emit_scene_concept(ttl_lines, "arto:hasWeather", "weather", weather)

            # Place, from the chosen Places365 category.
            self._emit_scene_concept(
                ttl_lines, "arto:hasPlace", "place",
                gv.place_label(environment.get("place365_name")),
                raw=environment.get("place365_name"))

            # Historical period.
            self._emit_scene_concept(ttl_lines, "arto:hasPeriod", "period",
                                     environment.get("period"))

            # Colour scheme node, carrying harmony, contrast and temperature;
            # its colours are emitted separately as scene colours.
            self._emit_colour_scheme(ttl_lines, main_scene_uri, json_data)

        # Remove last semicolon
        if ttl_lines and ttl_lines[-1].endswith(' ;'):
            ttl_lines[-1] = ttl_lines[-1].rstrip(' ;') + ' .'
        
        ttl_lines.append("")
        
        # Connect artwork to scene
        ttl_lines.append(f"{artwork_uri} arto:containsScene {main_scene_uri} .")
        ttl_lines.append("")
        
        return main_scene_uri

    def _emit_scene_concept(self, ttl_lines: List[str], predicate: str,
                            family: str, label: Any, raw: Any = None) -> None:
        """One scene dimension, as a concept IRI where the registry has one.

        A value without a concept is written as a literal.
        """
        if not label:
            return
        concept = gv.lookup(family, label)
        if concept:
            self._register_concept(concept)
            ttl_lines.append(f"    {predicate} {self._concept_curie(concept)} ;")
        else:
            text = str(raw if raw is not None else label)
            ttl_lines.append(f'    {predicate} "{self._escape_literal(text)}"@en ;')

    def _emit_colour_scheme(self, ttl_lines: List[str], scene_uri: str,
                            json_data: Dict[str, Any]) -> None:
        """Reference the scene's colour scheme node; it is written by _flush_colour_scheme."""
        scheme = self._find_colour_scheme(json_data)
        if not scheme:
            return
        scheme_uri = f"{scene_uri}_colourscheme"
        ttl_lines.append(f"    arto:hasColourScheme {scheme_uri} ;")
        self._pending_scheme = (scheme_uri, scheme)

    def _flush_colour_scheme(self, ttl_lines: List[str]) -> None:
        """Write the colour scheme node itself, after the scene block closes."""
        pending = getattr(self, "_pending_scheme", None)
        if not pending:
            return
        scheme_uri, scheme = pending
        self._pending_scheme = None

        ttl_lines.append(f"{scheme_uri} rdf:type arto:ColourScheme ;")
        ttl_lines.append('    rdfs:label "Colour Scheme"@en ;')
        for key, predicate in (("harmony_type", "arto:hasColourHarmony"),
                               ("contrast_level", "arto:hasContrastLevel"),
                               ("colour_temperature", "arto:hasColourTemperature")):
            value = scheme.get(key)
            if value:
                ttl_lines.append(
                    f'    {predicate} "{self._escape_literal(str(value))}"@en ;')
        if ttl_lines[-1].endswith(' ;'):
            ttl_lines[-1] = ttl_lines[-1].rstrip(' ;') + ' .'
        ttl_lines.append("")

    def _register_concept(self, concept: Dict[str, Any]) -> None:
        """Remember a concept so its declaration is emitted once per artwork."""
        if not hasattr(self, "_used_concepts"):
            self._used_concepts = {}
        self._used_concepts[concept["arto_uri"]] = concept

    @staticmethod
    def _concept_curie(concept: Dict[str, Any]) -> str:
        """`:Baroque` for a resource URI, an angle-bracket IRI otherwise.

        The empty prefix is bound to the resource namespace in the header, so a
        resource concept is written `:Baroque`.
        """
        uri = concept["arto_uri"]
        prefix = "http://w3id.org/arto/resource/"
        if uri.startswith(prefix):
            return ":" + uri[len(prefix):]
        return f"<{uri}>"

    def _emit_concept_declarations(self, ttl_lines: List[str]) -> None:
        """Declare each concept the artwork used, with its external links.

        Repeated in every artwork so each TTL file is self-contained.
        """
        used = getattr(self, "_used_concepts", None)
        if not used:
            return
        for uri in sorted(used):
            concept = used[uri]
            curie = self._concept_curie(concept)
            ttl_lines.append(f"{curie} rdf:type skos:Concept ;")
            ttl_lines.append(
                f'    rdfs:label "{self._escape_literal(concept["label"])}"@en ;')
            if concept.get("notation"):
                ttl_lines.append(f'    skos:notation "{concept["notation"]}" ;')
            links = gv.external_links(concept)
            for index, link in enumerate(links):
                relation = link.get("relation") or "skos:closeMatch"
                terminator = " ;" if index < len(links) - 1 else " ."
                ttl_lines.append(f"    {relation} <{link['uri']}>{terminator}")
            if not links:
                ttl_lines[-1] = ttl_lines[-1].rstrip(' ;') + ' .'
            ttl_lines.append("")
        self._used_concepts = {}

    def _object_index(self, json_data: Dict[str, Any],
                      scene_uri: str) -> List[Dict[str, Any]]:
        """Every object with its URI, keyed by instance_id.

        URIs are built from instance_id so same-named objects stay distinct.
        Specs without an `instances` block are numbered positionally.
        """
        index = []
        for instance_id, obj in ia.iter_objects(json_data):
            safe = self._safe_uri_name(instance_id)
            # Instance ids already start with "obj_"; do not repeat it.
            index.append({
                "instance_id": instance_id,
                "uri": f"{scene_uri}_{safe}" if safe.startswith("obj_") else f"{scene_uri}_obj_{safe}",
                "name": str(obj.get("name") or instance_id),
                "coco_category": str(obj.get("coco_category") or ""),
                "data": obj,
            })
        return index

    @staticmethod
    def _uri_lookup(index: List[Dict[str, Any]]) -> Dict[str, str]:
        """instance_id -> URI, plus lowercase name -> URI for name-keyed inputs.

        A name shared by two instances resolves to the first.
        """
        lookup: Dict[str, str] = {}
        for record in index:
            lookup[record["instance_id"]] = record["uri"]
        for record in index:
            key = record["name"].lower()
            if key and key not in lookup:
                lookup[key] = record["uri"]
        return lookup

    @staticmethod
    def _measured_bboxes(validation_data: Optional[Dict[str, Any]]) -> Dict[str, List[float]]:
        """{instance_id: bbox} from Stage 3's measurements block."""
        if not isinstance(validation_data, dict):
            return {}
        measurements = validation_data.get("measurements")
        if not isinstance(measurements, dict):
            return {}
        boxes = {}
        for entry in measurements.get("objects") or []:
            if isinstance(entry, dict) and entry.get("instance_id") and entry.get("bbox"):
                boxes[entry["instance_id"]] = entry["bbox"]
        return boxes

    def _generate_objects_triples(self, json_data: Dict[str, Any], scene_uri: str,
                                  ttl_lines: List[str],
                                  od_data: Optional[Dict[str, Any]] = None,
                                  validation_data: Optional[Dict[str, Any]] = None):
        """Generate object RDF triples"""
        index = self._object_index(json_data, scene_uri)
        measured = self._measured_bboxes(validation_data)
        measurements = (validation_data or {}).get("measurements") if isinstance(validation_data, dict) else None
        has_measurements = isinstance(measurements, dict) and isinstance(measurements.get("objects"), list)

        # Name-keyed fallback bboxes, only for artworks with no measurements
        # block: an instance the measurements leave without a box gets none.
        # The first detection per label wins; any layout Stage 3 writes is read
        # (flat, or nested in a fused result).
        od_map = {}
        for obj in ([] if has_measurements else detection_list(od_data or {})):
            if not isinstance(obj, dict):
                continue
            label = (obj.get("mapped_gt") or obj.get("label") or obj.get("class_name") or "").lower()
            box = obj.get("box") or obj.get("bbox")
            if label and box and label not in od_map:
                od_map[label] = box

        for record in index:
            obj_data = record["data"]
            obj_name = record["name"]
            obj_uri = record["uri"]

            # Where the registry knows the object, also type it as that concept
            # and as owl:NamedIndividual. Looked up by COCO category first,
            # since the variety name changes with the style.
            concept = (gv.lookup("object", record["coco_category"])
                       or gv.lookup("object", obj_name))
            if concept:
                self._register_concept(concept)
                ttl_lines.append(
                    f"{obj_uri} rdf:type arto:Object , owl:NamedIndividual , "
                    f"{self._concept_curie(concept)} ;")
            else:
                ttl_lines.append(f"{obj_uri} rdf:type arto:Object ;")
            ttl_lines.append(f'    rdfs:label "{self._escape_literal(obj_name.title())}"@en ;')
            # COCO category as its own property; the label is the style variety.
            if record["coco_category"]:
                ttl_lines.append(
                    f'    arto:cocoCategory "{self._escape_literal(record["coco_category"])}" ;')

            # Relative size within the composition, as a size concept.
            size = gv.lookup("size", obj_data.get("size"))
            if size:
                self._register_concept(size)
                ttl_lines.append(f"    arto:hasSize {self._concept_curie(size)} ;")

            # Artistic description
            if obj_data.get("artistic_description"):
                desc = obj_data["artistic_description"]
                ttl_lines.append(f'    dcterms:description "{self._escape_literal(desc)}"@en ;')
            
            # Materials: a vocabulary label is linked to its Getty AAT concept;
            # an unknown material keeps its label only.
            materials = materials_of(obj_data)
            for material in materials:
                aat_uri = material_vocabulary.aat_uri_of(material)
                label = material_vocabulary.canonical(material) or material
                if aat_uri:
                    ttl_lines.append(
                        f'    arto:material [ a arto:MaterialType ; '
                        f'rdfs:label "{self._escape_literal(label)}"@en ; '
                        f'skos:exactMatch {aat_uri} ] ;')
                else:
                    ttl_lines.append(
                        f'    arto:material [ a arto:MaterialType ; '
                        f'rdfs:label "{self._escape_literal(str(material))}"@en ] ;')

            # Physical state, as a flat string or a nested dict
            state = obj_data.get("state") or obj_data.get("physical_condition", {}).get("overall_state")
            if state:
                ttl_lines.append(f'    arto:state [ a skos:Concept ; rdfs:label "{self._escape_literal(str(state))}"@en ] ;')
            
            # Symbolic meaning
            symbolic_meaning = obj_data.get("symbolic_meaning", {})
            if symbolic_meaning.get("symbolic_interpretation"):
                symbol = symbolic_meaning["symbolic_interpretation"]
                ttl_lines.append(f'    arto:hasSymbolicMeaning "{self._escape_literal(symbol)}"@en ;')
            
            # Coordinates: Stage 3's measurement for this instance, else the
            # name-keyed detection.
            bbox = measured.get(record["instance_id"]) or od_map.get(obj_name.lower())
            if bbox:
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

        self._generate_colour_elements(json_data, scene_uri, ttl_lines, validation_data, od_data)

    def _generate_colour_elements(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str], validation_data: Optional[Dict[str, Any]] = None, od_data: Optional[Dict[str, Any]] = None):
        """Generate colour nodes: measured, scene-scheme and declared colours."""

        # Same object index as the object triples, so colours attach to the
        # same object URIs.
        index = self._object_index(json_data, scene_uri)
        obj_name_uri = self._uri_lookup(index)
        enhanced_objects = [record["data"] for record in index]

        # Measured colours from validation step3_color (RGB plus verified name).
        val_covered: set = set()
        if validation_data:
            try:
                root = validation_data.get("verification_result", {}) or validation_data.get("steps", {})
                details_list = self._colour_details(root.get("step3_color", {}))
                for item in details_list:
                    if not isinstance(item, dict):
                        continue
                    # Prefer instance_id, falling back to the object name.
                    obj_label = (item.get("instance_id")
                                 or item.get("object", "").lower())
                    obj_uri = obj_name_uri.get(obj_label, scene_uri)
                    cols = item.get("detailed_colors", [])
                    if not cols:
                        continue
                    val_covered.add(obj_label)
                    total_prop = sum(c.get("proportion", 0) for c in cols if isinstance(c, dict))
                    # Colour URIs use the instance id where present, so
                    # same-named objects get distinct colour nodes.
                    safe_obj = self._safe_uri_name(
                        item.get("instance_id") or item.get("object", "item"))
                    for idx, color_info in enumerate(cols):
                        if not isinstance(color_info, dict):
                            continue
                        rgb = color_info.get("rgb")
                        proportion = color_info.get("proportion", 0)
                        # Sanitised like every other URI; names such as
                        # "Payne's Gray" would otherwise break the Turtle.
                        raw_colour_name = color_info.get("name", "unknown")
                        color_name = self._safe_uri_name(
                            raw_colour_name.lower().replace(" ", "_"))
                        if not (rgb and len(rgb) == 3):
                            continue
                        norm_prop = (proportion / total_prop) if total_prop > 0 else proportion
                        if color_name and color_name != "unknown":
                            color_uri = f"{scene_uri}_{safe_obj}_colour_{color_name}_{idx}"
                        else:
                            color_uri = f"{scene_uri}_{safe_obj}_colour_{idx}"
                        # The label uses the declared name (escaped), not the
                        # sanitised URI part.
                        label_name = (raw_colour_name.title()
                                      if color_name != "unknown"
                                      else f"Colour {idx}")
                        ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
                        ttl_lines.append(
                            f'    rdfs:label "{self._escape_literal(label_name)}"@en ;')
                        ttl_lines.append(f"    arto:hasColourValue [")
                        ttl_lines.append(f"        a arto:ColourValue ;")
                        ttl_lines.append(f'        arto:colourSystem "RGB" ;')
                        ttl_lines.append(f"        qudt:vector ({rgb[0]} {rgb[1]} {rgb[2]})")
                        ttl_lines.append(f"    ] ;")
                        ttl_lines.append(f'    arto:hasMetric [ a arto:Metric ; qudt:numericValue "{norm_prop:.4f}"^^xsd:decimal ] .')
                        ttl_lines.append("")
                        ttl_lines.append(f"{obj_uri} arto:containsElement {color_uri} .")
                        ttl_lines.append("")
            except Exception as e:
                self.logger.warning(f"Error extracting validation colors: {e}")

        image_covered: set = set()

        # Scene colours from the environment colour scheme.
        color_scheme = self._find_colour_scheme(json_data)
        scheme_colours = self._scheme_colour_names(color_scheme)
        if scheme_colours:
            for i, color in enumerate(scheme_colours):
                color_uri = f"{scene_uri}_envcolor_{i}_{self._safe_uri_name(color)}"
                self._emit_declared_colour(color_uri, color, scene_uri, ttl_lines)

        # Declared object colours, skipped for objects that already have
        # measured colours.
        for record in index:
            obj_data = record["data"]
            obj_name = record["name"]
            if (record["instance_id"] in val_covered
                    or obj_name.lower() in val_covered
                    or obj_name.lower() in image_covered):
                continue
            declared = colours_of(obj_data) or obj_data.get("color_palette", {}).get("primary_colors", [])
            obj_uri = record["uri"]
            # "Light Blue" and "light-blue" are one colour: the URI is built
            # from the lowercased name, and each colour is written once.
            written = set()
            for color in declared:
                color_uri = f"{obj_uri}_color_{self._safe_uri_name(str(color).lower())}"
                if color_uri in written:
                    continue
                written.add(color_uri)
                self._emit_declared_colour(color_uri, color, obj_uri, ttl_lines)

    @staticmethod
    def _colour_details(colour_step: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Per-object measured colours from Stage 3, in either result shape.

        Accepts a `details` list of `{object, detailed_colors}` or an
        `object_color_analysis` dict keyed by instance_id, normalised to the
        former with `instance_id` kept.
        """
        if not isinstance(colour_step, dict):
            return []

        details = colour_step.get("details")
        if isinstance(details, list) and details:
            return [item for item in details if isinstance(item, dict)]

        analysis = colour_step.get("object_color_analysis")
        if not isinstance(analysis, dict):
            return []

        rebuilt: List[Dict[str, Any]] = []
        for instance_id, entry in analysis.items():
            if not isinstance(entry, dict):
                continue
            colours = [
                {
                    "rgb": c.get("rgb"),
                    "proportion": c.get("proportion", 0),
                    "name": c.get("color_name") or c.get("name", "unknown"),
                }
                for c in entry.get("dominant_colors") or []
                if isinstance(c, dict) and c.get("rgb")
            ]
            if colours:
                rebuilt.append({
                    "object": entry.get("name") or instance_id,
                    "instance_id": instance_id,
                    "detailed_colors": colours,
                })
        return rebuilt

    @staticmethod
    def _find_colour_scheme(json_data: Dict[str, Any]) -> Dict[str, Any]:
        """The colour scheme, under either spelling and either nesting depth."""
        environment = json_data.get("environment")
        if not isinstance(environment, dict):
            return {}
        containers = [environment]
        details = environment.get("environment_details")
        if isinstance(details, dict):
            containers.insert(0, details)
        for container in containers:
            for key in ("colour_scheme", "color_scheme"):
                scheme = container.get(key)
                if isinstance(scheme, dict) and scheme:
                    return scheme
        return {}

    @staticmethod
    def _scheme_colour_names(scheme: Dict[str, Any]) -> List[str]:
        """Every colour the scheme names, in either scheme shape."""
        if not isinstance(scheme, dict):
            return []

        names: List[str] = []
        main = scheme.get("main_colours")
        if isinstance(main, list) and main:
            names.extend(str(c).strip() for c in main if c)
            if scheme.get("background_colour"):
                names.append(str(scheme["background_colour"]).strip())
            shift = scheme.get("lighting_shift")
            if isinstance(shift, dict):
                names.extend(str(shift[key]).strip()
                             for key in ("light", "shadow") if shift.get(key))
        return names

    def _emit_declared_colour(self, color_uri: str, color: str,
                              parent_uri: str, ttl_lines: List[str]) -> None:
        """A colour declared by the spec, with its vocabulary RGB when known.

        A name outside the colour vocabulary gets the term only, with no RGB.
        """
        rgb = colour_vocabulary.rgb_of(color)
        ttl_lines.append(f"{color_uri} rdf:type arto:Colour ;")
        ttl_lines.append(f'    rdfs:label "{self._escape_literal(color)}"@en ;')
        if rgb:
            ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color)}"@en ;')
            ttl_lines.append("    arto:hasColourValue [")
            ttl_lines.append("        a arto:ColourValue ;")
            ttl_lines.append('        arto:colourSystem "RGB" ;')
            ttl_lines.append(f"        qudt:vector ({rgb[0]} {rgb[1]} {rgb[2]})")
            ttl_lines.append("    ] .")
        else:
            ttl_lines.append(f'    arto:hasColourTerm "{self._escape_literal(color)}"@en .')
        ttl_lines.append("")
        ttl_lines.append(f"{parent_uri} arto:containsElement {color_uri} .")
        ttl_lines.append("")

    def _generate_spatial_connectivity(self, json_data: Dict[str, Any], scene_uri: str, ttl_lines: List[str]):
        """Emit relation triples from the specification's `composition`.

        Relations are structured data and are never re-derived from text.
        """

        # The lookup accepts an instance_id or a lowercase name, so relations
        # in either form resolve to the emitted object URIs.
        index = self._object_index(json_data, scene_uri)
        obj_name_uri = self._uri_lookup(index)
        enhanced_objects = [record["data"] for record in index]

        # Object id -> all names, from the same objects as obj_name_uri so that
        # id -> name -> URI stays consistent.
        obj_id_to_all_names: Dict[int, List[str]] = {}
        for obj_data in enhanced_objects:
            oid = obj_data.get("object_id")
            oname = obj_data.get("name", "").lower()
            if oid is not None and oname:
                obj_id_to_all_names.setdefault(oid, []).append(oname)
        # First-name lookup for the common case (different subject/object ids)
        obj_id_to_name = {oid: names[0] for oid, names in obj_id_to_all_names.items()}

        emitted = set()

        def emit_triple(subj_uri: str, predicate: Optional[str], obj_uri: str):
            # A relation with no usable name carries no predicate and is not written.
            if predicate and subj_uri and obj_uri and subj_uri != obj_uri:
                triple = f"{subj_uri} arto:{predicate} {obj_uri} ."
                if triple not in emitted:
                    ttl_lines.append(triple)
                    emitted.add(triple)
                    self._used_relations.add(predicate)

        composition = json_data.get("composition", {})

        # Relations are either [id, "rel", id] lists or
        # {"subject", "relation", "object"} dicts.
        for rel_key in ("spatial_relations", "semantic_relations"):
            for rel in composition.get(rel_key, []):
                if isinstance(rel, list) and len(rel) == 3:
                    # List form: [subject_id, "relation", object_id]
                    subj_id, rel_str, obj_id = rel[0], str(rel[1]), rel[2]
                    predicate = relation_predicate(rel_str)
                    if subj_id == obj_id:
                        # Same id on both sides links distinct objects sharing it
                        # (e.g. cow partOf herd): emit ordered pairs of their names.
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
                    # Dict form: {"subject", "relation", "object"}
                    subj_name = rel.get("subject", "").lower()
                    rel_str = rel.get("relation", "")
                    obj_name = rel.get("object", "").lower()
                    emit_triple(
                        obj_name_uri.get(subj_name),
                        relation_predicate(rel_str),
                        obj_name_uri.get(obj_name),
                    )

        if emitted:
            ttl_lines.append("")

    # Helper methods
    def _build_artwork_description(self, json_data: Dict[str, Any]) -> str:
        """Fallback artwork description from style, objects and scene brief."""
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

        # Environment info
        env_details = json_data.get("environment", {}).get("environment_details", {})
        if env_details.get("time_of_day"):
            parts.append(f"set during {env_details['time_of_day']}")
        
        return "; ".join(parts) if parts else "Main artwork scene"

    def _safe_uri_name(self, name: str) -> str:
        """Sanitise a name for use as a URI local part."""
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
        """Escape special characters for a TTL string literal."""
        if not text:
            return ""
        
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        return text


def create_arto_mapper() -> ARTOMapper:
    """Create ARTO Mapper instance"""
    return ARTOMapper()