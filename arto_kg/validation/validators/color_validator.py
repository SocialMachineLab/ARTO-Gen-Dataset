
import cv2
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from .. import colour_difference as cd
from .. import colour_reference as cr
from .. import instance_assignment as ia
from ...conceptualization.object_instance import colours_of


class ColorValidator:
    """Object-level and whole-image colour validation using CIEDE2000."""

    def __init__(self, color_ontology_path: Optional[str] = None):
        """
        Colour names resolve through, in order (see lookup_with_provenance):

        1. Hex code
        2. color_vocabulary.csv - the vocabulary RGB for a colour name
        3. gt_color_mapping.json - free colour phrases
        4. Optional colour ontology
        5. Default mapping, exact key, then substring

        Args:
            color_ontology_path: Optional colour ontology CSV
        """
        self.gt_color_mapping = {}
        self._load_gt_color_mapping()

        self.color_vocabulary = None
        # Cluster count chosen by the last _cluster_by_best_k call.
        self.last_chosen_k = None
        self._load_color_vocabulary()

        # Optional colour ontology
        self.color_ontology = None
        if color_ontology_path and os.path.exists(color_ontology_path):
            self.load_color_ontology(color_ontology_path)
        
        # Default colour mapping (last resolution step)
        self.default_color_mapping = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'orange': (255, 165, 0),
            'purple': (128, 0, 128),
            'pink': (255, 192, 203),
            'brown': (139, 69, 19),
            'black': (0, 0, 0),
            'white': (255, 255, 255),
            'gray': (128, 128, 128),
            'grey': (128, 128, 128),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            # Metallic colors
            'silver': (192, 192, 192),
            'gold': (255, 215, 0),
            'metallic': (169, 169, 169),
            # Other common colors
            'amber': (255, 191, 0),
            'beige': (245, 245, 220),
            'tan': (210, 180, 140),
            'maroon': (128, 0, 0),
            'navy': (0, 0, 128),
            'teal': (0, 128, 128),
            'olive': (128, 128, 0),
        }
    
    def _load_gt_color_mapping(self):
        """Load gt_color_mapping.json."""
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "gt_color_mapping.json"),
            os.path.join(os.path.dirname(__file__), "..", "gt_color_mapping.json"),
            os.path.join(os.getcwd(), "gt_color_mapping.json"),
        ]
        
        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                try:
                    with open(normalized_path, 'r') as f:
                        self.gt_color_mapping = json.load(f)
                    print(f"[ColorValidator] Loaded GT color mapping: {len(self.gt_color_mapping)} entries from {normalized_path}")
                    return
                except Exception as e:
                    print(f"[ColorValidator] Warning: Could not load GT color mapping from {normalized_path}: {e}")
        
        print("[ColorValidator] Warning: GT color mapping not found, using default mapping only")
    
    def _load_color_vocabulary(self):
        """Load the colour vocabulary CSV from `config.colour_vocabulary.VOCABULARY_PATH`."""
        from arto_kg.config.colour_vocabulary import VOCABULARY_PATH

        possible_paths = [VOCABULARY_PATH]

        for path in possible_paths:
            normalized_path = os.path.normpath(path)
            if os.path.exists(normalized_path):
                try:
                    import pandas as pd
                    self.color_vocabulary = pd.read_csv(normalized_path)
                    print(f"[ColorValidator] Loaded color vocabulary: {len(self.color_vocabulary)} entries from {normalized_path}")
                    return
                except Exception as e:
                    print(f"[ColorValidator] Warning: Could not load color vocabulary from {normalized_path}: {e}")
        
        print("[ColorValidator] Warning: Color vocabulary not found, will use GT mapping and defaults only")
    
    def load_color_ontology(self, ontology_path: str):
        """Load color ontology"""
        try:
            import pandas as pd
            self.color_ontology = pd.read_csv(ontology_path)
            print(f"[ColorValidator] Loaded color ontology with {len(self.color_ontology)} entries")
        except Exception as e:
            print(f"[ColorValidator] Warning: Could not load color ontology: {e}")
            print("Will use default color mapping")
    
    # Pixels sampled before clustering the whole image.
    WHOLE_IMAGE_SAMPLE = 4000

    def extract_dominant_colors(self, image_path: str) -> List[Dict[str, Any]]:
        """Whole-image K-means palette with each cluster's share of the frame.

        The cluster count is chosen by `_cluster_by_best_k`. Entries
        have `rgb` and `proportion` (no `color_name`), largest first.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pixels = image_rgb.reshape(-1, 3)

            if len(pixels) > self.WHOLE_IMAGE_SAMPLE:
                indices = np.random.RandomState(42).choice(
                    len(pixels), self.WHOLE_IMAGE_SAMPLE, replace=False)
                pixels = pixels[indices]

            centers, labels = self._cluster_by_best_k(pixels)
            weights = np.bincount(labels, minlength=len(centers)) / len(labels)

            # No `color_name`: consumers only read rgb and proportion, and naming
            # costs a vocabulary sweep per cluster.
            return [
                {
                    'rgb': tuple(int(c) for c in centers[i]),
                    'proportion': round(float(weights[i]) * 100, 1),
                }
                for i in np.argsort(weights)[::-1]
            ]

        except Exception as e:
            print(f"Error extracting colors from {image_path}: {e}")
            return []


    @staticmethod
    def _find_colour_scheme(json_data: Dict[str, Any]) -> Dict[str, Any]:
        """The Environment Design colour scheme, or {}.

        Looks under `environment.environment_details` then `environment`, for
        `colour_scheme` or `color_scheme`.
        """
        environment = json_data.get('environment')
        if not isinstance(environment, dict):
            return {}

        containers = [environment]
        details = environment.get('environment_details')
        if isinstance(details, dict):
            containers.insert(0, details)

        for container in containers:
            for key in ('colour_scheme', 'color_scheme'):
                scheme = container.get(key)
                if isinstance(scheme, dict) and scheme:
                    return scheme
        return {}

    @staticmethod
    def _palette_names(scheme: Dict[str, Any]) -> Dict[str, List[str]]:
        """The scheme's colours in primary / secondary / accent tiers.

        Flat layout: `main_colours`, `background_colour`, `lighting_shift`.
        Nested layout: `main_palette` with `{primary,secondary,accent}_colors`.
        """
        tiers: Dict[str, List[str]] = {'primary': [], 'secondary': [], 'accent': []}
        if not isinstance(scheme, dict):
            return tiers

        main = scheme.get('main_colours')
        if isinstance(main, list) and main:
            tiers['primary'] = [str(c).strip() for c in main if c]

            background = scheme.get('background_colour')
            if background:
                tiers['secondary'] = [str(background).strip()]

            shift = scheme.get('lighting_shift')
            if isinstance(shift, dict):
                tiers['accent'] = [str(shift[key]).strip()
                                   for key in ('light', 'shadow')
                                   if shift.get(key)]
            return tiers

        palette = scheme.get('main_palette')
        if isinstance(palette, dict):
            for tier in tiers:
                values = palette.get(f'{tier}_colors', [])
                if isinstance(values, list):
                    tiers[tier] = [str(c).strip() for c in values if c]
        return tiers

    def extract_expected_colors(self, json_data: Dict[str, Any]) -> Dict[str, Any]:
        """Expected colours, keyed by instance_id (the same key as the boxes)."""
        color_expectations = {
            'object_colors': {},   # instance_id -> {primary, secondary}
            'object_names': {},    # instance_id -> name, for reporting
            'overall_colors': {'primary': [], 'secondary': [], 'accent': []},
        }

        try:
            # 1. Extract object-level color information
            for instance_id, obj in ia.iter_objects(json_data):
                obj_name = obj.get('name', '')
                if not obj_name:
                    continue
                color_expectations['object_names'][instance_id] = obj_name

                obj_colors = {'primary': [], 'secondary': []}

                # Extract object colors from multiple possible fields
                has_specific_colors = False

                # Method 1: declared colours (`main_colours` or `primary_colors`).
                declared = [self._normalize_color_name(c) for c in colours_of(obj)
                            if c and c != "natural color accuracy"]
                if declared:
                    obj_colors['primary'] = declared
                    has_specific_colors = True

                # Method 2: nested color_palette
                color_palette = obj.get('color_palette', {})
                if isinstance(color_palette, dict):
                    if not has_specific_colors:
                        primary_colors = color_palette.get('primary_colors', [])
                        if isinstance(primary_colors, list):
                            specific_primary = [self._normalize_color_name(c) for c in primary_colors
                                              if c and c != "natural color accuracy"]
                            if specific_primary:
                                obj_colors['primary'] = specific_primary
                                has_specific_colors = True

                    # Process color_variations (as secondary)
                    color_variations = color_palette.get('color_variations', [])
                    if isinstance(color_variations, list):
                        specific_variations = [self._normalize_color_name(c) for c in color_variations
                                             if c and c != "light and shadow variations"]
                        if specific_variations:
                            obj_colors['secondary'] = specific_variations
                            has_specific_colors = True

                # Method 3: plain colors list
                if not has_specific_colors:
                    colors = obj.get('colors', [])
                    if isinstance(colors, list):
                        specific_colors = [self._normalize_color_name(c) for c in colors
                                         if c and c != "natural color accuracy"]
                        if specific_colors:
                            obj_colors['primary'] = specific_colors[:3]  # Take top 3 as primary
                            if len(specific_colors) > 3:
                                obj_colors['secondary'] = specific_colors[3:5]  # Take 4th-5th as secondary
                            has_specific_colors = True

                if not has_specific_colors:
                    print(f"   Object '{obj_name}' ({instance_id}) has no specific colors; "
                          "its object-level colour is not measured")
                color_expectations['object_colors'][instance_id] = obj_colors

            # 2. Whole-image colours
            # Method 1: artistic_expression.color_palette
            artistic_expression = json_data.get('artistic_expression', {})
            if isinstance(artistic_expression, dict):
                art_color_palette = artistic_expression.get('color_palette', {})
                if isinstance(art_color_palette, dict):
                    # Overall primary colors
                    primary_colors = art_color_palette.get('primary_colors', [])
                    if isinstance(primary_colors, list):
                        color_expectations['overall_colors']['primary'] = [self._normalize_color_name(c) for c in primary_colors if c]

                    # Overall secondary colors
                    secondary_colors = art_color_palette.get('secondary_colors', [])
                    if isinstance(secondary_colors, list):
                        color_expectations['overall_colors']['secondary'] = [self._normalize_color_name(c) for c in secondary_colors if c]

                    # Overall accent colors
                    accent_colors = art_color_palette.get('accent_colors', [])
                    if isinstance(accent_colors, list):
                        color_expectations['overall_colors']['accent'] = [self._normalize_color_name(c) for c in accent_colors if c]
            
            # Method 2: the Environment Design colour scheme.
            if not any(color_expectations['overall_colors'].values()):
                scheme = self._find_colour_scheme(json_data)
                for level, values in self._palette_names(scheme).items():
                    # Hex values are kept verbatim; names are normalised.
                    color_expectations['overall_colors'][level] = [
                        colour if isinstance(colour, str) and colour.startswith('#')
                        else self._normalize_color_name(str(colour))
                        for colour in values if colour
                    ]
                # Record which scheme field each tier came from.
                if scheme.get('main_colours'):
                    color_expectations['overall_source'] = {
                        'primary': 'colour_scheme.main_colours',
                        'secondary': 'colour_scheme.background_colour',
                        'accent': 'colour_scheme.lighting_shift',
                    }

            # 3. Objects with no declared colour are recorded and not measured.
            undeclared = [instance_id
                          for instance_id, obj_colors
                          in color_expectations['object_colors'].items()
                          if not obj_colors['primary'] and not obj_colors['secondary']]
            if undeclared:
                names = [color_expectations['object_names'].get(i, i) for i in undeclared]
                print(f"   No declared colour for {names}; "
                      "object-level colour is not measured for them")
            color_expectations['objects_without_declared_colour'] = undeclared

        except Exception as e:
            print(f"Warning: Error extracting expected colors: {e}")

        return color_expectations

    def _normalize_color_name(self, color_name: str) -> str:
        """Normalize color name"""
        if not color_name or not isinstance(color_name, str):
            return ""

        color_lower = color_name.lower().strip()
        
        # If hex color code, return directly
        if color_lower.startswith('#') and len(color_lower) in [4, 7]:
            return color_lower

        # Map complex color descriptions to basic colors
        color_mapping = {
            'metallic silver': 'silver',
            'charcoal black': 'black',
            'pure white': 'white',
            'soft blue': 'blue',
            'warm amber': 'orange',
            'stainless steel': 'silver',
            'natural color accuracy': '',  # Ignore this generic description
            'light and shadow variations': '',  # Ignore lighting variation description
        }

        normalized = color_mapping.get(color_lower, color_lower)
        return normalized

    def evaluate_overall_color_accuracy(self, extracted_colors: List[Dict[str, Any]],
                                       expected_colors: Dict[str, List[str]]) -> Dict[str, Any]:
        """Whole-image colour accuracy, measured as at object level.

        Uses `measure_colour_differences` over all declared scheme colours.
        """
        all_expected_colors = []
        for color_list in expected_colors.values():
            all_expected_colors.extend([c for c in color_list if c])  # Filter empty strings

        if not all_expected_colors:
            # No declared scheme: reported as not measured, without a score.
            return {
                'metric': 'CIEDE2000',
                'mean_delta_e': None,
                'max_delta_e': None,
                'colours_measured': 0,
                'colours_matched': 0,
                'status': 'not_measured',
                'details': 'No colour scheme declared in the spec',
            }

        result = self.measure_colour_differences(
            extracted_colors, all_expected_colors, colour_region="whole_image")
        # Re-keyed to the names comprehensive_color_evaluation reads.
        result['colours_measured'] = result.pop('measured', 0)
        result['colours_matched'] = result.pop('matched', 0)
        return result

    # Range searched for the number of salient colours.
    K_RANGE = (2, 10)

    # K-means restarts per k; a fixed seed keeps the palette reproducible.
    KMEANS_N_INIT = 3

    def _cluster_by_best_k(self, pixels):
        """K-means on `pixels`, with k in K_RANGE chosen at the knee of the inertia curve.

        Returns (centres, labels) and sets `last_chosen_k`.
        """
        from sklearn.cluster import KMeans

        low, high = self.K_RANGE
        top = min(high, len(pixels) - 1)

        def single_cluster():
            model = KMeans(n_clusters=1, random_state=42,
                           n_init=self.KMEANS_N_INIT).fit(pixels)
            self.last_chosen_k = 1
            return model.cluster_centers_.astype(int), model.labels_

        # Too few pixels to compare clusterings: one centre, stated as such.
        if top < low:
            return single_cluster()

        models = {}
        for k in range(low, top + 1):
            model = KMeans(n_clusters=k, random_state=42,
                           n_init=self.KMEANS_N_INIT).fit(pixels)
            if len(set(model.labels_)) < 2:
                continue
            models[k] = model

        if not models:
            return single_cluster()

        model = models[self._knee_of(models)]
        self.last_chosen_k = int(model.n_clusters)
        return model.cluster_centers_.astype(int), model.labels_

    @staticmethod
    def _knee_of(models):
        """k at the greatest perpendicular distance from the inertia chord.

        Standard elbow construction; with fewer than three points the smallest k
        is returned.
        """
        ks = sorted(models)
        inertias = [models[k].inertia_ for k in ks]
        if len(ks) < 3:
            return ks[0]

        x1, y1 = ks[0], inertias[0]
        x2, y2 = ks[-1], inertias[-1]
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        if denominator == 0:
            return ks[0]

        best_distance, best_k = -1.0, ks[0]
        for k, inertia in zip(ks, inertias):
            distance = abs((y2 - y1) * k - (x2 - x1) * inertia
                           + x2 * y1 - y2 * x1) / denominator
            if distance > best_distance:
                best_distance, best_k = distance, k
        return best_k

    def extract_colors_from_region(self, image_path: str, bbox: List[int]) -> List[Dict[str, Any]]:
        """K-means colours of an image region, with proportion and vocabulary name, largest first."""
        try:
            import cv2
            from sklearn.cluster import KMeans

            image = cv2.imread(image_path)
            if image is None:
                return []

            # Crop object region, ensure bbox coordinates are numeric and convert to int
            x1 = int(float(bbox[0]))
            y1 = int(float(bbox[1]))
            x2 = int(float(bbox[2]))
            y2 = int(float(bbox[3]))
            
            # Clamp the lower bounds to 0
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = int(x2)
            y2 = int(y2)

            if x2 <= x1 or y2 <= y1:
                return []

            roi = image[y1:y2, x1:x2]

            if roi.size == 0:
                return []

            # Extract dominant colors using K-means
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            pixels = roi_rgb.reshape(-1, 3)

            # Seeded sample of at most 1000 pixels, so results are reproducible.
            if len(pixels) > 1000:
                indices = np.random.RandomState(42).choice(len(pixels), 1000, replace=False)
                pixels = pixels[indices]

            if len(pixels) > 0:
                colors, labels = self._cluster_by_best_k(pixels)

                # Calculate proportion and name for each color
                color_info = []
                for i, color in enumerate(colors):
                    # Ensure labels is array type
                    if hasattr(labels, '__len__') and len(labels) > 0:
                        proportion = np.sum(labels == i) / len(labels)
                    else:
                        proportion = 1.0 / len(colors)  # Distribute evenly

                    color_name = self.rgb_to_color_name_from_vocabulary(tuple(color))
                    color_info.append({
                        'rgb': tuple(int(c) for c in color),  # Ensure integer
                        'proportion': round(proportion * 100, 1),
                        'color_name': color_name
                    })

                # Sort by proportion
                color_info.sort(key=lambda x: x['proportion'], reverse=True)
                return color_info

        except Exception as e:
            print(f"Error extracting colors from region: {e}")
            return []

    def extract_background_colors(self, image_path: str,
                                  object_bboxes: List[List[float]]) -> List[Dict[str, Any]]:
        """K-means over the image with the detected objects cut out.

        Removing whole boxes gives a conservative background sample that
        excludes the objects. Returns [] when under 5% of the image remains.
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                return []

            height, width = image.shape[:2]
            mask = np.ones((height, width), dtype=bool)
            for bbox in object_bboxes or []:
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = (int(round(float(v))) for v in bbox[:4])
                x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
                y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))
                mask[y1:y2, x1:x2] = False

            remaining = int(mask.sum())
            if remaining < 0.05 * height * width:
                print(f"   Objects cover "
                      f"{100 * (1 - remaining / float(height * width)):.0f}% of the "
                      "image; too little background left to sample")
                return []

            pixels = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[mask]
            if len(pixels) > 2000:
                indices = np.random.RandomState(42).choice(len(pixels), 2000, replace=False)
                pixels = pixels[indices]

            if len(pixels) < 1:
                return []

            # k chosen by _cluster_by_best_k, as for object regions.
            centres, labels = self._cluster_by_best_k(pixels)

            colours = []
            for index, centre in enumerate(centres):
                proportion = float(np.sum(labels == index)) / len(labels)
                colours.append({
                    'rgb': tuple(int(c) for c in centre),
                    'proportion': round(proportion * 100, 1),
                })
            colours.sort(key=lambda c: c['proportion'], reverse=True)
            return colours

        except Exception as exc:
            print(f"   Background colour extraction failed: {exc}")
            return []

    def measure_colour_differences(self, extracted_colors: List[Dict[str, Any]],
                                   expected_color_names: List[str],
                                   colour_region: str = "bbox") -> Dict[str, Any]:
        """Delta E between each expected colour and the closest colour present.

        For each declared colour, the smallest CIEDE2000 distance to any
        extracted colour; the cluster's proportion is reported but not used in
        the number. Colours that cannot be resolved are listed as `unresolved`.
        """
        measurements: List[Dict[str, Any]] = []
        unresolved: List[str] = []

        for expected_name in expected_color_names:
            if not expected_name:
                continue

            reference = cr.resolve(expected_name, self.get_color_rgb,
                                   provenance=self.lookup_with_provenance)
            if reference is None:
                unresolved.append(expected_name)
                continue
            expected_rgb = tuple(reference['rgb'])

            best = None
            for extracted_color in extracted_colors:
                distance = cd.delta_e(tuple(extracted_color['rgb']), tuple(expected_rgb))
                if best is None or distance < best['delta_e']:
                    best = {
                        'delta_e': round(distance, 3),
                        'closest_rgb': list(extracted_color['rgb']),
                        'closest_proportion': extracted_color.get('proportion'),
                    }

            if best is None:
                unresolved.append(expected_name)
                continue

            best.update({
                'expected_color': expected_name,
                'expected_rgb': list(expected_rgb),
                'reference': reference,
                'colour_region': colour_region,
                'matched': best['delta_e'] <= cd.MATCH_THRESHOLD,
                'interpretation': cd.describe(best['delta_e']),
            })
            measurements.append(best)

        deltas = [m['delta_e'] for m in measurements]
        matched = [m for m in measurements if m['matched']]

        return {
            'metric': 'CIEDE2000',
            'match_threshold': cd.MATCH_THRESHOLD,
            'colours': measurements,
            'unresolved': unresolved,
            # Lower is better; None when nothing could be measured.
            'mean_delta_e': round(sum(deltas) / len(deltas), 3) if deltas else None,
            'max_delta_e': round(max(deltas), 3) if deltas else None,
            'measured': len(measurements),
            'matched': len(matched),
        }

    def calculate_color_distance(self, color1: Tuple[int, int, int],
                                color2: Tuple[int, int, int]) -> float:
        """Perceptual colour difference (CIEDE2000, roughly 0-100)."""
        try:
            return cd.delta_e(tuple(color1), tuple(color2))
        except Exception as e:
            print(f"Warning: Error calculating color distance: {e}")
            return float('inf')
    
    def find_best_color_match(self, extracted_color: Tuple[int, int, int], 
                             expected_color_name: str) -> Dict[str, Any]:
        """
        Find best expected color match for extracted color
        
        Args:
            extracted_color: Extracted RGB color
            expected_color_name: Expected color name
            
        Returns:
            Match result dictionary
        """
        # Get RGB value for expected color
        expected_rgb = self.get_color_rgb(expected_color_name)
        if expected_rgb is None:
            return {
                'color_name': expected_color_name,
                'match_found': False,
                'error': f'Unknown color name: {expected_color_name}'
            }
        
        # Apply colour modifiers ("muted teal"); see colour_reference.
        reference = cr.resolve(expected_color_name, self.get_color_rgb,
                               provenance=self.lookup_with_provenance)
        if reference is not None:
            expected_rgb = tuple(reference['rgb'])

        comparison = cd.compare(tuple(extracted_color), tuple(expected_rgb))
        delta_e = comparison['delta_e']

        # Reports Delta E (lower is better).
        return {
            'color_name': expected_color_name,
            'expected_rgb': expected_rgb,
            'reference': reference,
            'extracted_rgb': extracted_color,
            # Measured over the bounding box, so enclosed background is included.
            'colour_region': 'bbox',
            'delta_e': delta_e,
            'metric': comparison['metric'],
            'interpretation': comparison['interpretation'],
            'matched': comparison['matched'],
            'match_threshold': cd.MATCH_THRESHOLD,
            'match_found': True,
            # CIE interpretation bands.
            'match_quality': (
                'imperceptible' if delta_e <= cd.JND
                else 'close' if delta_e <= cd.CLOSE_INSPECTION
                else 'similar' if delta_e <= cd.SIMILAR
                else 'different'
            ),
        }
    
    # gt_color_mapping methods that state a colour verbatim. Other methods
    # (`stripped_exact`, `substring`, ...) drop words to find a match.
    AUTHORITATIVE_GT_METHODS = ("exact", "override_exact")

    def lookup_with_provenance(self, color_name: str) -> Optional[Dict[str, Any]]:
        """RGB for a colour name, with the tier that answered and whether it is
        authoritative. Returns None when nothing resolves.
        """
        color_name = (color_name or "").lower().strip()

        def record(rgb, tier, method, authoritative):
            if rgb is None:
                return None
            return {"rgb": tuple(int(c) for c in rgb), "tier": tier,
                    "method": method, "authoritative": authoritative}

        # 1. Hex states the colour outright.
        if color_name.startswith('#') and len(color_name) in [4, 7]:
            try:
                rgb = self.hex_to_rgb(color_name)
                if rgb is not None:
                    return record(rgb, "hex", "hex", True)
            except Exception:
                pass

        # 2. Colour vocabulary, exact name match.
        if self.color_vocabulary is not None:
            try:
                matches = self.color_vocabulary[
                    self.color_vocabulary['color_name'].str.lower() == color_name
                ]
                if not matches.empty:
                    rgb = self.hex_to_rgb(matches.iloc[0]['hex_string'])
                    if rgb is not None:
                        return record(rgb, "color_vocabulary", "exact", True)
            except Exception:
                pass

        # 3. GT colour mapping - free colour phrases.
        if color_name in self.gt_color_mapping:
            info = self.gt_color_mapping[color_name]
            method = info.get('method')
            rgb = None
            if 'hex' in info:
                rgb = self.hex_to_rgb(info['hex'])
            elif 'rgb' in info:
                rgb = tuple(info['rgb'])
            if rgb is not None:
                return record(rgb, "gt_color_mapping", method,
                              method in self.AUTHORITATIVE_GT_METHODS)

        # 4. Optional colour ontology, matched on the whole name.
        if self.color_ontology is not None:
            try:
                matches = self.color_ontology[
                    self.color_ontology['color_name'].str.lower() == color_name
                ]
                if not matches.empty:
                    row = matches.iloc[0]
                    return record((row['red'], row['green'], row['blue']),
                                  "color_ontology", "exact", True)
            except Exception:
                pass

        # 5. Default mapping, exact key.
        if color_name in self.default_color_mapping:
            return record(self.default_color_mapping[color_name],
                          "default_mapping", "exact", True)

        # 6. Substring match against the default mapping; marked non-authoritative.
        for key, rgb in self.default_color_mapping.items():
            if color_name in key or key in color_name:
                return record(rgb, "default_mapping", "substring", False)

        return None

    def get_color_rgb(self, color_name: str) -> Optional[Tuple[int, int, int]]:
        """
        RGB for a colour name or hex code (`lookup_with_provenance` without provenance).

        Resolution order:
        1. Hex code (if starts with #)
        2. Color vocabulary
        3. GT color mapping
        4. Color ontology (if provided)
        5. Default mapping (exact match)
        6. Substring match in default mapping

        Args:
            color_name: Color name or hex code

        Returns:
            RGB tuple or None
        """
        found = self.lookup_with_provenance(color_name)
        return found["rgb"] if found else None

    def hex_to_rgb(self, hex_string):
        """Convert hex string to RGB"""
        try:
            hex_string = hex_string.lstrip('#')
            if len(hex_string) == 6:
                return tuple(int(hex_string[i:i+2], 16) for i in (0, 2, 4))
            else:
                return None
        except:
            return None

    def rgb_to_color_name_from_vocabulary(self, rgb_values):
        """Name of the perceptually closest (CIEDE2000) vocabulary colour."""
        color_vocab = self.color_vocabulary
        if color_vocab is None or color_vocab.empty:
            return 'unknown'

        try:
            min_distance = float('inf')
            closest_color = 'unknown'

            for _, row in color_vocab.iterrows():
                vocab_rgb = self.hex_to_rgb(row['hex_string'])
                if vocab_rgb:
                    distance = cd.delta_e(tuple(rgb_values), tuple(vocab_rgb))
                    if distance < min_distance:
                        min_distance = distance
                        closest_color = row['color_name']

            return closest_color
        except Exception as e:
            print(f"Error converting RGB to color name: {e}")
            return 'unknown'
    
    def main_evaluation(self, json_data: Dict[str, Any], 
                       image_path: str,
                       bboxes_dict: Optional[Dict[str, List[int]]] = None,
                       visualize: bool = False) -> Dict[str, Any]:
        """Comprehensive color evaluation - Main entry point"""
        # If bboxes are provided, perform object-based color analysis
        if bboxes_dict:
            return self.comprehensive_color_evaluation_with_objects(json_data, image_path, bboxes_dict, visualize=visualize)
        else:
            return self.comprehensive_color_evaluation(json_data, image_path, visualize=visualize)
    
    def comprehensive_color_evaluation(self, json_data: Dict[str, Any],
                                     image_path: str,
                                     visualize: bool = False) -> Dict[str, Any]:
        """
        Whole-image colour evaluation.

        Args:
            json_data: Artwork JSON data
            image_path: Image file path
            visualize: Whether to generate visualization
            
        Returns:
            Complete color evaluation results
        """
        print(f"Starting comprehensive color evaluation for: {os.path.basename(image_path)}")
        
        try:
            # 1. Extract dominant colors
            print("   Step 1: Extracting dominant colors...")
            extracted_colors = self.extract_dominant_colors(image_path)
            if not extracted_colors:
                return {'error': 'Failed to extract colors from image'}
            
            print(f"   Extracted {len(extracted_colors)} dominant colors")
            
            # 2. Extract expected colors (object level and whole image)
            print("   Step 2: Extracting expected colors from JSON...")
            color_expectations = self.extract_expected_colors(json_data)

            # Count color expectations
            object_colors_count = sum(len(colors['primary']) + len(colors['secondary'])
                                    for colors in color_expectations['object_colors'].values())
            overall_colors_count = sum(len(colors) for colors in color_expectations['overall_colors'].values())

            print(f"   Found {object_colors_count} object-level colors, {overall_colors_count} overall colors")

            # 3. Evaluate overall color accuracy
            print("   Step 3: Evaluating overall color accuracy...")
            accuracy_result = self.evaluate_overall_color_accuracy(extracted_colors, color_expectations['overall_colors'])
            
            # 4. Report the measurement
            print("   Step 4: Summarising colour differences...")

            # Delta E, lower is better.
            mean_delta_e = accuracy_result.get('mean_delta_e')

            evaluation_result = {
                'metric': 'CIEDE2000',
                'match_threshold': cd.MATCH_THRESHOLD,
                'mean_delta_e': mean_delta_e,
                'max_delta_e': accuracy_result.get('max_delta_e'),
                'colours_measured': accuracy_result.get('colours_measured', 0),
                'colours_matched': accuracy_result.get('colours_matched', 0),
                'unresolved': accuracy_result.get('unresolved', 0),
                'detailed_results': {
                    'accuracy': accuracy_result,
                    'extracted_colors': extracted_colors,
                    'expected_colors': color_expectations
                },
                'evaluation_summary': {
                    'total_colors_extracted': len(extracted_colors),
                    'total_colors_expected': object_colors_count + overall_colors_count,
                },
                'image_file': image_path,
                'artwork_id': json_data.get('artwork_id', 'unknown')
            }
            
            if visualize:
                output_dir = os.path.join(os.path.dirname(image_path), "color_analysis")
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"color_analysis_{os.path.basename(image_path)}")
                self.create_color_visualization(evaluation_result, save_path)
                evaluation_result['visualization_path'] = save_path

            if mean_delta_e is None:
                print("   Colour evaluation completed. No colour was measurable.")
            else:
                print(f"   Colour evaluation completed. Mean Delta E {mean_delta_e:.2f} "
                      f"({evaluation_result['colours_matched']}/"
                      f"{evaluation_result['colours_measured']} within Delta E {cd.MATCH_THRESHOLD})")
            return evaluation_result
            
        except Exception as e:
            error_msg = f"Comprehensive color evaluation failed: {e}"
            print(f"   {error_msg}")
            return {'error': error_msg}
    
    def comprehensive_color_evaluation_with_objects(self, json_data: Dict[str, Any],
                                                   image_path: str,
                                                   bboxes_dict: Dict[str, List[int]],
                                                   visualize: bool = False) -> Dict[str, Any]:
        """Comprehensive color evaluation based on detected objects"""
        try:
            print(f"   Starting object-based color evaluation...")
            
            # 1. Read image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': f'Could not load image: {image_path}'}
            
            # 2. Global color analysis
            global_result = self.comprehensive_color_evaluation(json_data, image_path, visualize=visualize)
            if 'error' in global_result:
                return global_result
            
            # 3. Object-based color analysis; boxes and expectations are both
            # keyed by instance_id.
            color_expectations = self.extract_expected_colors(json_data)
            expectations = color_expectations['object_colors']
            object_names = color_expectations['object_names']
            object_color_analysis = {}

            for instance_id, bbox in bboxes_dict.items():
                label = object_names.get(instance_id, instance_id)
                print(f"     Analyzing colors for {instance_id} ({label})")

                if len(bbox) < 4:
                    continue

                # Clamp the box to the image before extracting colours.
                x1 = max(0, min(int(float(bbox[0])), image.shape[1] - 1))
                y1 = max(0, min(int(float(bbox[1])), image.shape[0] - 1))
                x2 = max(x1 + 1, min(int(float(bbox[2])), image.shape[1]))
                y2 = max(y1 + 1, min(int(float(bbox[3])), image.shape[0]))

                obj_colors = self.extract_colors_from_region(
                    image_path, [x1, y1, x2, y2])

                declared = expectations.get(instance_id, {})
                expected_colors = (declared.get('primary', [])
                                   + declared.get('secondary', []))

                # An object with no declared colour is not measured.
                if expected_colors and obj_colors:
                    color_match = self.measure_colour_differences(
                        obj_colors, expected_colors)
                else:
                    color_match = {
                        'mean_delta_e': None,
                        'status': 'not_measured',
                        'note': ('no colour declared for this object'
                                 if not expected_colors
                                 else 'no colour could be sampled from the box'),
                    }

                object_color_analysis[instance_id] = {
                    'name': label,
                    'dominant_colors': obj_colors,
                    'expected_colors': expected_colors,
                    'color_match': color_match,
                    'colour_region': 'bbox',
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                }

            # 4. Report per-object Delta E alongside the whole-image figure,
            # without combining them.
            object_deltas = [obj['color_match'].get('mean_delta_e')
                             for obj in object_color_analysis.values()]
            object_deltas = [d for d in object_deltas if d is not None]

            enhanced_result = global_result.copy()
            enhanced_result['object_mean_delta_e'] = (
                round(sum(object_deltas) / len(object_deltas), 3) if object_deltas else None)
            enhanced_result['object_color_analysis'] = object_color_analysis
            enhanced_result['evaluation_summary']['objects_analyzed'] = len(object_color_analysis)
            
            object_mean = enhanced_result['object_mean_delta_e']
            if object_mean is None:
                print("   Object-based colour evaluation completed. "
                      "No object colour was measurable.")
            else:
                print(f"   Object-based colour evaluation completed. "
                      f"Mean Delta E per object {object_mean:.2f} "
                      f"(whole image {global_result.get('mean_delta_e')})")
            return enhanced_result
            
        except Exception as e:
            error_msg = f"Object-based color evaluation failed: {e}"
            print(f"   {error_msg}")
            return {'error': error_msg}
    
    def create_color_visualization(self, evaluation_result: Dict[str, Any], 
                                 save_path: Optional[str] = None) -> None:
        """
        Create visualization of color evaluation results
        
        Args:
            evaluation_result: Evaluation results
            save_path: Save path
        """
        if 'error' in evaluation_result:
            print(f"Cannot create visualization due to error: {evaluation_result['error']}")
            return
        
        try:
            threshold = evaluation_result.get('match_threshold', cd.MATCH_THRESHOLD)
            mean_delta_e = evaluation_result.get('mean_delta_e')
            measurements = (evaluation_result.get('detailed_results', {})
                            .get('accuracy', {}).get('colours') or [])

            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            title = ('no colour measured' if mean_delta_e is None
                     else f'mean Delta E {mean_delta_e:.2f}')
            fig.suptitle(f'Colour Evaluation ({title})', fontsize=16, fontweight='bold')

            # 1. Delta E per declared colour; lower is better
            names = [m['expected_color'] for m in measurements]
            deltas = [m['delta_e'] for m in measurements]
            axes[0, 0].barh(names, deltas, alpha=0.7,
                            color=['green' if m['matched'] else 'red' for m in measurements])
            axes[0, 0].axvline(threshold, color='black', linestyle='--',
                               label=f'match threshold Delta E {threshold}')
            axes[0, 0].invert_yaxis()
            axes[0, 0].set_xlabel('CIEDE2000 to the closest extracted colour')
            axes[0, 0].set_title('Declared colours')
            axes[0, 0].legend(loc='lower right')

            # 2. Extracted dominant colours, with their share of the image
            extracted_colors = evaluation_result.get('detailed_results', {}).get('extracted_colors', [])[:12]
            for i, color in enumerate(extracted_colors):
                row, col = divmod(i, 6)
                axes[0, 1].add_patch(plt.Rectangle((col, 1 - row), 1, 1,
                                                   facecolor=np.array(color['rgb']) / 255.0,
                                                   edgecolor='black', linewidth=1))
                if color.get('proportion') is not None:
                    axes[0, 1].text(col + 0.5, 1 - row - 0.1, f"{color['proportion']:.0f}%",
                                    ha='center', va='top', fontsize=9)
            axes[0, 1].set_xlim(0, 6)
            axes[0, 1].set_ylim(-0.3, 2)
            axes[0, 1].set_title('Extracted dominant colours')
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])

            # 3. Each declared colour beside the closest extracted one
            for y_pos, m in enumerate(measurements[:15]):
                axes[1, 0].add_patch(plt.Rectangle((0, y_pos), 1, 0.8,
                                                   facecolor=np.array(m['expected_rgb']) / 255.0,
                                                   edgecolor='black', linewidth=1))
                axes[1, 0].add_patch(plt.Rectangle((1.1, y_pos), 1, 0.8,
                                                   facecolor=np.array(m['closest_rgb']) / 255.0,
                                                   edgecolor='black', linewidth=1))
                axes[1, 0].text(2.2, y_pos + 0.4, f"{m['expected_color']}: Delta E {m['delta_e']:.1f}",
                                ha='left', va='center', fontsize=9)
            axes[1, 0].set_xlim(0, 6)
            axes[1, 0].set_ylim(max(len(measurements[:15]), 1), 0)
            axes[1, 0].set_title('Declared (left) vs closest extracted (right)')
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

            # 4. Summary
            summary = evaluation_result.get('evaluation_summary', {})
            max_delta_e = evaluation_result.get('max_delta_e')
            unresolved = evaluation_result.get('unresolved', 0)
            if isinstance(unresolved, list):
                unresolved = len(unresolved)
            summary_text = (
                f"Metric: {evaluation_result.get('metric', 'CIEDE2000')}\n"
                f"Mean Delta E: {'-' if mean_delta_e is None else f'{mean_delta_e:.2f}'}\n"
                f"Max Delta E: {'-' if max_delta_e is None else f'{max_delta_e:.2f}'}\n\n"
                f"Matched (Delta E <= {threshold}): {evaluation_result.get('colours_matched', 0)}"
                f"/{evaluation_result.get('colours_measured', 0)}\n"
                f"Unresolved colour names: {unresolved}\n\n"
                f"Colours extracted: {summary.get('total_colors_extracted', '-')}\n"
                f"Colours expected: {summary.get('total_colors_expected', '-')}")
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                            fontsize=11, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
            axes[1, 1].set_title('Summary')
            axes[1, 1].axis('off')

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Color visualization saved to: {save_path}")

            plt.close(fig)

        except Exception as e:
            print(f"Error creating color visualization: {e}")


# Export main class
__all__ = ['ColorValidator']


