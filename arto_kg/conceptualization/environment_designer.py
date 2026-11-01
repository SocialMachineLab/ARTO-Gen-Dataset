import json
import random
import re
from typing import List, Dict, Any, Optional
from arto_kg.conceptualization.utils import setup_logger, REPLY_RECORD_KEYS
from arto_kg.conceptualization.object_instance import colours_of
from arto_kg.config.model_config import (
    TIME_OF_DAY_OPTIONS,
    WEATHER_OPTIONS,
    PERIOD_OPTIONS,
    LIGHTING_TYPE_OPTIONS,
    LIGHTING_QUALITY_OPTIONS,
    LIGHTING_DIRECTION_OPTIONS,
    COMMON_PLACE365_CATEGORIES,
    PLACE365_INDOOR,
    SCENE_PERIOD_MAPPING
)
# Imported as a module so generation and validation share its lazy CSV load.
from arto_kg.config import colour_vocabulary


def _mentions(term: str, text: str) -> bool:
    """`term` occurs in `text` as a whole word, a plural allowed.

    Substring matching would read "bar" out of "Baroque" or "shop" out of "workshop".
    """
    return re.search(r"\b" + re.escape(term) + r"(?:e?s)?\b", text) is not None


class EnvironmentDesigner:
    """Environment Designer"""

    def __init__(self, vllm_wrapper):
        self.logger = setup_logger("environment_designer")
        self.vllm_wrapper = vllm_wrapper

        # Semantic fallback categories. Each fallback is a Places365 category,
        # so it has an id and an indoor/outdoor label.
        self.SEMANTIC_CATEGORIES = {
            "indoor_residential": {
                "keywords": ["home", "house", "apartment", "room", "living", "bedroom", "bathroom",
                           "kitchen", "dining", "corridor", "hallway", "closet", "residence"],
                "fallback": ["/l/living_room", "/b/bedroom", "/k/kitchen", "/d/dining_room", "/b/bathroom"]
            },
            "commercial": {
                "keywords": ["shop", "store", "market", "restaurant", "cafe", "bar", "office",
                           "workspace", "mall", "supermarket", "boutique", "pub", "diner",
                           "business", "commercial", "retail", "desk"],
                "fallback": ["/g/general_store/indoor", "/r/restaurant", "/o/office", "/c/coffee_shop",
                             "/b/bar", "/m/market/indoor", "/m/market/outdoor"]
            },
            "cultural_education": {
                "keywords": ["museum", "gallery", "art", "artist", "studio", "theater", "stage",
                           "school", "classroom", "library", "education", "exhibition",
                           "performance", "concert", "lecture", "learning", "creative"],
                "fallback": ["/a/art_studio", "/m/museum/indoor", "/a/art_gallery", "/l/library/indoor",
                             "/c/classroom", "/s/stage/indoor"]
            },
            "outdoor": {
                "keywords": ["outdoor", "outside", "park", "garden", "forest", "beach", "mountain",
                           "field", "nature", "street", "road", "countryside", "path", "trail",
                           "landscape", "natural", "meadow", "wilderness", "wetland", "marsh", "swamp"],
                "fallback": ["/p/park", "/f/forest/broadleaf", "/b/beach", "/s/street", "/f/field/wild",
                             "/f/formal_garden", "/m/marsh"]
            },
            "sports_entertainment": {
                "keywords": ["sports", "gym", "stadium", "court", "playground", "pool", "exercise",
                           "amusement", "arcade", "game", "entertainment", "recreation",
                           "club", "nightclub", "cinema", "theater", "leisure", "play"],
                "fallback": ["/g/gymnasium/indoor", "/s/stadium/football", "/s/stadium/soccer",
                             "/p/playground", "/a/amusement_park", "/r/recreation_room"]
            },
            "transportation_industrial": {
                "keywords": ["airport", "station", "train", "subway", "bus", "parking", "garage",
                           "factory", "warehouse", "workshop", "industrial", "construction",
                           "junkyard", "storage", "vehicle", "terminal"],
                "fallback": ["/a/airport_terminal", "/t/train_station/platform", "/s/storage_room",
                             "/r/repair_shop", "/g/garage/indoor", "/j/junkyard"]
            },
            "healthcare": {
                "keywords": ["hospital", "clinic", "medical", "doctor", "health", "surgery",
                           "emergency", "patient", "pharmacy", "veterinary", "vet", "treatment"],
                "fallback": ["/h/hospital", "/h/hospital_room", "/p/pharmacy", "/v/veterinarians_office"]
            },
            "universal": {
                "keywords": [],  # Fallback category
                "fallback": ["/a/art_studio", "/p/park", "/f/field/wild", "/l/living_room", "/s/street"]
            }
        }

        # LLM Environment Selection System Prompt
        self.environment_selection_prompt = """You are an expert environment designer for artistic compositions.

Your task is to select the most suitable environmental parameters from the provided candidates to create the best atmosphere for the artwork.

Consider:
- The scene description and its implied setting
- The objects present and their roles
- The art style and its atmospheric requirements
- How time, weather, and lighting interact to create mood

Return ONLY valid JSON with your selections and reasoning."""

        # Colour selection takes two passes: first the colour families, then
        # specific names from those families, since the full vocabulary is too
        # long for one prompt.
        self.colour_family_prompt = """
You are an expert colour designer for artworks. Given the environment, objects and art style, choose which colour families the artwork's palette should be drawn from.

Choose 2 to 4 families. Fewer families gives a tighter, more coherent palette; more gives contrast. Consider the lighting, the time of day and the style.

Respond ONLY with a valid JSON object:
{
    "families": ["family name", "family name"],
    "reasoning": "why these families suit this scene and style"
}"""

        self.colour_scheme_prompt = """
You are an expert colour designer for artworks. Choose the artwork's colours from the named list you are given.

Every colour you name MUST be copied exactly from that list. Do not invent colour names, do not write hex codes, and do not add adjectives - "muted teal" is not a valid answer, but "Teal" or "Slate Gray" may be.

Respond ONLY with a valid JSON object:
{
    "main_colours": ["name", "name", "name", "name", "name"],
    "harmony_type": "complementary/analogous/triadic/monochromatic",
    "contrast_level": "high/medium/low",
    "colour_temperature": "warm/cool/mixed",
    "background_colour": "name",
    "lighting_shift": {"light": "name", "shadow": "name"},
    "reasoning": "why this palette suits the scene, lighting and style"
}

Notes:
- main_colours: exactly 5 names, the artwork's palette. They are a set, not ranked.
- background_colour: the dominant colour of the setting behind the objects.
- lighting_shift: the colour the light casts, and the colour the shadows take. At sunrise light may be golden and shadows cool blue.
- harmony_type, contrast_level and colour_temperature: answer with ONE word copied from the options above, not the whole list."""
    
    def determine_environment_details(self, scene_brief: str, style: str,
                                     objects: List[str] = None,
                                     object_interpretations: Dict = None) -> Dict[str, Any]:
        """Determine the environment parameters for a scene.

        Rules narrow the candidates, the LLM selects among them, then the colour
        scheme is designed.
        """
        self.logger.info(f"Determining environment for: {scene_brief}, style: {style}")

        # Step 1: a rule-derived period, offered to the model as a suggestion
        # only; the model sees more context and makes the final choice.
        suggested_period = self._determine_period(scene_brief, style)
        self.logger.info(f"Period suggested by rules: {suggested_period}")

        # Step 2: Get Place365 Candidates
        place365_candidates = self._get_place365_candidates(scene_brief)
        self.logger.info(f"Place365 candidates: {len(place365_candidates)} found")

        # Step 3: Let LLM Select Environment Parameters
        try:
            llm_selections = self._llm_select_environment(
                scene_brief, style, suggested_period, objects, object_interpretations,
                place365_candidates
            )

            # Validate Selections
            validated = self._validate_llm_selections(
                llm_selections, place365_candidates, suggested_period)

            # Build Final Environment Parameters
            period = validated["period"]["selected"]
            if period != suggested_period:
                self.logger.info(
                    f"Period: model chose {period} over the suggested {suggested_period}")
            environment = self._build_environment_details(scene_brief, period, validated)

        except Exception as e:
            self.logger.error(f"LLM environment selection failed: {e}, using fallback")
            environment = self._fallback_environment(
                scene_brief, suggested_period, place365_candidates)

        # Step 4: Colour scheme - the sixth environment dimension.
        environment["colour_scheme"] = self.design_color_scheme(
            objects or [], style, environment
        )
        return environment

    def _determine_period(self, scene_brief: str, style: str) -> str:
        """Infer period from scene description and style"""
        scene_lower = scene_brief.lower()
        style_lower = style.lower()

        # Match period keywords as whole words, so "classical" does not match
        # "neoclassical".
        for key, period in SCENE_PERIOD_MAPPING.items():
            if key != "default" and _mentions(key, scene_lower):
                return period

        # Infer from style
        style_period_map = {
            "renaissance": "Early modern",
            "baroque": "Early modern",
            "neoclassicism": "Early modern",
            "post-impressionism": "Modern",
            "impressionism": "Modern",
            "medieval": "Post-classical",
            "gothic": "Post-classical",
            "classical": "Ancient",
            "ancient": "Ancient",
            "contemporary": "Contemporary",
            "modern": "Modern",
            "photorealistic": "Contemporary",
            "abstract": "Contemporary",
            "chinese ink": "Contemporary",
        }
        
        for key, period in style_period_map.items():
            if key in style_lower:
                return period

        return "Contemporary"  # default

    def _get_place365_candidates(self, scene_brief: str) -> List[str]:
        """Get Places365 candidates for a scene.

        Keyword matching, then word-overlap filtering if too many, or semantic fallback if none.
        """
        scene_lower = scene_brief.lower()

        # Step 1: Match keywords from COMMON_PLACE365_CATEGORIES
        candidates = []
        matched_keys = []
        for key, place_list in COMMON_PLACE365_CATEGORIES.items():
            if _mentions(key, scene_lower):
                candidates.extend(place_list)
                matched_keys.append(key)

        # Deduplicate, keeping the first occurrence, so the menu order is the
        # same on every run.
        candidates = list(dict.fromkeys(candidates))

        if matched_keys:
            self.logger.info(f"Matched keywords: {matched_keys}, got {len(candidates)} candidates")

        # Step 2: Check candidate count
        if len(candidates) == 0:
            # Case A: No match -> Semantic fallback
            self.logger.info(f"No Place365 keyword match in '{scene_brief}', using semantic fallback")
            candidates = self._semantic_fallback(scene_brief)

        elif len(candidates) > 15:
            # Case B: Too many candidates -> Word overlap filtering
            self.logger.info(f"Too many candidates ({len(candidates)}), filtering by word overlap")
            candidates = self._filter_by_word_overlap(candidates, scene_brief, max_count=10)

        # Case C: 1-15 candidates -> Use directly
        self.logger.info(f"Final Place365 candidates ({len(candidates)}): {candidates[:5]}...")
        return candidates

    def _extract_keywords(self, scene_brief: str) -> List[str]:
        """Extract keywords from scene_brief"""
        # Stopwords list
        stopwords = {"a", "an", "the", "and", "or", "with", "featuring", "centered", "around", "in", "on", "at"}

        # Tokenize and filter
        words = scene_brief.lower().replace(",", " ").split()
        keywords = [w.strip() for w in words if w.strip() and w.strip() not in stopwords]

        return keywords

    def _filter_by_word_overlap(self, candidates: List[str], scene_brief: str, max_count: int = 10) -> List[str]:
        """Rank candidates by how many scene_brief keywords their path contains."""
        keywords = self._extract_keywords(scene_brief)

        # Score each candidate
        scores = {}
        for candidate in candidates:
            candidate_lower = candidate.lower()
            score = 0
            for keyword in keywords:
                # Prefix match: keyword starts a path segment or an underscore-separated part
                if f"/{keyword}" in candidate_lower or f"_{keyword}" in candidate_lower:
                    score += 2  # Prefix match gets 2 points
                # Fuzzy match: Keyword appears as a substring
                elif keyword in candidate_lower:
                    score += 1  # Fuzzy match gets 1 point
            scores[candidate] = score

        # Sort by score
        sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Take top max_count
        result = []
        for candidate, score in sorted_candidates:
            if len(result) >= max_count:
                break
            # Prioritize those with scores, if not enough, fill with those with 0 score
            if score > 0 or len(result) < 3:
                result.append(candidate)

        # If still not enough, complete randomly
        if len(result) < 3:
            remaining = [c for c in candidates if c not in result]
            result.extend(random.sample(remaining, min(3 - len(result), len(remaining))))

        return result

    def _semantic_fallback(self, scene_brief: str) -> List[str]:
        """Pick candidates from the first semantic category whose keywords match."""
        scene_lower = scene_brief.lower()

        # Iterate through the semantic categories
        for category_name, category_data in self.SEMANTIC_CATEGORIES.items():
            if category_name == "universal":
                continue  # Skip fallback category

            # Check if keywords match this category
            keywords = category_data["keywords"]
            for keyword in keywords:
                if _mentions(keyword, scene_lower):
                    self.logger.info(f"Semantic fallback matched category: {category_name}")
                    return category_data["fallback"]

        # No match, use universal fallback
        self.logger.info("Using universal fallback category")
        return self.SEMANTIC_CATEGORIES["universal"]["fallback"]

    def _llm_select_environment(self, scene_brief: str, style: str, period: str,
                               objects: List[str], object_interpretations: Dict,
                               place365_candidates: List[str]) -> Dict[str, Any]:
        """Let the LLM select environment parameters from the candidates."""
        # Build prompt
        user_prompt = self._build_selection_prompt(
            scene_brief, style, period, objects, object_interpretations, place365_candidates
        )

        # Call LLM
        response = self.vllm_wrapper.generate_json_response(
            self.environment_selection_prompt,
            user_prompt
        )

        if "error" in response:
            raise Exception(f"LLM returned error: {response.get('error')}")

        return response

    def _build_selection_prompt(self, scene_brief: str, style: str, period: str,
                                objects: List[str], object_interpretations: Dict,
                                place365_candidates: List[str]) -> str:
        """Build LLM selection prompt"""

        # Format object info
        objects_str = ', '.join(objects) if objects else "N/A"
        interp_str = json.dumps(object_interpretations, indent=2) if object_interpretations else "N/A"

        # Format candidate list
        place365_str = '\n'.join([f"  - {p}" for p in place365_candidates])
        time_str = '\n'.join([f"  - {t}" for t in TIME_OF_DAY_OPTIONS])
        weather_str = '\n'.join([f"  - {w}" for w in WEATHER_OPTIONS + [None]])
        period_str = '\n'.join([f"  - {p}" for p in PERIOD_OPTIONS])
        light_type_str = ' / '.join(LIGHTING_TYPE_OPTIONS)
        light_quality_str = ' / '.join(LIGHTING_QUALITY_OPTIONS)
        light_direction_str = ' / '.join(LIGHTING_DIRECTION_OPTIONS)

        prompt = f"""SCENE DESCRIPTION:
{scene_brief}

OBJECTS IN SCENE:
{objects_str}

OBJECT DETAILS:
{interp_str}

ART STYLE:
{style}

PERIOD SUGGESTED BY THE SCENE AND STYLE:
{period}

---

SELECT the best option from each candidate list below:

1. PLACE365 LOCATION (select ONE):
{place365_str}

2. TIME OF DAY (select ONE):
{time_str}

3. WEATHER (select ONE, or null if not visible/relevant):
{weather_str}

4. HISTORICAL PERIOD (select ONE; "{period}" is suggested, override it if the
   scene and style point elsewhere):
{period_str}

---

RESPONSE FORMAT (JSON only, no other text):
{{
  "place365": {{
    "selected": "/category/place_name",
    "reasoning": "Why this location best fits the scene and style"
  }},
  "time_of_day": {{
    "selected": "morning/afternoon/etc",
    "reasoning": "Why this time creates the best atmosphere"
  }},
  "weather": {{
    "selected": "sunny/cloudy/etc or null",
    "is_visible": true/false,
    "reasoning": "Whether weather is visible and which fits best"
  }},
  "period": {{
    "selected": "one of the five epochs listed above",
    "reasoning": "Why this epoch fits the scene, objects and style"
  }},
  "lighting": {{
    "type": "one of: {light_type_str}",
    "quality": "one of: {light_quality_str}",
    "direction": "one of: {light_direction_str}",
    "mood": "free text - describe the mood the light creates",
    "reasoning": "Lighting setup based on time, weather, and style"
  }}
}}

IMPORTANT RULES:
- For place365, time_of_day, weather, period, and lighting type/quality/direction:
  answer with ONE value copied exactly from the list. Do not invent a value and
  do not answer with the list itself.
- lighting mood and every reasoning field are free text - write them yourself.
- Consider BOTH scene description AND art style
- Consider the objects and their roles
- For weather: indoor scenes CAN have visible weather through windows
- Return ONLY valid JSON, no additional text
"""
        return prompt

    # Lighting keys the schema allows; any other key from the LLM is dropped.
    LIGHTING_FIELDS = ("type", "quality", "direction", "mood", "reasoning")

    # Axes with a fixed vocabulary; `mood` and `reasoning` are free text.
    LIGHTING_VOCABULARIES = {
        "type": set(LIGHTING_TYPE_OPTIONS),
        "quality": set(LIGHTING_QUALITY_OPTIONS),
        "direction": set(LIGHTING_DIRECTION_OPTIONS),
    }
    LIGHTING_DEFAULTS = {
        "type": "natural",
        "quality": "soft",
        "direction": "front-lit",
        "mood": "neutral",
        "reasoning": "",
    }

    def _validate_llm_selections(self, llm_response: Dict[str, Any],
                                 place365_candidates: List[str],
                                 suggested_period: str = "Modern") -> Dict[str, Any]:
        """Validate LLM choices, tolerating missing or malformed blocks.

        Dimensions with a vocabulary are checked against it; free-text fields
        (lighting mood, reasoning) are kept as written.
        """
        validated = dict(llm_response)

        def block(name: str) -> Dict[str, Any]:
            value = validated.get(name)
            if not isinstance(value, dict):
                value = {}
            validated[name] = value
            return value

        # Validate place365
        place_block = block("place365")
        if place_block.get("selected") not in place365_candidates:
            self.logger.warning(
                f"Invalid place365 selection: {place_block.get('selected')}, using first candidate"
            )
            place_block["selected"] = place365_candidates[0]

        # Menu answers are matched case-insensitively and stored in the menu's
        # own spelling.
        def from_menu(value: Any, options: List[str]) -> Optional[str]:
            text = str(value or "").strip().lower()
            return next((option for option in options if option.lower() == text), None)

        # Validate time_of_day
        time_block = block("time_of_day")
        time_choice = from_menu(time_block.get("selected"), TIME_OF_DAY_OPTIONS)
        if time_choice is None:
            self.logger.warning(
                f"Invalid time_of_day selection: {time_block.get('selected')}, using afternoon"
            )
            time_choice = "afternoon"
        time_block["selected"] = time_choice

        # Validate weather (null is a legitimate choice for indoor scenes)
        weather_block = block("weather")
        selected_weather = weather_block.get("selected")
        if selected_weather is not None:
            weather_choice = from_menu(selected_weather, WEATHER_OPTIONS)
            if weather_choice is None:
                self.logger.warning(f"Invalid weather selection: {selected_weather}, using null")
            weather_block["selected"] = weather_choice

        # Validate period. The rule-derived value is only a suggestion and is
        # used when the model's choice is not in the vocabulary.
        period_block = block("period")
        period_choice = from_menu(period_block.get("selected"), PERIOD_OPTIONS)
        if period_choice is None:
            if period_block.get("selected") is not None:
                self.logger.warning(
                    f"Invalid period selection: {period_block.get('selected')}, "
                    f"falling back to the rule-derived {suggested_period}"
                )
            period_choice = suggested_period
        period_block["selected"] = period_choice

        # Validate lighting: fixed key set, defaults for missing, extras dropped.
        lighting = validated.get("lighting")
        if not isinstance(lighting, dict):
            lighting = {}
        dropped = [key for key in lighting if key not in self.LIGHTING_FIELDS]
        if dropped:
            self.logger.warning(f"Dropped unexpected lighting fields: {dropped}")
        lighting = {
            field: lighting.get(field) or self.LIGHTING_DEFAULTS[field]
            for field in self.LIGHTING_FIELDS
        }

        # Check the vocabulary axes, so a menu copied back verbatim is rejected.
        for field, allowed in self.LIGHTING_VOCABULARIES.items():
            choice = from_menu(lighting.get(field), sorted(allowed))
            if choice is None:
                self.logger.warning(
                    f"Invalid lighting {field}: {lighting.get(field)}, "
                    f"using {self.LIGHTING_DEFAULTS[field]}"
                )
                choice = self.LIGHTING_DEFAULTS[field]
            lighting[field] = choice

        validated["lighting"] = lighting
        return validated

    def _build_environment_details(self, scene_brief: str, period: str,
                                   validated_selections: Dict[str, Any]) -> Dict[str, Any]:
        """Build final environment parameters"""
        place365_name = validated_selections["place365"]["selected"]
        time_of_day = validated_selections["time_of_day"]["selected"]
        weather = validated_selections["weather"]["selected"]
        lighting = validated_selections["lighting"]

        # Determine Indoor/Outdoor
        is_indoor = self._is_indoor(place365_name)

        return {
            "scene_brief": scene_brief,
            "place365_name": place365_name,
            "place365_id": self._get_place365_id(place365_name),
            "time_of_day": time_of_day,
            "weather": weather,
            "period": period,
            "is_indoor": is_indoor,
            "lighting": lighting,
            "atmosphere": self._generate_atmosphere_from_selections(
                scene_brief, time_of_day, weather, lighting
            )
        }

    def _is_indoor(self, place365_name: str) -> Optional[bool]:
        """Indoor or outdoor, read from the Places365 label of the category.

        Returns None for a name that is not a Places365 category.
        """
        indoor = PLACE365_INDOOR.get(place365_name)
        if indoor is None:
            self.logger.warning(
                f"Not a Places365 category, indoor/outdoor unknown: {place365_name}")
        return indoor

    def _generate_atmosphere_from_selections(self, scene_brief: str, time_of_day: str,
                                            weather: Optional[str], lighting: Dict) -> str:
        """Generate atmosphere description based on selections"""
        descriptors = []

        # Based on time
        time_atmosphere = {
            "morning": "fresh and energetic",
            "afternoon": "bright and active",
            "sunset": "warm and peaceful",
            "evening": "calm and relaxed",
            "night": "quiet and mysterious",
            "noon": "vibrant and lively",
            "sunrise": "hopeful and serene"
        }
        descriptors.append(time_atmosphere.get(time_of_day, "pleasant"))

        # Based on weather
        if weather:
            weather_atmosphere = {
                "sunny": "cheerful",
                "cloudy": "subdued",
                "rainy": "moody",
                "snowy": "serene",
                "foggy": "mysterious"
            }
            if weather in weather_atmosphere:
                descriptors.append(weather_atmosphere[weather])

        # Based on lighting quality
        lighting_quality = lighting.get("quality", "")
        if lighting_quality == "dramatic":
            descriptors.append("striking")
        elif lighting_quality == "soft":
            descriptors.append("gentle")

        return " and ".join(descriptors[:3])  # Max 3 descriptors

    def _fallback_environment(self, scene_brief: str, period: str,
                             place365_candidates: List[str]) -> Dict[str, Any]:
        """Fallback when LLM fails.

        No place is chosen: the candidates come from keywords in the brief and
        nothing picks one of them over the others, so the place, its id and
        indoor / outdoor are left empty.
        """
        self.logger.info("Using fallback environment parameters; place left empty "
                         f"({len(place365_candidates)} candidates were not chosen between)")

        # Default values
        # Same key set as the validated path, so the schema stays uniform.
        default_lighting = {
            "type": "mixed",
            "quality": "soft",
            "direction": "front-lit",
            "mood": "neutral",
            "reasoning": "fallback: LLM environment selection unavailable"
        }

        return {
            "scene_brief": scene_brief,
            "place365_name": None,
            "place365_id": None,
            "time_of_day": "afternoon",
            "weather": None,
            "period": period,
            "is_indoor": None,
            "lighting": default_lighting,
            "atmosphere": "balanced and harmonious",
            # Marks the result as fixed defaults, not a designed environment.
            "source": "fallback"
        }

    def _get_place365_id(self, place_name: str) -> Optional[int]:
        """Look up the Places365 category id."""
        place_id_map = {
            "/a/airfield": 0,
            "/a/airplane_cabin": 1,
            "/a/airport_terminal": 2,
            "/a/alcove": 3,
            "/a/alley": 4,
            "/a/amphitheater": 5,
            "/a/amusement_arcade": 6,
            "/a/amusement_park": 7,
            "/a/apartment_building/outdoor": 8,
            "/a/aquarium": 9,
            "/a/aqueduct": 10,
            "/a/arcade": 11,
            "/a/arch": 12,
            "/a/archaelogical_excavation": 13,
            "/a/archive": 14,
            "/a/arena/hockey": 15,
            "/a/arena/performance": 16,
            "/a/arena/rodeo": 17,
            "/a/army_base": 18,
            "/a/art_gallery": 19,
            "/a/art_school": 20,
            "/a/art_studio": 21,
            "/a/artists_loft": 22,
            "/a/assembly_line": 23,
            "/a/athletic_field/outdoor": 24,
            "/a/atrium/public": 25,
            "/a/attic": 26,
            "/a/auditorium": 27,
            "/a/auto_factory": 28,
            "/a/auto_showroom": 29,
            "/b/badlands": 30,
            "/b/bakery/shop": 31,
            "/b/balcony/exterior": 32,
            "/b/balcony/interior": 33,
            "/b/ball_pit": 34,
            "/b/ballroom": 35,
            "/b/bamboo_forest": 36,
            "/b/bank_vault": 37,
            "/b/banquet_hall": 38,
            "/b/bar": 39,
            "/b/barn": 40,
            "/b/barndoor": 41,
            "/b/baseball_field": 42,
            "/b/basement": 43,
            "/b/basketball_court/indoor": 44,
            "/b/bathroom": 45,
            "/b/bazaar/indoor": 46,
            "/b/bazaar/outdoor": 47,
            "/b/beach": 48,
            "/b/beach_house": 49,
            "/b/beauty_salon": 50,
            "/b/bedchamber": 51,
            "/b/bedroom": 52,
            "/b/beer_garden": 53,
            "/b/beer_hall": 54,
            "/b/berth": 55,
            "/b/biology_laboratory": 56,
            "/b/boardwalk": 57,
            "/b/boat_deck": 58,
            "/b/boathouse": 59,
            "/b/bookstore": 60,
            "/b/booth/indoor": 61,
            "/b/botanical_garden": 62,
            "/b/bow_window/indoor": 63,
            "/b/bowling_alley": 64,
            "/b/boxing_ring": 65,
            "/b/bridge": 66,
            "/b/building_facade": 67,
            "/b/bullring": 68,
            "/b/burial_chamber": 69,
            "/b/bus_interior": 70,
            "/b/bus_station/indoor": 71,
            "/b/butchers_shop": 72,
            "/b/butte": 73,
            "/c/cabin/outdoor": 74,
            "/c/cafeteria": 75,
            "/c/campsite": 76,
            "/c/campus": 77,
            "/c/canal/natural": 78,
            "/c/canal/urban": 79,
            "/c/candy_store": 80,
            "/c/canyon": 81,
            "/c/car_interior": 82,
            "/c/carrousel": 83,
            "/c/castle": 84,
            "/c/catacomb": 85,
            "/c/cemetery": 86,
            "/c/chalet": 87,
            "/c/chemistry_lab": 88,
            "/c/childs_room": 89,
            "/c/church/indoor": 90,
            "/c/church/outdoor": 91,
            "/c/classroom": 92,
            "/c/clean_room": 93,
            "/c/cliff": 94,
            "/c/closet": 95,
            "/c/clothing_store": 96,
            "/c/coast": 97,
            "/c/cockpit": 98,
            "/c/coffee_shop": 99,
            "/c/computer_room": 100,
            "/c/conference_center": 101,
            "/c/conference_room": 102,
            "/c/construction_site": 103,
            "/c/corn_field": 104,
            "/c/corral": 105,
            "/c/corridor": 106,
            "/c/cottage": 107,
            "/c/courthouse": 108,
            "/c/courtyard": 109,
            "/c/creek": 110,
            "/c/crevasse": 111,
            "/c/crosswalk": 112,
            "/d/dam": 113,
            "/d/delicatessen": 114,
            "/d/department_store": 115,
            "/d/desert/sand": 116,
            "/d/desert/vegetation": 117,
            "/d/desert_road": 118,
            "/d/diner/outdoor": 119,
            "/d/dining_hall": 120,
            "/d/dining_room": 121,
            "/d/discotheque": 122,
            "/d/doorway/outdoor": 123,
            "/d/dorm_room": 124,
            "/d/downtown": 125,
            "/d/dressing_room": 126,
            "/d/driveway": 127,
            "/d/drugstore": 128,
            "/e/elevator/door": 129,
            "/e/elevator_lobby": 130,
            "/e/elevator_shaft": 131,
            "/e/embassy": 132,
            "/e/engine_room": 133,
            "/e/entrance_hall": 134,
            "/e/escalator/indoor": 135,
            "/e/excavation": 136,
            "/f/fabric_store": 137,
            "/f/farm": 138,
            "/f/fastfood_restaurant": 139,
            "/f/field/cultivated": 140,
            "/f/field/wild": 141,
            "/f/field_road": 142,
            "/f/fire_escape": 143,
            "/f/fire_station": 144,
            "/f/fishpond": 145,
            "/f/flea_market/indoor": 146,
            "/f/florist_shop/indoor": 147,
            "/f/food_court": 148,
            "/f/football_field": 149,
            "/f/forest/broadleaf": 150,
            "/f/forest_path": 151,
            "/f/forest_road": 152,
            "/f/formal_garden": 153,
            "/f/fountain": 154,
            "/g/galley": 155,
            "/g/garage/indoor": 156,
            "/g/garage/outdoor": 157,
            "/g/gas_station": 158,
            "/g/gazebo/exterior": 159,
            "/g/general_store/indoor": 160,
            "/g/general_store/outdoor": 161,
            "/g/gift_shop": 162,
            "/g/glacier": 163,
            "/g/golf_course": 164,
            "/g/greenhouse/indoor": 165,
            "/g/greenhouse/outdoor": 166,
            "/g/grotto": 167,
            "/g/gymnasium/indoor": 168,
            "/h/hangar/indoor": 169,
            "/h/hangar/outdoor": 170,
            "/h/harbor": 171,
            "/h/hardware_store": 172,
            "/h/hayfield": 173,
            "/h/heliport": 174,
            "/h/highway": 175,
            "/h/home_office": 176,
            "/h/home_theater": 177,
            "/h/hospital": 178,
            "/h/hospital_room": 179,
            "/h/hot_spring": 180,
            "/h/hotel/outdoor": 181,
            "/h/hotel_room": 182,
            "/h/house": 183,
            "/h/hunting_lodge/outdoor": 184,
            "/i/ice_cream_parlor": 185,
            "/i/ice_floe": 186,
            "/i/ice_shelf": 187,
            "/i/ice_skating_rink/indoor": 188,
            "/i/ice_skating_rink/outdoor": 189,
            "/i/iceberg": 190,
            "/i/igloo": 191,
            "/i/industrial_area": 192,
            "/i/inn/outdoor": 193,
            "/i/islet": 194,
            "/j/jacuzzi/indoor": 195,
            "/j/jail_cell": 196,
            "/j/japanese_garden": 197,
            "/j/jewelry_shop": 198,
            "/j/junkyard": 199,
            "/k/kasbah": 200,
            "/k/kennel/outdoor": 201,
            "/k/kindergarden_classroom": 202,
            "/k/kitchen": 203,
            "/l/lagoon": 204,
            "/l/lake/natural": 205,
            "/l/landfill": 206,
            "/l/landing_deck": 207,
            "/l/laundromat": 208,
            "/l/lawn": 209,
            "/l/lecture_room": 210,
            "/l/legislative_chamber": 211,
            "/l/library/indoor": 212,
            "/l/library/outdoor": 213,
            "/l/lighthouse": 214,
            "/l/living_room": 215,
            "/l/loading_dock": 216,
            "/l/lobby": 217,
            "/l/lock_chamber": 218,
            "/l/locker_room": 219,
            "/m/mansion": 220,
            "/m/manufactured_home": 221,
            "/m/market/indoor": 222,
            "/m/market/outdoor": 223,
            "/m/marsh": 224,
            "/m/martial_arts_gym": 225,
            "/m/mausoleum": 226,
            "/m/medina": 227,
            "/m/mezzanine": 228,
            "/m/moat/water": 229,
            "/m/mosque/outdoor": 230,
            "/m/motel": 231,
            "/m/mountain": 232,
            "/m/mountain_path": 233,
            "/m/mountain_snowy": 234,
            "/m/movie_theater/indoor": 235,
            "/m/museum/indoor": 236,
            "/m/museum/outdoor": 237,
            "/m/music_studio": 238,
            "/n/natural_history_museum": 239,
            "/n/nursery": 240,
            "/n/nursing_home": 241,
            "/o/oast_house": 242,
            "/o/ocean": 243,
            "/o/office": 244,
            "/o/office_building": 245,
            "/o/office_cubicles": 246,
            "/o/oilrig": 247,
            "/o/operating_room": 248,
            "/o/orchard": 249,
            "/o/orchestra_pit": 250,
            "/p/pagoda": 251,
            "/p/palace": 252,
            "/p/pantry": 253,
            "/p/park": 254,
            "/p/parking_garage/indoor": 255,
            "/p/parking_garage/outdoor": 256,
            "/p/parking_lot": 257,
            "/p/pasture": 258,
            "/p/patio": 259,
            "/p/pavilion": 260,
            "/p/pet_shop": 261,
            "/p/pharmacy": 262,
            "/p/phone_booth": 263,
            "/p/physics_laboratory": 264,
            "/p/picnic_area": 265,
            "/p/pier": 266,
            "/p/pizzeria": 267,
            "/p/playground": 268,
            "/p/playroom": 269,
            "/p/plaza": 270,
            "/p/pond": 271,
            "/p/porch": 272,
            "/p/promenade": 273,
            "/p/pub/indoor": 274,
            "/r/racecourse": 275,
            "/r/raceway": 276,
            "/r/raft": 277,
            "/r/railroad_track": 278,
            "/r/rainforest": 279,
            "/r/reception": 280,
            "/r/recreation_room": 281,
            "/r/repair_shop": 282,
            "/r/residential_neighborhood": 283,
            "/r/restaurant": 284,
            "/r/restaurant_kitchen": 285,
            "/r/restaurant_patio": 286,
            "/r/rice_paddy": 287,
            "/r/river": 288,
            "/r/rock_arch": 289,
            "/r/roof_garden": 290,
            "/r/rope_bridge": 291,
            "/r/ruin": 292,
            "/r/runway": 293,
            "/s/sandbox": 294,
            "/s/sauna": 295,
            "/s/schoolhouse": 296,
            "/s/science_museum": 297,
            "/s/server_room": 298,
            "/s/shed": 299,
            "/s/shoe_shop": 300,
            "/s/shopfront": 301,
            "/s/shopping_mall/indoor": 302,
            "/s/shower": 303,
            "/s/ski_resort": 304,
            "/s/ski_slope": 305,
            "/s/sky": 306,
            "/s/skyscraper": 307,
            "/s/slum": 308,
            "/s/snowfield": 309,
            "/s/soccer_field": 310,
            "/s/stable": 311,
            "/s/stadium/baseball": 312,
            "/s/stadium/football": 313,
            "/s/stadium/soccer": 314,
            "/s/stage/indoor": 315,
            "/s/stage/outdoor": 316,
            "/s/staircase": 317,
            "/s/storage_room": 318,
            "/s/street": 319,
            "/s/subway_station/platform": 320,
            "/s/supermarket": 321,
            "/s/sushi_bar": 322,
            "/s/swamp": 323,
            "/s/swimming_hole": 324,
            "/s/swimming_pool/indoor": 325,
            "/s/swimming_pool/outdoor": 326,
            "/s/synagogue/outdoor": 327,
            "/t/television_room": 328,
            "/t/television_studio": 329,
            "/t/temple/asia": 330,
            "/t/throne_room": 331,
            "/t/ticket_booth": 332,
            "/t/topiary_garden": 333,
            "/t/tower": 334,
            "/t/toyshop": 335,
            "/t/train_interior": 336,
            "/t/train_station/platform": 337,
            "/t/tree_farm": 338,
            "/t/tree_house": 339,
            "/t/trench": 340,
            "/t/tundra": 341,
            "/u/underwater/ocean_deep": 342,
            "/u/utility_room": 343,
            "/v/valley": 344,
            "/v/vegetable_garden": 345,
            "/v/veterinarians_office": 346,
            "/v/viaduct": 347,
            "/v/village": 348,
            "/v/vineyard": 349,
            "/v/volcano": 350,
            "/v/volleyball_court/outdoor": 351,
            "/w/waiting_room": 352,
            "/w/water_park": 353,
            "/w/water_tower": 354,
            "/w/waterfall": 355,
            "/w/watering_hole": 356,
            "/w/wave": 357,
            "/w/wet_bar": 358,
            "/w/wheat_field": 359,
            "/w/wind_farm": 360,
            "/w/windmill": 361,
            "/y/yard": 362,
            "/y/youth_hostel": 363,
            "/z/zen_garden": 364
        }

        # Returns None for a name that is not a Places365 category.
        return place_id_map.get(place_name)

    # Enough families to build a palette from, few enough to keep the second
    # prompt short.
    MIN_COLOUR_FAMILIES = 2
    MAX_COLOUR_FAMILIES = 4

    def _choose_colour_families(self, context: str) -> List[str]:
        """Pass 1 of colour selection: which families to draw the palette from.

        Falls back to every family rather than failing, since narrowing is only
        an optimisation.
        """
        available = colour_vocabulary.groups()
        listing = "\n".join(f"  - {family}" for family in available)

        try:
            answer = self.vllm_wrapper.generate_json_response(
                self.colour_family_prompt,
                f"{context}\n\nCOLOUR FAMILIES:\n{listing}\n"
            )
            chosen = answer.get("families") if isinstance(answer, dict) else None
            if isinstance(chosen, list):
                valid = [f for f in available
                         if f in {str(c).strip() for c in chosen}]
                if self.MIN_COLOUR_FAMILIES <= len(valid) <= self.MAX_COLOUR_FAMILIES:
                    return valid
                if valid:
                    self.logger.warning(
                        f"{len(valid)} colour families chosen, expected "
                        f"{self.MIN_COLOUR_FAMILIES}-{self.MAX_COLOUR_FAMILIES}; "
                        "offering all families instead")
        except Exception as exc:
            self.logger.warning(f"Colour family selection failed: {exc}")

        return available

    def design_color_scheme(self, objects: List[str], style: str,
                          environment_data: Dict[str, Any],
                          enhanced_objects: Optional[List[Dict[str, Any]]] = None,
                          main_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Design the colour scheme from the environment, objects and style.

        `main_prompt` is given only when the image already exists, so the palette
        describes that picture. Falls back to a canned palette after repeated failures.
        """
        self.logger.info("Designing color scheme")

        # Build user prompt
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\n"
        user_prompt += f"Environment: {json.dumps(environment_data)}\n"

        if enhanced_objects:
            # colours_of reads the object's colour list.
            object_colors = {}
            for obj in enhanced_objects:
                object_colors[obj.get("name", "")] = colours_of(obj)
            user_prompt += f"Object Colors: {json.dumps(object_colors)}\n"

        if main_prompt:
            user_prompt += f"The image was painted from this description: {main_prompt}\n"
            user_prompt += "\nDescribe the colour scheme of that painting."
        else:
            user_prompt += "\nCreate a colour scheme that suits the environment, objects and style."

        # Pass 1: choose colour families, narrowing the menu for pass 2.
        families = self._choose_colour_families(user_prompt)
        palette_names = colour_vocabulary.names_in_groups(families)
        self.logger.info(
            f"Colour families {families}: {len(palette_names)} names offered")

        menu = "\n".join(f"  - {name}" for name in palette_names)
        scheme_prompt = f"{user_prompt}\n\nCHOOSE ONLY FROM THESE COLOURS:\n{menu}\n"

        # Pass 2, retried on invalid answers; the canned fallback is a last resort.
        for attempt in range(1, self.COLOUR_SCHEME_ATTEMPTS + 1):
            try:
                result = self.vllm_wrapper.generate_json_response(
                    self.colour_scheme_prompt,
                    scheme_prompt
                )

                if "error" in result:
                    self.logger.warning(
                        f"Colour scheme attempt {attempt} failed: {result}")
                    continue

                if not self._validate_color_scheme(result):
                    self.logger.warning(
                        f"Colour scheme attempt {attempt} did not validate")
                    continue

                # The scheme keeps the answer's fields, not the reply record.
                result = {key: value for key, value in result.items()
                          if key not in REPLY_RECORD_KEYS}
                self._canonicalise_color_scheme(result)
                result["colour_families"] = families
                self.logger.info(
                    f"Colour scheme: {result.get('colour_temperature', 'undefined')}, "
                    f"{len(result.get('main_colours', []))} colours")
                return result

            except Exception as e:
                self.logger.error(f"Colour scheme attempt {attempt} raised: {e}")

        self.logger.warning(
            f"Colour scheme did not validate in {self.COLOUR_SCHEME_ATTEMPTS} "
            "attempts; using the fallback palette")
        fallback = self._create_fallback_color_scheme(style, environment_data)
        # Marks the palette as a canned fallback, not a designed one.
        fallback["source"] = "fallback"
        fallback["attempts"] = self.COLOUR_SCHEME_ATTEMPTS
        return fallback
    
    # Per-style fallback palettes. Every name is in the colour vocabulary and
    # every temperature in its menu, so the fallback passes the same validation.
    FALLBACK_PALETTES = {
        "Baroque": {
            "temperature": "warm",
            "colours": ["Gold", "Crimson Red", "Dark Brown", "Ivory", "Olive"],
            "background": "Dark Brown",
            "light": "Gold", "shadow": "Dark Brown",
        },
        "Impressionism": {
            "temperature": "warm",
            "colours": ["Sky Blue", "Lemon", "Pink", "Lavender", "Ivory"],
            "background": "Sky Blue",
            "light": "Lemon", "shadow": "Lavender",
        },
        "Chinese Ink Painting": {
            "temperature": "cool",
            "colours": ["Black", "Ash Gray", "Silver", "Ivory", "Slate Gray"],
            "background": "Ivory",
            "light": "Ivory", "shadow": "Slate Gray",
        },
        "Post-Impressionism": {
            "temperature": "warm",
            "colours": ["Orange Red", "Cobalt Blue", "Emerald", "Gold", "Crimson Red"],
            "background": "Cobalt Blue",
            "light": "Gold", "shadow": "Cobalt Blue",
        },
        "Neoclassicism": {
            "temperature": "mixed",
            "colours": ["Ivory", "Steel Blue", "Ochre", "Ash Gray", "Crimson Red"],
            "background": "Ivory",
            "light": "Ivory", "shadow": "Steel Blue",
        },
    }

    # Last-resort substitute, also present in the vocabulary.
    FALLBACK_SPARE_COLOUR = "Ash Gray"

    def _create_fallback_color_scheme(self, style: str, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """A canned scheme, used only after every attempt to design one failed."""
        preset = self.FALLBACK_PALETTES.get(style, self.FALLBACK_PALETTES["Neoclassicism"])

        def known(name: str, spare: str) -> str:
            """Keep the fallback valid even if a name is missing from the file."""
            canonical = colour_vocabulary.canonical(name)
            if canonical:
                return canonical
            self.logger.warning(
                f"Fallback palette names '{name}', which is not in the colour "
                f"vocabulary; using '{spare}'")
            return colour_vocabulary.canonical(spare) or spare

        spare = self.FALLBACK_SPARE_COLOUR
        colours = [known(name, spare) for name in preset["colours"]]

        return {
            "main_colours": colours,
            "harmony_type": "analogous",
            "contrast_level": "medium",
            "colour_temperature": preset["temperature"],
            "background_colour": known(preset["background"], spare),
            "lighting_shift": {
                "light": known(preset["light"], spare),
                "shadow": known(preset["shadow"], spare),
            },
        }
    
    # How many times to ask before falling back to the canned palette.
    COLOUR_SCHEME_ATTEMPTS = 3

    # Closed vocabularies the prompt offers; answers outside them are retried.
    CATEGORICAL_FIELDS = {
        "harmony_type": {"complementary", "analogous", "triadic", "monochromatic"},
        "contrast_level": {"high", "medium", "low"},
        "colour_temperature": {"warm", "cool", "mixed"},
    }

    # Five-colour themes, the standard in colour-aesthetics work
    # (O'Donovan et al. 2011; Lin & Hanrahan 2013).
    MAIN_COLOUR_COUNT = 5

    def _validate_color_scheme(self, result: Dict[str, Any]) -> bool:
        """Validate the colour scheme.

        Covers main, background and lighting-shift colours and the harmony fields.
        Every colour must be a vocabulary name; every categorical field must
        answer from its menu.
        """
        colours = result.get("main_colours")
        if not isinstance(colours, list) or len(colours) != self.MAIN_COLOUR_COUNT:
            self.logger.warning(
                f"main_colours must hold {self.MAIN_COLOUR_COUNT} names, got "
                f"{len(colours) if isinstance(colours, list) else type(colours).__name__}")
            return False

        shift = result.get("lighting_shift")
        if not isinstance(shift, dict):
            self.logger.warning("Colour scheme missing lighting_shift")
            return False

        named = list(colours) + [result.get("background_colour"),
                                 shift.get("light"), shift.get("shadow")]
        for name in named:
            if not colour_vocabulary.is_known(name):
                self.logger.warning(
                    f"'{name}' is not a vocabulary colour name")
                return False

        for field, allowed in self.CATEGORICAL_FIELDS.items():
            value = str(result.get(field, "")).strip().lower()
            if value not in allowed:
                self.logger.warning(
                    f"Colour scheme {field} is '{result.get(field)}', "
                    f"not one of {sorted(allowed)}")
                return False

        return True

    def _canonicalise_color_scheme(self, result: Dict[str, Any]) -> None:
        """Store a validated scheme in the vocabulary's own spelling.

        Keeps scene and object colour names consistent with each other.
        """
        result["main_colours"] = [colour_vocabulary.canonical(name)
                                  for name in result["main_colours"]]
        result["background_colour"] = colour_vocabulary.canonical(result["background_colour"])
        shift = result["lighting_shift"]
        for key in ("light", "shadow"):
            shift[key] = colour_vocabulary.canonical(shift[key])
        for field in self.CATEGORICAL_FIELDS:
            result[field] = str(result[field]).strip().lower()
    
    @staticmethod
    def _scheme_tone(color_scheme: Dict[str, Any]) -> str:
        """The scheme's temperature, under whichever key holds it.

        Reads the top-level or nested harmony key; "neutral" if none is set.
        """
        if not isinstance(color_scheme, dict):
            return "neutral"
        temperature = (color_scheme.get("colour_temperature")
                       or color_scheme.get("color_temperature"))
        if not temperature:
            harmony = color_scheme.get("color_harmony")
            if isinstance(harmony, dict):
                temperature = (harmony.get("colour_temperature")
                               or harmony.get("color_temperature"))
        return str(temperature or "neutral")
    
    def analyze_lighting_setup(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Describe the lighting setup: shadows, highlights and overall effect."""
        lighting = environment_data.get("lighting", {})
        
        analysis = {
            "lighting_type": lighting.get("type", "natural"),
            "light_direction": lighting.get("direction", "front-lit"),
            "light_quality": lighting.get("quality", "soft"),
            "mood_impact": lighting.get("mood", "neutral"),
            "shadow_characteristics": self._analyze_shadow_characteristics(lighting),
            "highlight_areas": self._determine_highlight_areas(lighting),
            "overall_lighting_effect": self._describe_lighting_effect(lighting)
        }
        
        return analysis
    
    def _analyze_shadow_characteristics(self, lighting: Dict[str, Any]) -> str:
        """Analyze shadow characteristics"""
        quality = lighting.get("quality", "soft")
        direction = lighting.get("direction", "front-lit")
        
        if quality == "harsh" and direction == "side-lit":
            return "strong, defined shadows with clear edges"
        elif quality == "soft":
            return "gentle, diffused shadows with soft edges"
        elif direction == "back-lit":
            return "dramatic silhouette effects with rim lighting"
        else:
            return "balanced shadows with moderate definition"
    
    def _determine_highlight_areas(self, lighting: Dict[str, Any]) -> str:
        """Determine highlight areas"""
        direction = lighting.get("direction", "front-lit")
        quality = lighting.get("quality", "soft")
        
        if direction == "front-lit":
            return "even highlights across visible surfaces"
        elif direction == "side-lit":
            return "strong highlights on one side, creating volume"
        elif direction == "back-lit":
            return "rim lighting and edge highlights"
        else:
            return "natural highlight distribution"
    
    def _describe_lighting_effect(self, lighting: Dict[str, Any]) -> str:
        """Describe overall lighting effect"""
        light_type = lighting.get("type", "natural")
        quality = lighting.get("quality", "soft")
        mood = lighting.get("mood", "neutral")
        
        effect_desc = f"{quality} {light_type} lighting"
        
        if mood == "warm":
            effect_desc += " creating a cozy, inviting atmosphere"
        elif mood == "cool":
            effect_desc += " producing a calm, serene feeling"
        elif mood == "dramatic":
            effect_desc += " generating strong contrasts and emotional intensity"
        else:
            effect_desc += " providing balanced illumination"
        
        return effect_desc
    
    def get_color_tags(self, color_data: Dict[str, Any]) -> List[str]:
        """Colour tags for prompt generation."""
        tags = []

        # Flat shape: vocabulary names.
        main_colours = color_data.get("main_colours")
        if isinstance(main_colours, list) and main_colours:
            tags.append(f"main colours: {', '.join(str(c) for c in main_colours)}")
            if color_data.get("background_colour"):
                tags.append(f"{color_data['background_colour']} background")
            shift = color_data.get("lighting_shift")
            if isinstance(shift, dict) and shift.get("light") and shift.get("shadow"):
                tags.append(f"{shift['light']} light, {shift['shadow']} shadows")
        else:
            # Nested shape: main_palette with tone/saturation.
            main_palette = color_data.get("main_palette", {})
            if main_palette:
                primary_colors = main_palette.get("primary_colors", [])
                if primary_colors:
                    tags.append(f"primary colors: {', '.join(primary_colors)}")

                overall_tone = main_palette.get("overall_tone", "")
                if overall_tone:
                    tags.append(f"{overall_tone} color tone")

                saturation = main_palette.get("saturation", "")
                brightness = main_palette.get("brightness", "")
                if saturation and brightness:
                    tags.append(f"{saturation} saturation, {brightness} brightness")

        tone = self._scheme_tone(color_data)
        if tone and tone != "neutral":
            tags.append(f"{tone} colour temperature")

        # Colour harmony: top level, or nested under color_harmony.
        harmony_type = color_data.get("harmony_type")
        if not harmony_type:
            harmony = color_data.get("color_harmony")
            harmony_type = harmony.get("harmony_type") if isinstance(harmony, dict) else None
        if harmony_type:
            tags.append(f"{harmony_type} color harmony")

        return [tag for tag in tags if tag.strip()]