import json
import random
from typing import List, Dict, Any, Optional
from arto_kg.conceptualization.utils import setup_logger
from arto_kg.config.model_config import (
    TIME_OF_DAY_OPTIONS,
    WEATHER_OPTIONS,
    PERIOD_OPTIONS,
    COMMON_PLACE365_CATEGORIES,
    SCENE_TIME_MAPPING,
    SCENE_PERIOD_MAPPING
)


class EnvironmentDesigner:
    """Environment Designer"""

    def __init__(self, vllm_wrapper):
        self.logger = setup_logger("environment_designer")
        self.vllm_wrapper = vllm_wrapper

        # Semantic Fallback Categories (8 main categories)
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
                "fallback": ["/s/shop", "/r/restaurant", "/o/office", "/c/cafe", "/b/bar", "/m/market"]
            },
            "cultural_education": {
                "keywords": ["museum", "gallery", "art", "artist", "studio", "theater", "stage",
                           "school", "classroom", "library", "education", "exhibition",
                           "performance", "concert", "lecture", "learning", "creative"],
                "fallback": ["/a/art_studio", "/m/museum", "/a/art_gallery", "/l/library", "/c/classroom", "/t/theater"]
            },
            "outdoor": {
                "keywords": ["outdoor", "outside", "park", "garden", "forest", "beach", "mountain",
                           "field", "nature", "street", "road", "countryside", "path", "trail",
                           "landscape", "natural", "meadow", "wilderness", "wetland", "marsh", "swamp"],
                "fallback": ["/p/park", "/f/forest", "/b/beach", "/s/street", "/f/field", "/g/garden", "/m/marsh"]
            },
            "sports_entertainment": {
                "keywords": ["sports", "gym", "stadium", "court", "playground", "pool", "exercise",
                           "amusement", "arcade", "game", "entertainment", "recreation",
                           "club", "nightclub", "cinema", "theater", "leisure", "play"],
                "fallback": ["/g/gymnasium", "/s/stadium", "/p/playground", "/a/amusement_park", "/g/game_room"]
            },
            "transportation_industrial": {
                "keywords": ["airport", "station", "train", "subway", "bus", "parking", "garage",
                           "factory", "warehouse", "workshop", "industrial", "construction",
                           "junkyard", "storage", "vehicle", "terminal"],
                "fallback": ["/a/airport_terminal", "/t/train_station", "/w/warehouse", "/w/workshop", "/g/garage", "/j/junkyard"]
            },
            "healthcare": {
                "keywords": ["hospital", "clinic", "medical", "doctor", "health", "surgery",
                           "emergency", "patient", "pharmacy", "veterinary", "vet", "treatment"],
                "fallback": ["/h/hospital", "/h/hospital_room", "/c/clinic", "/p/pharmacy", "/v/veterinarians_office"]
            },
            "universal": {
                "keywords": [],  # Fallback category
                "fallback": ["/a/art_studio", "/r/room", "/p/park", "/f/field/wild", "/l/living_room", "/s/street"]
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

        # Color Design System Prompt
        self.color_system_prompt = """
You are an expert color designer for artworks. Based on the environment, objects, and art style, create a comprehensive color scheme.

Your task is to determine:
1. Main color palette for the artwork
2. Background color scheme
3. Object color coordination
4. Overall color harmony and mood

Respond ONLY with a valid JSON object in this format:
{
    "main_palette": {
        "primary_colors": ["color1", "color2", "color3"],
        "secondary_colors": ["color1", "color2"],
        "accent_colors": ["color1"],
        "overall_tone": "warm/cool/neutral",
        "saturation": "high/medium/low",
        "brightness": "bright/medium/dark"
    },
    "background_scheme": {
        "base_color": "main background color",
        "gradient_colors": ["color1", "color2"],
        "texture_colors": ["color1", "color2"],
        "atmospheric_effects": "description of color effects"
    },
    "color_harmony": {
        "harmony_type": "complementary/analogous/triadic/monochromatic",
        "contrast_level": "high/medium/low",
        "color_temperature": "warm/cool/mixed",
        "emotional_impact": "description of emotional effect"
    },
    "lighting_colors": {
        "light_color": "color of the light source",
        "shadow_color": "color of shadows",
        "highlight_color": "color of highlights",
        "ambient_color": "overall ambient color"
    }
}"""
    
    def determine_environment_details(self, scene_brief: str, style: str,
                                     objects: List[str] = None,
                                     object_interpretations: Dict = None) -> Dict[str, Any]:
        """
        Determine detailed environment parameters based on scene description
        New process: Rule-based candidate filtering + LLM intelligent selection

        Args:
            scene_brief: Scene description (e.g., "A modern kitchen featuring oven, with toy train")
            style: Art style
            objects: Object list (optional, for assisting LLM judgment)
            object_interpretations: Object interpretations (optional, from object_selector)

        Returns:
            Complete environment parameters
        """
        self.logger.info(f"Determining environment for: {scene_brief}, style: {style}")

        # Step 1: Determine Period (Rule-based)
        period = self._determine_period(scene_brief, style)
        self.logger.info(f"Period determined by rules: {period}")

        # Step 2: Get Place365 Candidates
        place365_candidates = self._get_place365_candidates(scene_brief)
        self.logger.info(f"Place365 candidates: {len(place365_candidates)} found")

        # Step 3: Let LLM Select Environment Parameters
        try:
            llm_selections = self._llm_select_environment(
                scene_brief, style, period, objects, object_interpretations, place365_candidates
            )

            # Validate Selections
            validated = self._validate_llm_selections(llm_selections, place365_candidates)

            # Build Final Environment Parameters
            return self._build_environment_details(scene_brief, period, validated)

        except Exception as e:
            self.logger.error(f"LLM environment selection failed: {e}, using fallback")
            return self._fallback_environment(scene_brief, period, place365_candidates)

    def _determine_period(self, scene_brief: str, style: str) -> str:
        """Infer period from scene description and style"""
        scene_lower = scene_brief.lower()
        style_lower = style.lower()

        # Find period keywords in scene description
        for key, period in SCENE_PERIOD_MAPPING.items():
            if key in scene_lower:
                return period

        # Infer from style
        # style_period_map = {
        #     "renaissance": "Renaissance",
        #     "baroque": "Renaissance",
        #     "medieval": "Medieval",
        #     "gothic": "Medieval",
        #     "classical": "Classical",
        #     "ancient": "Ancient",
        #     "contemporary": "Contemporary",
        #     "modern": "Modern",
        #     "photorealistic": "Contemporary",
        #     "abstract": "Contemporary",
        #     "impressionism": "Modern", # Added
        #     "realistic": "Modern"     # Added
        # }
        style_period_map = {
            "renaissance": "Early modern",
            "baroque": "Early modern",
            "medieval": "Post-classical",
            "gothic": "Post-classical",
            "classical": "Post-classical",
            "ancient": "Ancient",
            "contemporary": "Contemporary",
            "modern": "Modern",
            "photorealistic": "Contemporary",
            "abstract": "Contemporary"
        }
        
        for key, period in style_period_map.items():
            if key in style_lower:
                return period

        return "Contemporary"  # default

    def _get_place365_candidates(self, scene_brief: str) -> List[str]:
        """
        Get Place365 Candidate List
        Process: Keyword Matching -> Word Overlap Filtering (if too many) -> Semantic Fallback (if no match)
        """
        scene_lower = scene_brief.lower()

        # Step 1: Match keywords from COMMON_PLACE365_CATEGORIES
        candidates = []
        matched_keys = []
        for key, place_list in COMMON_PLACE365_CATEGORIES.items():
            if key != "default" and key in scene_lower:
                candidates.extend(place_list)
                matched_keys.append(key)

        # Deduplicate
        candidates = list(set(candidates))

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
        """
        Word Overlap Filtering
        Count how many scene_brief keywords each candidate path contains, sort by score
        """
        keywords = self._extract_keywords(scene_brief)

        # Score each candidate
        scores = {}
        for candidate in candidates:
            candidate_lower = candidate.lower()
            score = 0
            for keyword in keywords:
                # Exact match: Keyword appears as an independent word
                if f"/{keyword}" in candidate_lower or f"_{keyword}" in candidate_lower:
                    score += 2  # Exact match gets 2 points
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
        """
        Semantic fallback: When no match at all, infer suitable candidates based on semantics
        """
        scene_lower = scene_brief.lower()

        # Iterate through 8 semantic categories
        for category_name, category_data in self.SEMANTIC_CATEGORIES.items():
            if category_name == "universal":
                continue  # Skip fallback category

            # Check if keywords match this category
            keywords = category_data["keywords"]
            for keyword in keywords:
                if keyword in scene_lower:
                    self.logger.info(f"Semantic fallback matched category: {category_name}")
                    return category_data["fallback"]

        # No match, use universal fallback
        self.logger.info("Using universal fallback category")
        return self.SEMANTIC_CATEGORIES["universal"]["fallback"]

    def _llm_select_environment(self, scene_brief: str, style: str, period: str,
                               objects: List[str], object_interpretations: Dict,
                               place365_candidates: List[str]) -> Dict[str, Any]:
        """
        Let LLM select environment parameters from candidates
        """
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

        prompt = f"""SCENE DESCRIPTION:
{scene_brief}

OBJECTS IN SCENE:
{objects_str}

OBJECT DETAILS:
{interp_str}

ART STYLE:
{style}

DETERMINED PERIOD (already decided by rules):
{period}

---

SELECT the best option from each candidate list below:

1. PLACE365 LOCATION (select ONE):
{place365_str}

2. TIME OF DAY (select ONE):
{time_str}

3. WEATHER (select ONE, or null if not visible/relevant):
{weather_str}

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
  "lighting": {{
    "type": "natural/artificial/mixed",
    "quality": "soft/harsh/dramatic/even",
    "direction": "front-lit/back-lit/side-lit",
    "mood": "warm/cool/neutral",
    "reasoning": "Lighting setup based on time, weather, and style"
  }}
}}

IMPORTANT RULES:
- MUST select from provided candidates only
- Consider BOTH scene description AND art style
- Consider the objects and their roles
- Period is already determined: {period}
- For weather: indoor scenes CAN have visible weather through windows
- Provide clear reasoning for each choice
- Return ONLY valid JSON, no additional text
"""
        return prompt

    def _validate_llm_selections(self, llm_response: Dict[str, Any],
                                 place365_candidates: List[str]) -> Dict[str, Any]:
        """Validate if LLM choices are valid"""
        validated = llm_response.copy()

        # Validate place365
        selected_place = llm_response.get("place365", {}).get("selected")
        if selected_place not in place365_candidates:
            self.logger.warning(f"Invalid place365 selection: {selected_place}, using first candidate")
            validated["place365"]["selected"] = place365_candidates[0]

        # Validate time_of_day
        selected_time = llm_response.get("time_of_day", {}).get("selected")
        if selected_time not in TIME_OF_DAY_OPTIONS:
            self.logger.warning(f"Invalid time_of_day selection: {selected_time}, using afternoon")
            validated["time_of_day"]["selected"] = "afternoon"

        # Validate weather
        selected_weather = llm_response.get("weather", {}).get("selected")
        if selected_weather not in WEATHER_OPTIONS and selected_weather is not None:
            self.logger.warning(f"Invalid weather selection: {selected_weather}, using null")
            validated["weather"]["selected"] = None

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

    def _is_indoor(self, place365_name: str) -> bool:
        """Determine if it is an indoor scene"""
        place_lower = place365_name.lower()

        # Explicit outdoor keywords
        outdoor_keywords = [
            "outdoor", "field", "park", "beach", "mountain", "street",
            "road", "forest", "garden", "farm", "desert", "ocean",
            "marsh", "swamp", "wetland", "lake", "river", "pond", "coast",
            "valley", "canyon", "glacier", "volcano", "tundra", "rainforest",
            "plaza", "bridge", "construction", "harbor", "pier", "runway",
            "highway", "driveway", "railroad", "path", "trail"
        ]

        # Explicit indoor keywords
        indoor_keywords = [
            "kitchen", "bedroom", "bathroom", "living_room", "dining_room",
            "office", "classroom", "lobby", "hallway", "closet", "theater",
            "restaurant", "cafe", "bar", "shop", "store", "museum", "library",
            "studio", "gym", "hospital", "hotel", "airport_terminal"
        ]

        # Check outdoor keywords first
        if any(keyword in place_lower for keyword in outdoor_keywords):
            return False

        # Then check indoor keywords
        if any(keyword in place_lower for keyword in indoor_keywords):
            return True

        # Default: Judge by Place365 path
        # If "/indoor" or "/room" exists, consider as indoor
        if "/indoor" in place_lower or "/room" in place_lower:
            return True

        # Otherwise default to indoor (safer default)
        return True

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
        """Fallback when LLM fails"""
        self.logger.info("Using fallback environment parameters")

        # Select first candidate, if none, use default
        place365_name = place365_candidates[0] if place365_candidates else "/i/indoor"

        # Default values
        default_lighting = {
            "type": "mixed",
            "quality": "soft",
            "direction": "front-lit",
            "mood": "neutral"
        }

        is_indoor = self._is_indoor(place365_name)

        return {
            "scene_brief": scene_brief,
            "place365_name": place365_name,
            "place365_id": self._get_place365_id(place365_name),
            "time_of_day": "afternoon",
            "weather": None,
            "period": period,
            "is_indoor": is_indoor,
            "lighting": default_lighting,
            "atmosphere": "balanced and harmonious"
        }

    def _get_place365_id(self, place_name: str) -> int:
        """
        Get Place365 ID
        Lookup from complete Place365 list
        """
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

        return place_id_map.get(place_name, 21)  # default to art_studio

    def design_color_scheme(self, objects: List[str], style: str,
                          environment_data: Dict[str, Any],
                          enhanced_objects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        # Design color scheme
        # New feature: Design color scheme based on environment and objects
        
        Args:
            objects: List of object names
            style: Art style
            environment_data: Environment data
            enhanced_objects: Enhanced object information
            
        Returns:
            Color scheme information
        """
        self.logger.info("Designing color scheme")
        
        # Build user prompt
        user_prompt = f"Objects: {', '.join(objects)}\nArt Style: {style}\n"
        user_prompt += f"Environment: {json.dumps(environment_data)}\n"
        
        if enhanced_objects:
            object_colors = {}
            for obj in enhanced_objects:
                object_colors[obj.get("name", "")] = obj.get("colors", [])
            user_prompt += f"Object Colors: {json.dumps(object_colors)}\n"
        
        user_prompt += "\nCreate a comprehensive color scheme that harmonizes with the environment and objects."
        
        try:
            result = self.vllm_wrapper.generate_json_response(
                self.color_system_prompt,
                user_prompt
            )
            
            if "error" in result:
                self.logger.warning(f"Color scheme design failed: {result}")
                return self._create_fallback_color_scheme(style, environment_data)
            
            # Validate color scheme
            if not self._validate_color_scheme(result):
                self.logger.warning("Invalid color scheme, using fallback")
                return self._create_fallback_color_scheme(style, environment_data)
            
            self.logger.info(f"Color scheme: {result.get('main_palette', {}).get('overall_tone', 'undefined')} tone")
            return result
            
        except Exception as e:
            self.logger.error(f"Color scheme error: {e}")
            return self._create_fallback_color_scheme(style, environment_data)
    
    def create_complete_environment(self, objects: List[str], style: str,
                                  framework_data: Dict[str, Any],
                                  layout_data: Dict[str, Any],
                                  enhanced_objects: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Create complete environment design
        Combine environment details and color scheme
        
        Args:
            objects: List of object names
            style: Art style
            framework_data: Scene framework data
            layout_data: Spatial layout data
            enhanced_objects: Enhanced object information
            
        Returns:
            Complete environment design information
        """
        self.logger.info("Creating complete environment design")
        
        # Step 1: Design environment details
        environment = self.design_environment(objects, style, framework_data, layout_data)
        
        # Step 2: Design color scheme
        color_scheme = self.design_color_scheme(objects, style, environment, enhanced_objects)
        
        # Step 3: Integrate environment information
        complete_environment = {
            "environment_details": environment,
            "color_scheme": color_scheme,
            "environment_summary": self._create_environment_summary(environment, color_scheme),
            "atmospheric_notes": self._generate_atmospheric_notes(environment, color_scheme, style)
        }
        
        return complete_environment

    def _create_fallback_color_scheme(self, style: str, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback color scheme"""
        # Determine base tone based on style
        style_colors = {
            "Renaissance": {"tone": "warm", "primary": ["gold", "deep blue", "rich red"]},
            "Impressionism": {"tone": "warm", "primary": ["light blue", "yellow", "pink"]},
            "Abstract": {"tone": "neutral", "primary": ["bold red", "bright blue", "yellow"]},
            "Chinese Ink Painting": {"tone": "cool", "primary": ["black", "gray", "subtle blue"]},
            "Photorealistic": {"tone": "neutral", "primary": ["natural colors", "realistic tones"]}
        }
        
        style_info = style_colors.get(style, {"tone": "neutral", "primary": ["blue", "brown", "white"]})
        
        return {
            "main_palette": {
                "primary_colors": style_info["primary"],
                "secondary_colors": ["white", "gray"],
                "accent_colors": ["gold"],
                "overall_tone": style_info["tone"],
                "saturation": "medium",
                "brightness": "medium"
            },
            "background_scheme": {
                "base_color": "neutral background",
                "gradient_colors": ["light", "medium"],
                "texture_colors": ["subtle variations"],
                "atmospheric_effects": "soft transitions"
            },
            "color_harmony": {
                "harmony_type": "analogous",
                "contrast_level": "medium",
                "color_temperature": style_info["tone"],
                "emotional_impact": "balanced and pleasing"
            },
            "lighting_colors": {
                "light_color": "warm white",
                "shadow_color": "cool gray",
                "highlight_color": "bright white",
                "ambient_color": "neutral"
            }
        }
    
    def _validate_color_scheme(self, result: Dict[str, Any]) -> bool:
        """Validate validity of color scheme"""
        required_fields = ["main_palette", "color_harmony"]
        return all(field in result for field in required_fields)
    
    def _create_environment_summary(self, environment: Dict[str, Any], 
                                  color_scheme: Dict[str, Any]) -> Dict[str, Any]:
        """Create environment summary"""
        lighting = environment.get("lighting", {})
        main_palette = color_scheme.get("main_palette", {})
        
        return {
            "setting_type": environment.get("specific_location", "undefined"),
            "time_context": f"{environment.get('time_period', 'contemporary')} - {environment.get('time_of_day', 'day')}",
            "weather_condition": environment.get("weather", "clear"),
            "lighting_mood": f"{lighting.get('quality', 'soft')} {lighting.get('mood', 'neutral')} lighting",
            "color_mood": f"{main_palette.get('overall_tone', 'neutral')} tone with {main_palette.get('saturation', 'medium')} saturation",
            "atmosphere": self._determine_atmosphere(environment, color_scheme)
        }
    
    def _generate_atmospheric_notes(self, environment: Dict[str, Any], 
                                  color_scheme: Dict[str, Any], style: str) -> Dict[str, Any]:
        """Generate atmospheric effect notes"""
        notes = {
            "environmental_effects": [],
            "color_considerations": [],
            "style_specific_notes": []
        }
        
        # Environmental effects
        weather = environment.get("weather", "")
        if weather in ["rainy", "stormy"]:
            notes["environmental_effects"].append("Consider water effects and dramatic lighting")
        elif weather == "sunny":
            notes["environmental_effects"].append("Emphasize bright highlights and clear shadows")
        elif weather == "cloudy":
            notes["environmental_effects"].append("Use diffused lighting and soft shadows")
        
        # Color considerations
        harmony = color_scheme.get("color_harmony", {})
        if harmony.get("contrast_level") == "high":
            notes["color_considerations"].append("Maintain strong color contrasts throughout")
        elif harmony.get("contrast_level") == "low":
            notes["color_considerations"].append("Use subtle color variations and smooth transitions")
        
        # Style-specific notes
        if style == "Impressionism":
            notes["style_specific_notes"].append("Focus on capturing light effects and atmospheric conditions")
        elif style == "Baroque":
            notes["style_specific_notes"].append("Emphasize dramatic lighting contrasts and rich colors")
        elif style == "Chinese Ink Painting":
            notes["style_specific_notes"].append("Use minimal color and focus on tonal variations")
        elif style == "Abstract":
            notes["style_specific_notes"].append("Prioritize color relationships over realistic representation")
        
        return notes
    
    def _determine_atmosphere(self, environment: Dict[str, Any], color_scheme: Dict[str, Any]) -> str:
        """Determine overall atmosphere"""
        time_of_day = environment.get("time_of_day", "")
        weather = environment.get("weather", "")
        lighting_mood = environment.get("lighting", {}).get("mood", "neutral")
        color_tone = color_scheme.get("main_palette", {}).get("overall_tone", "neutral")
        
        atmosphere_factors = []
        
        # Time factors
        if time_of_day == "morning":
            atmosphere_factors.append("fresh and hopeful")
        elif time_of_day == "evening":
            atmosphere_factors.append("peaceful and contemplative")
        elif time_of_day == "night":
            atmosphere_factors.append("mysterious and dramatic")
        
        # Weather factors
        if weather == "sunny":
            atmosphere_factors.append("bright and cheerful")
        elif weather in ["rainy", "stormy"]:
            atmosphere_factors.append("dramatic and moody")
        elif weather == "cloudy":
            atmosphere_factors.append("soft and subdued")
        
        # Light and color factors
        if lighting_mood == "warm" and color_tone == "warm":
            atmosphere_factors.append("cozy and inviting")
        elif lighting_mood == "cool" and color_tone == "cool":
            atmosphere_factors.append("calm and serene")
        elif lighting_mood == "dramatic":
            atmosphere_factors.append("intense and striking")
        
        if atmosphere_factors:
            return ", ".join(atmosphere_factors[:3])  # Max 3 descriptors
        else:
            return "balanced and harmonious"
    
    def analyze_lighting_setup(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze lighting settings
        
        Args:
            environment_data: Environment data
            
        Returns:
            Lighting analysis result
        """
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
    
    def get_environment_tags(self, environment_data: Dict[str, Any]) -> List[str]:
        """
        Get environment tags for prompt generation
        
        Args:
            environment_data: Environment data
            
        Returns:
            List of environment tags
        """
        tags = []
        
        # Location tags
        location = environment_data.get("specific_location", "")
        if location:
            tags.append(location)
        
        # Time tags
        time_period = environment_data.get("time_period", "")
        time_of_day = environment_data.get("time_of_day", "")
        if time_period and time_period != "contemporary":
            tags.append(time_period)
        if time_of_day:
            tags.append(f"{time_of_day} lighting")
        
        # Weather tags
        weather = environment_data.get("weather", "")
        if weather and weather != "not applicable":
            tags.append(f"{weather} weather")
        
        # Lighting tags
        lighting = environment_data.get("lighting", {})
        if lighting:
            light_desc = f"{lighting.get('quality', 'soft')} {lighting.get('mood', 'neutral')} lighting"
            tags.append(light_desc)
        
        return [tag for tag in tags if tag.strip()]
    
    def get_color_tags(self, color_data: Dict[str, Any]) -> List[str]:
        """
        Get color tags for prompt generation
        
        Args:
            color_data: Color scheme data
            
        Returns:
            List of color tags
        """
        tags = []
        
        main_palette = color_data.get("main_palette", {})
        if main_palette:
            # Primary colors
            primary_colors = main_palette.get("primary_colors", [])
            if primary_colors:
                tags.append(f"primary colors: {', '.join(primary_colors)}")
            
            # Tone
            overall_tone = main_palette.get("overall_tone", "")
            if overall_tone:
                tags.append(f"{overall_tone} color tone")
            
            # Saturation and Brightness
            saturation = main_palette.get("saturation", "")
            brightness = main_palette.get("brightness", "")
            if saturation and brightness:
                tags.append(f"{saturation} saturation, {brightness} brightness")
        
        # Color harmony
        harmony = color_data.get("color_harmony", {})
        if harmony:
            harmony_type = harmony.get("harmony_type", "")
            if harmony_type:
                tags.append(f"{harmony_type} color harmony")
        
        return [tag for tag in tags if tag.strip()]