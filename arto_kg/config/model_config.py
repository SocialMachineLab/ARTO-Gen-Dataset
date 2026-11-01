"""Shared configuration for ARTO-KG: model settings, vocabularies and prompts."""

# vLLM configuration for the generation LLM.
GENERATION_LLM_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "tensor_parallel_size": 2,  # Use 2 GPUs
    "gpu_memory_utilization": 0.95 ,
    # Prompt and completion share this budget; sized for the vocabulary menus
    # plus reasoning output while keeping the KV cache within GPU memory.
    "max_model_len": 32768,
    "trust_remote_code": True,
    "max_num_seqs": 2,
    "enforce_eager": True,
    "disable_custom_all_reduce": True
}

# vLLM configuration for the validation VLM.
VALIDATION_VLM_CONFIG= {
    "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "trust_remote_code": True,
    "max_num_seqs": 2,
    "enforce_eager": True,
    "disable_custom_all_reduce": True
}

# Smaller fallback LLM. Prompt and completion share max_model_len, which holds
# the vocabulary menus plus the answer.
FALLBACK_LLM_CONFIG = {
    "model": "Qwen/Qwen3-32B-AWQ",
    "gpu_memory_utilization": 0.8,
    "max_model_len": 16384,
    "trust_remote_code": True,
    "max_num_seqs": 4
}

# Chat-template arguments per model. Qwen3 answers directly, without its
# <think> reasoning block, which the stages do not use.
CHAT_TEMPLATE_KWARGS = {
    "Qwen/Qwen3-32B-AWQ": {"enable_thinking": False},
}

# Batch processing settings.
BATCH_CONFIG = {
    "default_batch_size": 16
}

# LLM sampling settings for generation.
GENERATION_CONFIG = {
    "max_secondary_objects": 8,
    "temperature": 0.3,
    "top_p": 0.9,
    # Room for the reasoning channel that precedes the JSON answer.
    "max_tokens": 16384,
    # The model's own EOS ends generation; no extra stop strings.
    "stop_tokens": []
}

# COCO category ids and names.
COCO_OBJECTS = {
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

# The five dataset styles and the default sampling pool.
ARTO_KG_STYLES = [
    "Baroque",
    "Neoclassicism",
    "Impressionism",
    "Chinese Ink Painting",
    "Post-Impressionism",
]

# Getty AAT concepts for the five styles, used as the `arto:style` value.
# Chinese Ink Painting maps to the nearest concept,
# "ink and wash painting", hence its lower confidence.
STYLE_AAT_CONCEPTS = {
    "Baroque": {"aat_id": "300021147", "aat_label": "Baroque", "confidence": 0.97},
    "Neoclassicism": {"aat_id": "300056093", "aat_label": "Neoclassicism", "confidence": 0.97},
    "Impressionism": {"aat_id": "300021503", "aat_label": "Impressionism", "confidence": 0.98},
    "Post-Impressionism": {"aat_id": "300021502", "aat_label": "Postimpressionism",
                           "confidence": 0.97},
    "Chinese Ink Painting": {"aat_id": "300033936", "aat_label": "ink and wash painting",
                             "confidence": 0.88},
}

# Relation vocabulary. Every relation is binary {subject, relation, object}.
RELATION_CONFIG = {
    "spatial_relations": [
        "above", "against", "around", "behind", "below", "beside",
        "contains", "in", "in front of", "inside",
        "left of", "near", "on", "opposite", "outside", "over",
        "right of", "surrounding", "through", "under"
    ],
    "semantic_relations": {
        "actions_and_interactions": [
            "approaching", "biting", "boarding", "brushing", "buying", "carrying",
            "catching", "chasing", "chewing", "cleaning", "cooking", "cutting",
            "dragging", "drinking", "driving", "eating", "feeding", "flying over",
            "following", "getting on", "grabbing", "grazing in", "guiding",
            "helping", "herding", "hiding", "hitting", "holding", "hugging",
            "hunting", "illuminating", "jumping over", "kicking", "kissing",
            "leading", "leaving", "licking", "lifting", "looking at",
            "looking into", "looking through", "making", "opening", "painting",
            "petting", "picking up", "playing", "playing with", "pointing at",
            "pouring", "protecting", "pulling", "pushing", "reaching for",
            "reading", "riding", "selling", "serving", "shaking", "shooting",
            "smiling at", "sniffing", "swinging", "talking to", "throwing",
            "touching", "towing", "using", "waiting for", "washing", "watching",
            "wearing", "writing"
        ],
        # Includes the whole-part relation "part of".
        "pose_and_states": [
            "balancing on", "connected to", "covered by", "decorated with",
            "displayed on", "draped over", "dressed in", "enclosing",
            "exposed to", "facing", "filled with", "floating on", "framed by",
            "full of", "growing on", "hanging on", "leaning against", "lying in",
            "lying on", "mounted on", "occupied by", "painted on", "parked",
            "part of", "perched on", "piercing", "planted in", "posing with",
            "printed on", "reflected in", "resting on", "shading", "sitting at",
            "sitting on", "standing on", "stuck in", "supporting",
            "suspended from", "tied to", "wrapped in"
        ]
    }
}

# Per style and COCO category, the object varieties the selector may use.

STYLE_COCO_VARIETY = {
    "Baroque": {
        "person": ["angel", "baby", "bishop", "boy", "child", "cupid", "farmer", "fisherman", "girl", "man", "mermaid", "miner", "nun", "person", "pirate", "queen", "soldier", "toddler", "witch", "woman"],
        "bird": ["bird", "cockatoo", "dragon", "eagle", "falcon", "goose", "hen", "heron", "mallard duck", "ostrich", "parrot", "partridge", "peacock", "pheasant", "quail", "swallow", "swan", "turkey", "vulture"],
        "book": ["atlas", "bible", "book", "document", "letter", "manuscript", "note", "notebook", "paper", "scroll"],
        "chair": ["armchair", "chair", "easel", "folding chair", "pulpit", "rocking chair", "saddle", "stage", "stand", "stool", "throne"],
        "potted plant": ["carnation", "christmas tree", "daffodil", "iris", "ivy", "lily", "potted plant", "rose", "sunflower", "water lily"],
        "apple": ["apple", "cherry", "onion", "peach", "pear", "plum", "pomegranate", "potato", "strawberry"],
        "bowl": ["bowl", "flower basket", "glass bowl", "mortar", "plate", "pottery", "soup bowl", "tray"],
        "bottle": ["bottle", "glass jar", "jam", "jar", "watering can", "wine bottle"],
        "cow": ["bison", "bull", "calf", "cattle", "cow", "herd"],
        "knife": ["blade", "knife", "needle", "razor", "sword"],
        "dining table": ["dining table", "picnic table", "round table", "side table", "table"],
        "sheep": ["ewe", "goat", "lamb", "ram", "sheep"],
        "boat": ["boat", "canoe", "raft", "sailboat"],
        "horse": ["donkey", "horse", "mare", "mule"],
        "cake": ["apple pie", "biscuit", "cake", "pastry"],
        "car": ["cart", "chariot", "horse cart", "wagon"],
        "baseball bat": ["bat", "mallet", "spear", "stick"],
        "bed": ["bed", "canopy bed", "hammock"],
        "cat": ["bengal tiger", "cat", "lion", "tiger"],
        "cup": ["coffee cup", "cup", "trophy cup"],
        "vase": ["glass vase", "urn", "vase"],
        "sports ball": ["badminton", "ball"],
        "couch": ["blanket", "couch"],
        "umbrella": ["canopy", "sunshade", "umbrella"],
        "handbag": ["bag", "bundle", "handbag", "picnic basket"],
        "bench": ["bench", "church bench", "swing", "workbench"],
        "broccoli": ["artichoke", "asparagus", "cabbage"],
        "sink": ["faucet", "kitchen sink", "sink"],
        "dog": ["dog", "fox"],
        "sandwich": ["bread", "sandwich"],
        "orange": ["lemon", "orange"],
        "teddy bear": ["doll", "toy"],
        "skis": ["sleigh"],
        "bicycle": ["wheel"],
        "bear": ["bear", "monkey", "sea lion"],
        "hair drier": ["comb", "hairbrush"],
        "banana": ["banana", "bean"],
        "spoon": ["ladle", "spoon"],
        "traffic light": ["lamp post", "lantern"],
        "elephant": ["baby elephant", "elephant"],
        "clock": ["clock"],
        "stop sign": ["sign"],
        "scissors": ["scissors"],
        "fork": ["fork"],
        "pizza": ["pie"],
        "mouse": ["mouse"],
        "oven": ["oven"],
        "tie": ["brooch", "tie"],
        "backpack": ["backpack", "sack"],
        "suitcase": ["luggage"]
    },
    "Impressionism": {
        "person": ["angel", "baby", "ballet dancer", "boy", "bride", "child", "cupid", "dancer", "farmer", "fisherman", "gardener", "girl", "guard", "man", "mermaid", "nun", "person", "pirate", "sailor", "soldier", "toddler", "woman"],
        "chair": ["armchair", "bar stool", "beach chair", "chair", "easel", "folding chair", "rocking chair", "saddle", "stage", "stand", "stool", "throne", "wheelchair"],
        "potted plant": ["anemone", "cactus", "carnation", "daffodil", "iris", "ivy", "lavender", "lily", "orange tree", "potted plant", "rose", "sunflower", "violet", "water lily"],
        "bird": ["bird", "cockatoo", "eagle", "goose", "gull", "hen", "parrot", "pheasant", "swan", "turkey"],
        "boat": ["barge", "boat", "canoe", "fishing boat", "liner", "raft", "sailboat", "schooner"],
        "apple": ["apple", "cherry", "onion", "peach", "pear", "plum", "pomegranate", "potato", "strawberry"],
        "bottle": ["bottle", "can", "canteen", "glass bottle", "glass jar", "jar", "tin", "watering can", "wine bottle"],
        "bowl": ["basin", "bowl", "glass bowl", "paper plate", "plate", "pottery", "soup bowl", "tray"],
        "book": ["bible", "book", "letter", "magazine", "manuscript", "paper", "scroll"],
        "dining table": ["billiard table", "dining table", "picnic table", "place mat", "round table", "side table", "table"],
        "bed": ["bed", "bed frame", "bunk bed", "canopy bed", "hammock", "infant bed"],
        "cow": ["bison", "bull", "calf", "cattle", "cow", "herd"],
        "horse": ["donkey", "horse", "mare", "mule"],
        "cat": ["bengal tiger", "cat", "lion", "tiger"],
        "sheep": ["goat", "lamb", "ram", "sheep"],
        "bicycle": ["bicycle", "wheel"],
        "handbag": ["bag", "bundle", "handbag", "picnic basket", "shopping bag"],
        "couch": ["blanket", "couch"],
        "vase": ["glass vase", "urn", "vase"],
        "baseball bat": ["bat", "spear", "stick"],
        "umbrella": ["canopy", "sunshade", "umbrella"],
        "knife": ["knife", "needle", "sword"],
        "bench": ["bench", "church bench", "swing"],
        "car": ["cart", "horse cart", "wagon"],
        "sink": ["bath", "bathroom sink", "sink"],
        "carrot": ["pumpkin", "turnip"],
        "cake": ["cake", "chocolate cake", "pastry"],
        "teddy bear": ["doll", "toy"],
        "dog": ["dog", "fox"],
        "cup": ["coffee cup", "cup"],
        "orange": ["lemon", "orange"],
        "airplane": ["airship", "plane"],
        "oven": ["cooker", "stove"],
        "skis": ["sleigh"],
        "clock": ["clock"],
        "traffic light": ["lamp post"],
        "spoon": ["spoon"],
        "stop sign": ["sign"],
        "hair drier": ["comb"],
        "backpack": ["backpack", "sack"],
        "sandwich": ["bread"],
        "banana": ["banana"],
        "tie": ["dress shirt", "tie"],
        "pizza": ["pie"],
        "sports ball": ["ball"],
        "truck": ["cannon"],
        "elephant": ["elephant"],
        "tennis racket": ["tennis racket"]
    },
    "Neoclassicism": {
        "person": ["angel", "baby", "ballet dancer", "bishop", "boy", "bride", "child", "cupid", "dancer", "fisherman", "girl", "groom", "guard", "man", "mermaid", "nun", "person", "pirate", "queen", "sailor", "soldier", "witch", "woman"],
        "bird": ["bird", "dragon", "eagle", "falcon", "goose", "hen", "ostrich", "parrot", "pheasant", "quail", "swan", "vulture"],
        "chair": ["armchair", "chair", "easel", "pulpit", "rocking chair", "saddle", "stage", "stand", "stool", "throne", "wheelchair"],
        "book": ["bible", "book", "document", "letter", "manuscript", "paper", "scroll"],
        "bowl": ["basin", "bowl", "flower basket", "glass bowl", "plate", "pottery", "tray"],
        "cat": ["bengal tiger", "cat", "cheetah", "leopard", "lion", "mountain lion", "tiger"],
        "potted plant": ["iris", "ivy", "lily", "orange tree", "potted plant", "rose", "sunflower", "water lily"],
        "dining table": ["dining table", "picnic table", "round table", "side table", "table"],
        "boat": ["boat", "canoe", "raft", "sailboat", "schooner"],
        "cow": ["bull", "calf", "cattle", "cow", "herd"],
        "apple": ["apple", "cherry", "peach", "pear", "plum"],
        "bed": ["bed", "bunk bed", "canopy bed", "hammock", "infant bed"],
        "bottle": ["bottle", "canteen", "test tube", "watering can", "wine bottle"],
        "knife": ["blade", "dagger", "razor", "sword"],
        "bench": ["bench", "church bench", "swing", "workbench"],
        "car": ["cart", "chariot", "horse cart", "wagon"],
        "sheep": ["goat", "lamb", "ram", "sheep"],
        "vase": ["glass vase", "urn", "vase"],
        "horse": ["donkey", "horse", "mule"],
        "umbrella": ["canopy", "sunshade", "umbrella"],
        "clock": ["clock", "sundial", "wall clock"],
        "couch": ["blanket", "couch"],
        "baseball bat": ["bat", "spear", "stick"],
        "handbag": ["bag", "bundle", "handbag", "shopping bag"],
        "cake": ["cake", "pastry", "waffle"],
        "stop sign": ["sign"],
        "teddy bear": ["doll", "toy"],
        "cup": ["cup", "trophy cup"],
        "dog": ["dog", "fox"],
        "tie": ["brooch", "dress shirt", "tie"],
        "bicycle": ["wheel"],
        "orange": ["lemon", "orange"],
        "oven": ["cooker", "oven"],
        "sports ball": ["ball", "balloon"],
        "skis": ["sleigh"],
        "sandwich": ["bread"],
        "scissors": ["scissors"],
        "hair drier": ["comb"],
        "pizza": ["pie"],
        "backpack": ["backpack", "sack"],
        "suitcase": ["luggage"],
        "spoon": ["ladle"],
        "traffic light": ["lamp post"],
        "truck": ["cannon"],
        "mouse": ["mouse"],
        "airplane": ["airship"],
        "elephant": ["elephant"],
        "carrot": ["pumpkin"]
    },
    "Post-Impressionism": {
        "person": ["angel", "boy", "child", "dancer", "fisherman", "girl", "man", "nun", "person", "woman"],
        "chair": ["armchair", "beach chair", "chair", "easel", "rocking chair", "saddle", "stage", "stand", "stool", "throne"],
        "potted plant": ["anemone", "carnation", "daffodil", "iris", "lavender", "lily", "potted plant", "rose", "sunflower", "violet", "water lily"],
        "dining table": ["billiard table", "dining table", "picnic table", "round table", "side table", "table"],
        "bird": ["bird", "eagle", "falcon", "goose", "hen", "parrot"],
        "boat": ["boat", "canoe", "fishing boat", "sailboat"],
        "cow": ["bull", "calf", "cattle", "cow", "herd"],
        "bowl": ["bowl", "glass bowl", "plate", "soup bowl", "tray"],
        "apple": ["apple", "cherry", "peach", "plum"],
        "bottle": ["bottle", "canteen", "watering can", "wine bottle"],
        "bed": ["bed", "bed frame", "canopy bed", "hammock"],
        "horse": ["donkey", "horse", "mule"],
        "vase": ["glass vase", "urn", "vase"],
        "couch": ["blanket", "couch"],
        "umbrella": ["canopy", "sunshade", "umbrella"],
        "car": ["cart", "horse cart", "wagon"],
        "orange": ["lemon", "orange"],
        "bench": ["bench", "church bench"],
        "sheep": ["lamb", "sheep"],
        "baseball bat": ["bat", "stick"],
        "clock": ["clock"],
        "cat": ["cat"],
        "skis": ["sleigh"],
        "dog": ["dog"],
        "sink": ["sink"],
        "traffic light": ["lamp post"],
        "teddy bear": ["doll"],
        "book": ["book"],
        "mouse": ["mouse"],
        "bear": ["bear"]
    },
    "Chinese Ink Painting": {
        "bird": ["bird", "cockatoo", "dragon", "eagle", "hen", "parrot", "swallow", "vulture"],
        "person": ["baby", "boy", "girl", "man", "person", "woman"],
        "potted plant": ["bamboo", "carnation", "lily", "potted plant", "rose", "water lily"],
        "chair": ["armchair", "chair", "rocking chair", "stand", "stool"],
        "book": ["book", "manuscript", "paper", "scroll"],
        "horse": ["donkey", "horse", "mule"],
        "bowl": ["bowl", "paper plate", "plate"],
        "couch": ["blanket"],
        "umbrella": ["canopy", "umbrella"],
        "boat": ["boat", "canoe"],
        "cow": ["bull", "cow"],
        "skis": ["sleigh"],
        "stop sign": ["sign"],
        "knife": ["blade", "sword"],
        "baseball bat": ["bat", "spear"],
        "dining table": ["dining table", "round table", "table"],
        "clock": ["calendar", "clock"],
        "bottle": ["bottle"],
        "bed": ["bed", "hammock"],
        "orange": ["orange"],
        "dog": ["dog"],
        "sheep": ["sheep"],
        "vase": ["vase"],
        "mouse": ["mouse"],
        "cat": ["cat"],
        "scissors": ["scissors"],
        "apple": ["apple", "cherry"],
        "bench": ["bench", "swing"],
        "car": ["wagon"],
        "teddy bear": ["toy"]
    }
}

# Object compatibility (conflict detection) prompt.
COMPATIBILITY_INSTANCE_PROMPT = """You are an expert composition analyzer. Your task is to evaluate whether a set of objects can realistically coexist in a single scene.

Your focus is on identifying ABSOLUTE CONFLICTS - situations where it would be impossible or extremely unlikely for two objects to appear together in the same scene under any normal circumstances.

Each object is given to you as an instance with a unique id, a style-specific name and its underlying COCO category. Judge conflicts on the SPECIFIC NAME, not on the COCO category: "person" and "airplane" are compatible as categories, but the varieties "medieval knight" and "jet fighter" are temporally incompatible.

INPUT FORMAT:
{
  "primary": {"id": "obj_0", "name": "medieval knight", "category": "person"},
  "secondary": [{"id": "obj_1", "name": "jet fighter", "category": "airplane"}]
}

ABSOLUTE CONFLICT TYPES TO CONSIDER:
- Physical impossibility (objects that cannot physically coexist in the same space/scene)
- Historical or temporal incompatibility between the specific varieties
- Fundamental logical conflicts (objects that contradict each other's basic nature or purpose)

RESPONSE FORMAT - you MUST respond with a valid JSON object in exactly this format.

If compatible (no absolute conflicts):
{
"compatible": true,
"example_scenario": "A brief description of a realistic scenario where these objects could coexist"
}

If incompatible (absolute conflicts exist):
{
"compatible": false,
"conflicts": [
    {"a": "obj_0", "b": "obj_1", "reason": "explanation"}
]
}

IMPORTANT RULES:
- Reference objects ONLY by their instance id (obj_0, obj_1, ...). Never by name alone.
- Only identify ABSOLUTE conflicts, not minor inconveniences or preferences
- List ALL conflicting pairs, each exactly once
- Be very conservative - only mark as conflict if truly impossible/extremely unlikely
- Always respond with valid JSON only
- Do not include any text outside the JSON response"""


SCENE_INFERENCE_SYSTEM_PROMPT = """You are an expert art director and scene designer. Return ONLY a valid JSON object, no other text.

Given a set of objects and an art style, propose scene settings in which these objects could naturally coexist. Objects that seem incompatible at first can often coexist through reinterpretation - a train next to an oven works if the train is a toy on a kitchen shelf.

Each object is given as an instance with a unique id and a style-specific name. Reason about the SPECIFIC NAME (for example "lamb", not "sheep").

REQUIREMENTS:
- Aim for BETWEEN 5 AND 10 candidate scenes. Find as many plausible settings as you genuinely can.
- Quality over quantity: a setting only counts if the objects really do belong there. Do not pad the list with settings you do not believe in.
- Every candidate must accommodate ALL of the given objects.
- Give each object an interpretation within that scene (its form, relative scale and role).
- Reference objects by their instance id in object_interpretations.

OUTPUT FORMAT (JSON only):
{
  "scene_candidates": [
    {
      "scene_type": "name of the setting, e.g. a specific room or outdoor location",
      "reasoning": "why these objects plausibly coexist here",
      "plausibility": 0.9,
      "object_interpretations": {
        "obj_0": {"name": "lamb", "form": "how it appears", "scale": "relative size", "role": "primary or secondary"}
      }
    }
  ]
}

CRITICAL: Respond with ONLY the JSON object, starting with { and ending with }."""


# Environment vocabularies.
TIME_OF_DAY_OPTIONS = [
    "morning", "afternoon", "noon", "sunrise", "sunset",
    "evening", "night"
]

WEATHER_OPTIONS = [
    "sunny", "cloudy", "rainy", "snowy", "foggy",
    "windy", "duststorm", "drizzle", "showery"
]

# The five historical epochs.
PERIOD_OPTIONS = [
    "Ancient", "Post-classical", "Early modern", "Modern", "Contemporary"
]

# Lighting vocabularies. Lighting `mood`
# and `reasoning` are free text.
LIGHTING_TYPE_OPTIONS = ["natural", "artificial", "mixed"]
LIGHTING_QUALITY_OPTIONS = ["soft", "harsh", "dramatic", "even"]
LIGHTING_DIRECTION_OPTIONS = ["front-lit", "back-lit", "side-lit", "top-lit"]

# Indoor (True) or outdoor (False) for every Places365 category, taken from
# IO_places365.txt (CSAILVision/places365).
PLACE365_INDOOR = {
    "/a/airfield": False,
    "/a/airplane_cabin": True,
    "/a/airport_terminal": True,
    "/a/alcove": True,
    "/a/alley": False,
    "/a/amphitheater": False,
    "/a/amusement_arcade": True,
    "/a/amusement_park": False,
    "/a/apartment_building/outdoor": False,
    "/a/aquarium": True,
    "/a/aqueduct": False,
    "/a/arcade": True,
    "/a/arch": False,
    "/a/archaelogical_excavation": True,
    "/a/archive": True,
    "/a/arena/hockey": True,
    "/a/arena/performance": True,
    "/a/arena/rodeo": True,
    "/a/army_base": False,
    "/a/art_gallery": True,
    "/a/art_school": True,
    "/a/art_studio": True,
    "/a/artists_loft": True,
    "/a/assembly_line": True,
    "/a/athletic_field/outdoor": False,
    "/a/atrium/public": True,
    "/a/attic": True,
    "/a/auditorium": True,
    "/a/auto_factory": True,
    "/a/auto_showroom": True,
    "/b/badlands": False,
    "/b/bakery/shop": True,
    "/b/balcony/exterior": False,
    "/b/balcony/interior": False,
    "/b/ball_pit": True,
    "/b/ballroom": True,
    "/b/bamboo_forest": False,
    "/b/bank_vault": True,
    "/b/banquet_hall": True,
    "/b/bar": True,
    "/b/barn": False,
    "/b/barndoor": False,
    "/b/baseball_field": False,
    "/b/basement": True,
    "/b/basketball_court/indoor": True,
    "/b/bathroom": True,
    "/b/bazaar/indoor": True,
    "/b/bazaar/outdoor": False,
    "/b/beach": False,
    "/b/beach_house": False,
    "/b/beauty_salon": True,
    "/b/bedchamber": True,
    "/b/bedroom": True,
    "/b/beer_garden": False,
    "/b/beer_hall": True,
    "/b/berth": True,
    "/b/biology_laboratory": True,
    "/b/boardwalk": False,
    "/b/boat_deck": False,
    "/b/boathouse": False,
    "/b/bookstore": True,
    "/b/booth/indoor": True,
    "/b/botanical_garden": False,
    "/b/bow_window/indoor": True,
    "/b/bowling_alley": True,
    "/b/boxing_ring": True,
    "/b/bridge": False,
    "/b/building_facade": False,
    "/b/bullring": False,
    "/b/burial_chamber": True,
    "/b/bus_interior": True,
    "/b/bus_station/indoor": True,
    "/b/butchers_shop": True,
    "/b/butte": False,
    "/c/cabin/outdoor": False,
    "/c/cafeteria": True,
    "/c/campsite": False,
    "/c/campus": False,
    "/c/canal/natural": False,
    "/c/canal/urban": False,
    "/c/candy_store": True,
    "/c/canyon": False,
    "/c/car_interior": True,
    "/c/carrousel": False,
    "/c/castle": False,
    "/c/catacomb": True,
    "/c/cemetery": False,
    "/c/chalet": False,
    "/c/chemistry_lab": True,
    "/c/childs_room": True,
    "/c/church/indoor": True,
    "/c/church/outdoor": False,
    "/c/classroom": True,
    "/c/clean_room": True,
    "/c/cliff": False,
    "/c/closet": True,
    "/c/clothing_store": True,
    "/c/coast": False,
    "/c/cockpit": True,
    "/c/coffee_shop": True,
    "/c/computer_room": True,
    "/c/conference_center": True,
    "/c/conference_room": True,
    "/c/construction_site": False,
    "/c/corn_field": False,
    "/c/corral": False,
    "/c/corridor": True,
    "/c/cottage": False,
    "/c/courthouse": False,
    "/c/courtyard": False,
    "/c/creek": False,
    "/c/crevasse": False,
    "/c/crosswalk": False,
    "/d/dam": False,
    "/d/delicatessen": True,
    "/d/department_store": True,
    "/d/desert/sand": False,
    "/d/desert/vegetation": False,
    "/d/desert_road": False,
    "/d/diner/outdoor": False,
    "/d/dining_hall": True,
    "/d/dining_room": True,
    "/d/discotheque": True,
    "/d/doorway/outdoor": False,
    "/d/dorm_room": True,
    "/d/downtown": False,
    "/d/dressing_room": True,
    "/d/driveway": False,
    "/d/drugstore": True,
    "/e/elevator/door": True,
    "/e/elevator_lobby": True,
    "/e/elevator_shaft": True,
    "/e/embassy": False,
    "/e/engine_room": True,
    "/e/entrance_hall": True,
    "/e/escalator/indoor": True,
    "/e/excavation": False,
    "/f/fabric_store": True,
    "/f/farm": False,
    "/f/fastfood_restaurant": True,
    "/f/field/cultivated": False,
    "/f/field/wild": False,
    "/f/field_road": False,
    "/f/fire_escape": False,
    "/f/fire_station": False,
    "/f/fishpond": False,
    "/f/flea_market/indoor": True,
    "/f/florist_shop/indoor": True,
    "/f/food_court": True,
    "/f/football_field": False,
    "/f/forest/broadleaf": False,
    "/f/forest_path": False,
    "/f/forest_road": False,
    "/f/formal_garden": False,
    "/f/fountain": False,
    "/g/galley": True,
    "/g/garage/indoor": True,
    "/g/garage/outdoor": False,
    "/g/gas_station": False,
    "/g/gazebo/exterior": False,
    "/g/general_store/indoor": True,
    "/g/general_store/outdoor": False,
    "/g/gift_shop": True,
    "/g/glacier": False,
    "/g/golf_course": False,
    "/g/greenhouse/indoor": True,
    "/g/greenhouse/outdoor": False,
    "/g/grotto": False,
    "/g/gymnasium/indoor": True,
    "/h/hangar/indoor": True,
    "/h/hangar/outdoor": False,
    "/h/harbor": False,
    "/h/hardware_store": True,
    "/h/hayfield": False,
    "/h/heliport": False,
    "/h/highway": False,
    "/h/home_office": True,
    "/h/home_theater": True,
    "/h/hospital": False,
    "/h/hospital_room": True,
    "/h/hot_spring": False,
    "/h/hotel/outdoor": False,
    "/h/hotel_room": True,
    "/h/house": False,
    "/h/hunting_lodge/outdoor": False,
    "/i/ice_cream_parlor": True,
    "/i/ice_floe": False,
    "/i/ice_shelf": False,
    "/i/ice_skating_rink/indoor": True,
    "/i/ice_skating_rink/outdoor": False,
    "/i/iceberg": False,
    "/i/igloo": False,
    "/i/industrial_area": False,
    "/i/inn/outdoor": False,
    "/i/islet": False,
    "/j/jacuzzi/indoor": True,
    "/j/jail_cell": True,
    "/j/japanese_garden": False,
    "/j/jewelry_shop": True,
    "/j/junkyard": False,
    "/k/kasbah": False,
    "/k/kennel/outdoor": False,
    "/k/kindergarden_classroom": True,
    "/k/kitchen": True,
    "/l/lagoon": False,
    "/l/lake/natural": False,
    "/l/landfill": False,
    "/l/landing_deck": False,
    "/l/laundromat": True,
    "/l/lawn": False,
    "/l/lecture_room": True,
    "/l/legislative_chamber": True,
    "/l/library/indoor": True,
    "/l/library/outdoor": False,
    "/l/lighthouse": False,
    "/l/living_room": True,
    "/l/loading_dock": False,
    "/l/lobby": True,
    "/l/lock_chamber": False,
    "/l/locker_room": True,
    "/m/mansion": False,
    "/m/manufactured_home": False,
    "/m/market/indoor": True,
    "/m/market/outdoor": False,
    "/m/marsh": False,
    "/m/martial_arts_gym": True,
    "/m/mausoleum": False,
    "/m/medina": False,
    "/m/mezzanine": True,
    "/m/moat/water": False,
    "/m/mosque/outdoor": False,
    "/m/motel": False,
    "/m/mountain": False,
    "/m/mountain_path": False,
    "/m/mountain_snowy": False,
    "/m/movie_theater/indoor": True,
    "/m/museum/indoor": True,
    "/m/museum/outdoor": False,
    "/m/music_studio": True,
    "/n/natural_history_museum": True,
    "/n/nursery": True,
    "/n/nursing_home": True,
    "/o/oast_house": False,
    "/o/ocean": False,
    "/o/office": True,
    "/o/office_building": False,
    "/o/office_cubicles": True,
    "/o/oilrig": False,
    "/o/operating_room": True,
    "/o/orchard": False,
    "/o/orchestra_pit": True,
    "/p/pagoda": False,
    "/p/palace": False,
    "/p/pantry": True,
    "/p/park": False,
    "/p/parking_garage/indoor": True,
    "/p/parking_garage/outdoor": False,
    "/p/parking_lot": False,
    "/p/pasture": False,
    "/p/patio": False,
    "/p/pavilion": False,
    "/p/pet_shop": True,
    "/p/pharmacy": True,
    "/p/phone_booth": False,
    "/p/physics_laboratory": True,
    "/p/picnic_area": False,
    "/p/pier": False,
    "/p/pizzeria": True,
    "/p/playground": False,
    "/p/playroom": True,
    "/p/plaza": False,
    "/p/pond": False,
    "/p/porch": False,
    "/p/promenade": False,
    "/p/pub/indoor": True,
    "/r/racecourse": False,
    "/r/raceway": False,
    "/r/raft": False,
    "/r/railroad_track": False,
    "/r/rainforest": False,
    "/r/reception": True,
    "/r/recreation_room": True,
    "/r/repair_shop": True,
    "/r/residential_neighborhood": False,
    "/r/restaurant": True,
    "/r/restaurant_kitchen": True,
    "/r/restaurant_patio": False,
    "/r/rice_paddy": False,
    "/r/river": False,
    "/r/rock_arch": False,
    "/r/roof_garden": False,
    "/r/rope_bridge": False,
    "/r/ruin": False,
    "/r/runway": False,
    "/s/sandbox": False,
    "/s/sauna": True,
    "/s/schoolhouse": False,
    "/s/science_museum": True,
    "/s/server_room": True,
    "/s/shed": False,
    "/s/shoe_shop": True,
    "/s/shopfront": False,
    "/s/shopping_mall/indoor": True,
    "/s/shower": True,
    "/s/ski_resort": False,
    "/s/ski_slope": False,
    "/s/sky": False,
    "/s/skyscraper": False,
    "/s/slum": False,
    "/s/snowfield": False,
    "/s/soccer_field": False,
    "/s/stable": True,
    "/s/stadium/baseball": False,
    "/s/stadium/football": False,
    "/s/stadium/soccer": False,
    "/s/stage/indoor": True,
    "/s/stage/outdoor": False,
    "/s/staircase": True,
    "/s/storage_room": True,
    "/s/street": False,
    "/s/subway_station/platform": True,
    "/s/supermarket": True,
    "/s/sushi_bar": True,
    "/s/swamp": False,
    "/s/swimming_hole": True,
    "/s/swimming_pool/indoor": True,
    "/s/swimming_pool/outdoor": False,
    "/s/synagogue/outdoor": False,
    "/t/television_room": True,
    "/t/television_studio": True,
    "/t/temple/asia": False,
    "/t/throne_room": True,
    "/t/ticket_booth": True,
    "/t/topiary_garden": False,
    "/t/tower": False,
    "/t/toyshop": True,
    "/t/train_interior": True,
    "/t/train_station/platform": True,
    "/t/tree_farm": False,
    "/t/tree_house": False,
    "/t/trench": False,
    "/t/tundra": False,
    "/u/underwater/ocean_deep": False,
    "/u/utility_room": True,
    "/v/valley": False,
    "/v/vegetable_garden": False,
    "/v/veterinarians_office": True,
    "/v/viaduct": False,
    "/v/village": False,
    "/v/vineyard": False,
    "/v/volcano": False,
    "/v/volleyball_court/outdoor": False,
    "/w/waiting_room": True,
    "/w/water_park": False,
    "/w/water_tower": False,
    "/w/waterfall": False,
    "/w/watering_hole": False,
    "/w/wave": False,
    "/w/wet_bar": True,
    "/w/wheat_field": False,
    "/w/wind_farm": False,
    "/w/windmill": False,
    "/y/yard": False,
    "/y/youth_hostel": True,
    "/z/zen_garden": False,
}

# Common scene names mapped to Places365 categories.
COMMON_PLACE365_CATEGORIES = {
    "kitchen": ["/k/kitchen", "/r/restaurant_kitchen"],
    "bedroom": ["/b/bedroom", "/b/bedchamber"],
    "living room": ["/l/living_room", "/t/television_room"],
    "dining room": ["/d/dining_room", "/d/dining_hall"],
    "bathroom": ["/b/bathroom", "/s/shower"],
    "childs room": ["/c/childs_room", "/p/playroom", "/n/nursery"],
    "office": ["/h/home_office", "/o/office", "/o/office_building", "/o/office_cubicles"],
    "basement": ["/b/basement"],
    "attic": ["/a/attic"],
    "closet": ["/c/closet"],
    "pantry": ["/p/pantry"],
    "utility room": ["/u/utility_room"],
    "home theater": ["/h/home_theater"],
    "wet bar": ["/w/wet_bar"],

    "shop": [
        "/b/bakery/shop", "/b/butchers_shop", "/c/candy_store", "/c/clothing_store",
        "/d/delicatessen", "/d/drugstore", "/f/fabric_store", "/f/florist_shop/indoor",
        "/g/gift_shop", "/h/hardware_store", "/j/jewelry_shop", "/p/pet_shop",
        "/s/shoe_shop", "/s/shopfront", "/t/toyshop", "/s/supermarket",
        "/b/bookstore", "/d/department_store"
    ],
    "restaurant": [
        "/r/restaurant", "/f/fastfood_restaurant", "/p/pizzeria", "/s/sushi_bar",
        "/d/diner/outdoor", "/r/restaurant_patio", "/r/restaurant_kitchen"
    ],
    "cafe": ["/c/coffee_shop", "/c/cafeteria", "/f/food_court"],
    "bar": ["/b/bar", "/p/pub/indoor", "/b/beer_hall"],
    "studio": ["/a/art_studio", "/m/music_studio", "/t/television_studio"],
    "gallery": ["/a/art_gallery"],
    "museum": ["/m/museum/indoor", "/m/museum/outdoor", "/n/natural_history_museum", "/s/science_museum"],
    "library": ["/l/library/indoor", "/l/library/outdoor"],
    "school": ["/a/art_school", "/c/classroom", "/k/kindergarden_classroom", "/l/lecture_room", "/s/schoolhouse"],
    "gym": ["/m/martial_arts_gym", "/g/gymnasium/indoor"],
    "station": ["/b/bus_station/indoor", "/f/fire_station", "/g/gas_station", "/s/subway_station/platform", "/t/train_station/platform"],
    "lobby": ["/e/elevator_lobby", "/l/lobby", "/r/reception"],
    "room": [
        "/a/archive", "/b/ballroom", "/b/banquet_hall", "/c/conference_room",
        "/d/dorm_room", "/d/dressing_room", "/e/engine_room", "/e/entrance_hall",
        "/l/locker_room", "/r/recreation_room", "/s/server_room", "/s/storage_room",
        "/t/throne_room", "/w/waiting_room"
    ],
    "arena": ["/a/arena/hockey", "/a/arena/performance", "/a/arena/rodeo", "/b/boxing_ring"],
    "theater": ["/a/auditorium", "/m/movie_theater/indoor", "/o/orchestra_pit", "/h/home_theater"],
    "mall": ["/s/shopping_mall/indoor"],
    "arcade": ["/a/amusement_arcade", "/a/arcade"],

    "street": ["/a/alley", "/c/crosswalk", "/d/downtown", "/p/promenade", "/s/street", "/r/residential_neighborhood"],
    "plaza": ["/p/plaza"],
    "bridge": ["/b/bridge", "/r/rope_bridge"],
    "building": ["/a/apartment_building/outdoor", "/b/building_facade", "/h/house", "/m/mansion", "/s/skyscraper"],
    "construction": ["/c/construction_site"],
    "road": ["/d/desert_road", "/f/field_road", "/h/highway", "/d/driveway"],
    "track": ["/r/railroad_track", "/r/racecourse", "/r/raceway"],
    "airport": ["/a/airfield", "/a/airport_terminal", "/h/hangar/outdoor", "/r/runway"],
    "harbor": ["/h/harbor", "/b/berth", "/l/loading_dock", "/p/pier"],
    "park": ["/a/amusement_park", "/p/park", "/p/playground", "/w/water_park"],
    "garden": ["/b/botanical_garden", "/f/formal_garden", "/j/japanese_garden", "/r/roof_garden", "/t/topiary_garden", "/v/vegetable_garden", "/z/zen_garden"],
    "fountain": ["/f/fountain"],
    "courtyard": ["/c/courtyard"],
    "cemetery": ["/c/cemetery"],
    "castle": ["/c/castle"],
    "ruin": ["/r/ruin", "/a/archaelogical_excavation"],

    "forest": ["/b/bamboo_forest", "/f/forest/broadleaf", "/f/forest_path", "/f/forest_road", "/r/rainforest"],
    "field": ["/a/athletic_field/outdoor", "/b/baseball_field", "/c/corn_field", "/f/field/cultivated", "/f/field/wild", "/h/hayfield", "/r/rice_paddy", "/s/soccer_field", "/w/wheat_field"],
    "beach": ["/b/beach", "/c/coast"],
    "mountain": ["/m/mountain", "/m/mountain_path", "/m/mountain_snowy"],
    "desert": ["/b/badlands", "/b/butte", "/d/desert/sand", "/d/desert/vegetation"],
    "water": [
        "/c/canal/natural", "/c/canal/urban", "/c/creek", "/f/fishpond", "/l/lagoon",
        "/l/lake/natural", "/m/moat/water", "/o/ocean", "/p/pond", "/r/river",
        "/s/swimming_hole", "/w/watering_hole", "/w/wave"
    ],
    "valley": ["/v/valley", "/c/canyon"],
    "glacier": ["/g/glacier", "/i/ice_floe", "/i/ice_shelf", "/i/iceberg"],
    "volcano": ["/v/volcano"],
    "swamp": ["/s/swamp"],
    "marsh": ["/m/marsh", "/s/swamp"],
    "wetland": ["/m/marsh", "/s/swamp"],
    "tundra": ["/t/tundra"]
}

# Scene to environment mapping rules
SCENE_PERIOD_MAPPING = {
    "modern": "Contemporary",
    "contemporary": "Contemporary",
    "vintage": "Modern",
    "antique": "Early modern",
    # Classical antiquity belongs to the Ancient epoch.
    "classical": "Ancient",
    "medieval": "Post-classical",
    "renaissance": "Early modern",
    "ancient": "Ancient",
    "default": "Contemporary"
}

# Object capabilities used to constrain semantic relations.
OBJECT_CAPABILITIES = {
    "can_see": ["person", "animal", "cat", "dog", "bird", "horse", "elephant", "bear", "zebra", "giraffe"],
    "can_communicate": ["person", "animal", "cat", "dog", "bird"],
    "can_use_tools": ["person"],
    "can_hold": ["person"],
    "can_move": ["person", "animal", "cat", "dog", "bird", "horse", "car", "train", "bus", "bicycle", "motorcycle", "truck", "airplane", "boat"],
    "is_container": ["cup", "bowl", "bottle", "vase", "box", "backpack", "bag", "suitcase", "refrigerator", "oven", "microwave"]
}
