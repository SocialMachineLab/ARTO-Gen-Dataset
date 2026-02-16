"""
# model_config.py
# Unified configuration file for ARTO-KG
"""

# openai/gpt-oss-120b
# vLLM model configuration
GENERATION_LLM_CONFIG = {
    "model": "openai/gpt-oss-120b",
    "tensor_parallel_size": 2,  # Use 2 GPUs
    "gpu_memory_utilization": 0.95 ,
    "max_model_len": 4096,
    "trust_remote_code": True,
    "max_num_seqs": 2,
    "enforce_eager": True,
    "disable_custom_all_reduce": True
}

VALIDATION_VLM_CONFIG= {
    "model": "Qwen/Qwen2.5-VL-32B-Instruct-AWQ",
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "trust_remote_code": True,
    "max_num_seqs": 2,
    "enforce_eager": True,
    "disable_custom_all_reduce": True
}

# fallback to a smaller model
FALLBACK_LLM_CONFIG = {
    "model": "Qwen/Qwen3-32B-AWQ",
    "gpu_memory_utilization": 0.8,
    "max_model_len": 4096,
    "trust_remote_code": True,
    "max_num_seqs": 4
}

FALLBACK_VLM_CONFIG = {
    "model": "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    "gpu_memory_utilization": 0.8,
    "max_model_len": 4096,
    "trust_remote_code": True,
    "max_num_seqs": 4
}

# batch processing config
BATCH_CONFIG = {
    "default_batch_size": 16,
    "max_batch_size": 32,
    "timeout_seconds": 300,
    "max_retries": 3,
    "retry_delay": 10
}

# generation config
GENERATION_CONFIG = {
    "max_secondary_objects": 8,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 8192,
    "stop_tokens": [
        "<|im_end|>" 
    ]
}

# SYSTEM DEFAULT CONFIG (from default_config.json)
DEFAULT_CONFIG = {
  "model": {
    "name": "Qwen/Qwen-Image",
    "torch_dtype": "bfloat16",
    "device_map": "balanced",
    "low_cpu_mem_usage": True,
    "enable_optimizations": {
      "vae_slicing": True,
      "vae_tiling": True,
      "attention_slicing": True
    }
  },
  "generation": {
    "default_params": {
      "width": 1024,
      "height": 1024,
      "num_inference_steps": 80,
      "true_cfg_scale": 1.0,
      "seed": 42
    },
    "style_adjustments": {
      "photorealistic": {
        "num_inference_steps": 100,
        "true_cfg_scale": 1.2
      },
      "anime": {
        "true_cfg_scale": 1.3,
        "num_inference_steps": 90
      },
      "sketch": {
        "num_inference_steps": 60,
        "true_cfg_scale": 0.8
      },
      "cinematic": {
        "width": 1280,
        "height": 720,
        "true_cfg_scale": 1.1
      }
    }
  },
  "processing": {
    "batch_size": 1,
    "max_prompt_length": 450,
    "backup_existing": True,
    "save_generation_info": True,
    "memory_cleanup_interval": 5
  },
  "logging": {
    "level": "INFO",
    "save_to_file": True,
    "log_filename": "generation.log"
  },
  "paths": {
    "default_output_dir": "outputs",
    "default_cache_dir": "cache",
    "log_dir": "logs"
  },
  "safety": {
    "min_disk_space_gb": 10.0,
    "max_generation_time_minutes": 30,
    "retry_failed_generations": 2
  }
}

# COCO objects list
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

# Art Styles List
ART_STYLES = [
    "Renaissance", "Baroque", "Impressionism", "Romantic", "Realism", 
    "Abstract", "Surrealism", "Cubism", "Pop Art", "Minimalism",
    "Classical", "Gothic", "Art Nouveau", "Expressionism", "Fauvism",
    "Post-Impressionism", "Sketch", "Chinese Ink Painting", "Photorealistic", "Oil Painting","Realistic","Van Gogh Style","Neoclassicism"
]

# Style Database (Merged from style_database.json)
STYLE_DATABASE = {
  "Van Gogh Style":{
  },
  "Renaissance": {
    "characteristics": ["balanced composition", "realistic proportions", "classical harmony", "mathematical precision"],
    "color_tendencies": ["warm earth tones", "rich blues", "golden highlights", "natural flesh tones"],
    "typical_subjects": ["portraits", "religious scenes", "mythological themes", "architectural studies"],
    "techniques": ["sfumato", "chiaroscuro", "linear perspective", "anatomical accuracy"]
  },
  "Baroque": {
    "characteristics": ["dramatic contrast", "dynamic movement", "emotional intensity", "theatrical composition"],
    "color_tendencies": ["deep shadows", "brilliant highlights", "rich gold", "dramatic reds"],
    "typical_subjects": ["religious drama", "historical scenes", "still life", "portraits"],
    "techniques": ["tenebrism", "foreshortening", "diagonal compositions", "rich textures"]
  },
  "Impressionism": {
    "characteristics": ["loose brushwork", "light effects", "outdoor scenes", "momentary impressions"],
    "color_tendencies": ["bright colors", "natural light", "color mixing", "pure pigments"],
    "typical_subjects": ["landscapes", "everyday scenes", "light studies", "water reflections"],
    "techniques": ["broken color", "en plein air", "visible brushstrokes", "color theory"]
  },
  "Romantic": {
    "characteristics": ["emotional expression", "dramatic scenes", "sublime nature", "individualism"],
    "color_tendencies": ["rich colors", "atmospheric effects", "moody tones", "expressive palette"],
    "typical_subjects": ["dramatic landscapes", "historical events", "exotic scenes", "emotional portraits"],
    "techniques": ["expressive brushwork", "atmospheric perspective", "dramatic lighting", "emotional color"]
  },
  "Realistic": {
    "characteristics": ["accurate representation", "everyday subjects", "social commentary", "objective observation"],
    "color_tendencies": ["natural colors", "earth tones", "realistic lighting", "subdued palette"],
    "typical_subjects": ["working class", "rural scenes", "social issues", "contemporary life"],
    "techniques": ["precise detail", "accurate proportions", "realistic lighting", "careful observation"]
  },
  "Realism": {  # Alias for Realistic
    "characteristics": ["accurate representation", "everyday subjects", "social commentary", "objective observation"],
    "color_tendencies": ["natural colors", "earth tones", "realistic lighting", "subdued palette"],
    "typical_subjects": ["working class", "rural scenes", "social issues", "contemporary life"],
    "techniques": ["precise detail", "accurate proportions", "realistic lighting", "careful observation"]
  },
  "Abstract": {
    "characteristics": ["non-representational", "focus on form and color", "geometric or organic shapes", "pure visual elements"],
    "color_tendencies": ["bold contrasts", "pure colors", "experimental combinations", "symbolic color use"],
    "typical_subjects": ["geometric forms", "color studies", "emotional expressions", "conceptual ideas"],
    "techniques": ["color field", "geometric abstraction", "gestural painting", "mixed media"]
  },
  "Surrealism": {
    "characteristics": ["dreamlike imagery", "subconscious exploration", "impossible combinations", "symbolic content"],
    "color_tendencies": ["unexpected combinations", "hyper-realistic detail", "fantastical colors", "psychological impact"],
    "typical_subjects": ["dreams", "fantasies", "psychological states", "impossible worlds"],
    "techniques": ["automatism", "photomontage", "detailed realism", "symbolic imagery"]
  },
  "Cubism": {
    "characteristics": ["geometric forms", "multiple perspectives", "fragmented objects", "analytical approach"],
    "color_tendencies": ["muted earth tones", "analytical grays", "ochres and browns", "reduced palette"],
    "typical_subjects": ["still life", "portraits", "musical instruments", "everyday objects"],
    "techniques": ["geometric reduction", "multiple viewpoints", "collage", "analytical deconstruction"]
  },
  "Pop Art": {
    "characteristics": ["commercial imagery", "bold colors", "mass culture references", "repetition and variation"],
    "color_tendencies": ["bright primary colors", "commercial printing colors", "high contrast", "artificial colors"],
    "typical_subjects": ["consumer products", "celebrities", "advertising imagery", "popular culture"],
    "techniques": ["screen printing", "commercial techniques", "mass production methods", "appropriation"]
  },
  "Minimalism": {
    "characteristics": ["simplicity", "reduced elements", "geometric forms", "repetition"],
    "color_tendencies": ["monochromatic", "neutral tones", "pure colors", "limited palette"],
    "typical_subjects": ["geometric shapes", "industrial materials", "simple forms", "spatial relationships"],
    "techniques": ["reduction", "repetition", "industrial materials", "precise execution"]
  },
  "Classical": {
    "characteristics": ["idealized forms", "balanced composition", "noble subjects", "timeless beauty"],
    "color_tendencies": ["harmonious colors", "golden ratios", "natural flesh tones", "classical palette"],
    "typical_subjects": ["mythology", "historical events", "idealized figures", "architectural subjects"],
    "techniques": ["academic drawing", "classical proportions", "smooth finish", "idealization"]
  },
  "Gothic": {
    "characteristics": ["vertical emphasis", "spiritual themes", "ornate details", "dramatic lighting"],
    "color_tendencies": ["rich jewel tones", "gold leaf", "deep blues", "symbolic colors"],
    "typical_subjects": ["religious scenes", "architectural details", "manuscripts", "spiritual themes"],
    "techniques": ["detailed craftsmanship", "symbolic imagery", "gold leaf application", "manuscript illumination"]
  },
  "Art Nouveau": {
    "characteristics": ["organic forms", "flowing lines", "natural motifs", "decorative elements"],
    "color_tendencies": ["natural colors", "soft pastels", "organic earth tones", "flowing gradients"],
    "typical_subjects": ["natural forms", "decorative arts", "feminine figures", "botanical studies"],
    "techniques": ["curved lines", "natural patterns", "decorative integration", "stylized forms"]
  },
  "Expressionism": {
    "characteristics": ["emotional intensity", "distorted forms", "bold colors", "psychological content"],
    "color_tendencies": ["intense colors", "non-naturalistic palette", "emotional color use", "high contrast"],
    "typical_subjects": ["emotional states", "social criticism", "psychological portraits", "urban scenes"],
    "techniques": ["distortion", "bold brushwork", "emotional color", "expressive mark-making"]
  },
  "Fauvism": {
    "characteristics": ["wild colors", "bold brushwork", "simplified forms", "pure color expression"],
    "color_tendencies": ["pure bright colors", "non-naturalistic color", "high saturation", "color liberation"],
    "typical_subjects": ["landscapes", "portraits", "still life", "scenes of leisure"],
    "techniques": ["pure color application", "bold brushstrokes", "color liberation", "simplified forms"]
  },
  "Post-Impressionism": {
    "characteristics": ["personal expression", "symbolic color", "structured composition", "emotional content"],
    "color_tendencies": ["expressive color", "symbolic use", "structural color", "personal palette"],
    "typical_subjects": ["landscapes", "still life", "portraits", "everyday scenes"],
    "techniques": ["personal style", "color symbolism", "structural approach", "expressive brushwork"]
  },
  "Sketch": {
    "characteristics": ["loose drawing", "gestural marks", "quick studies", "preparatory work"],
    "color_tendencies": ["monochromatic", "limited palette", "tonal studies", "linear emphasis"],
    "typical_subjects": ["figure studies", "landscape sketches", "compositional studies", "observational drawing"],
    "techniques": ["quick drawing", "gestural marks", "loose technique", "observational skills"]
  },
  "Chinese Ink Painting": {
    "characteristics": ["brush economy", "spiritual expression", "nature harmony", "empty space"],
    "color_tendencies": ["black ink variations", "subtle colors", "water effects", "tonal gradations"],
    "typical_subjects": ["landscapes", "bamboo", "flowers", "birds"],
    "techniques": ["brush control", "ink gradation", "wet-on-wet", "calligraphic strokes"]
  },
  "Photorealistic": {
    "characteristics": ["extreme detail", "photographic accuracy", "technical precision", "reality simulation"],
    "color_tendencies": ["accurate color reproduction", "photographic lighting", "realistic shadows", "precise tones"],
    "typical_subjects": ["portraits", "still life", "urban scenes", "contemporary subjects"],
    "techniques": ["precise detail", "accurate proportions", "photographic reference", "technical skill"]
  },
  "Photorealism": { # Alias for Photorealistic
    "characteristics": ["extreme detail", "photographic accuracy", "technical precision", "reality simulation"],
    "color_tendencies": ["accurate color reproduction", "photographic lighting", "realistic shadows", "precise tones"],
    "typical_subjects": ["portraits", "still life", "urban scenes", "contemporary subjects"],
    "techniques": ["precise detail", "accurate proportions", "photographic reference", "technical skill"]
  },
  "Oil Painting": {
    "characteristics": ["rich textures", "blending capability", "layered technique", "traditional medium"],
    "color_tendencies": ["rich colors", "subtle gradations", "warm undertones", "traditional palette"],
    "typical_subjects": ["portraits", "landscapes", "still life", "classical subjects"],
    "techniques": ["glazing", "impasto", "wet-on-wet", "traditional methods"]
  },
  "Neoclassicism": {
    "characteristics": ["inspired by classical antiquity", "emphasizes order and symmetry", "idealized forms"],
    "color_tendencies": ["clear colors", "somber colors", "uncomplicated palette"],
    "typical_subjects": ["mythology", "historical events", "heroes", "patriotism"],
    "techniques": ["smooth brushwork", "clean lines", "shallow depth"]
  }
}

# Relations Config (Merged from merge relations.json)
RELATION_CONFIG = {
    "spatial_relations": [
        "above", "against", "around", "behind", "below", "beside",
        "between", "centered", "contains", "in", "in front of", "inside",
        "left of", "near", "on", "opposite", "outside", "over",
        "part of", "right of", "surrounding", "through", "under"
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
        "pose_and_states": [
            "balancing on", "connected to", "covered by", "decorated with",
            "displayed on", "draped over", "dressed in", "enclosing",
            "exposed to", "facing", "filled with", "floating on", "framed by",
            "full of", "growing on", "hanging on", "leaning against", "lying in",
            "lying on", "mounted on", "occupied by", "painted on", "parked",
            "perched on", "piercing", "planted in", "posing with", "printed on",
            "reflected in", "resting on", "shading", "sitting at", "sitting on",
            "standing on", "stuck in", "supporting", "suspended from",
            "tied to", "wrapped in"
        ]
    }
}

# Style Coco Variety (Merged from style_coco_variety.json) - Partial/Structure Only as example or Full
# Truncating slightly for brewity in generation if allowed, but user wants full merge.
# I will include the full dictionary structure for the key styles provided in the json.

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
        "dining table": ["dining table", "dinning table", "picnic table", "round table", "side table", "table"],
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
        "dining table": ["billiard table", "dining table", "dinning table", "picnic table", "place mat", "round table", "side table", "table"],
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
        "dining table": ["dining table", "dinning table", "picnic table", "round table", "side table", "table"],
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
        "dining table": ["billiard table", "dining table", "dinning table", "picnic table", "round table", "side table", "table"],
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

# Style Object Mapping (Derived from STYLE_COCO_VARIETY for compatibility)
# This controls which COCO objects are allowed for each style
STYLE_OBJECT_MAPPING = {
    style: {"objects": list(data.keys())}
    for style, data in STYLE_COCO_VARIETY.items()
}

# conflict detection prompt
COMPATIBILITY_SYSTEM_PROMPT = """
You are an expert composition analyzer. Your task is to evaluate whether a set of objects can realistically coexist in a single scene.

Your focus is on identifying ABSOLUTE CONFLICTS - situations where it would be impossible or extremely unlikely for two objects to appear together in the same scene under any normal circumstances.

INPUT FORMAT:
You will receive objects in this format:
main object: [main_object_name]
secondary objects: [list_of_secondary_objects]

ABSOLUTE CONFLICT TYPES TO CONSIDER:
- Physical impossibility (objects that cannot physically coexist in the same space/scene)
- Universal incompatibility (objects that are universally recognized as never appearing together)
- Fundamental logical conflicts (objects that contradict each other's basic nature or purpose)

RESPONSE FORMAT:
You MUST respond with a valid JSON object in exactly this format:

If compatible (no absolute conflicts):
{
"compatible": true,
"example_scenario": "A brief description of a realistic scenario where these objects could coexist"
}

If incompatible (absolute conflicts exist):
{
"compatible": false,
"main_object_conflicts": [
    {"object1": "main_object_name", "object2": "conflicting_object_name", "reason": "explanation"}
],
"secondary_object_conflicts": [
    {"object1": "object_name", "object2": "object_name", "reason": "explanation"}
]
}

IMPORTANT RULES:
- Only identify ABSOLUTE conflicts, not minor inconveniences or preferences
- If "compatible" is true, provide a realistic example scenario
- If "compatible" is false, list ALL absolute conflict pairs with reasons
- Be very conservative - only mark as conflict if truly impossible/extremely unlikely
- Always respond with valid JSON only
- Do not include any text outside the JSON response"""


COMPATIBILITY_WITH_SCENE_INFERENCE_PROMPT = """CRITICAL: Your entire response must be ONLY the JSON object. Start with { and end with }. Do not write any explanations, thinking, or text before or after the JSON.

You are an expert composition analyzer and scene inference system. Your tasks are:

1. Evaluate whether objects can realistically coexist in a single scene
2. Infer 5-10 plausible scenes where ALL these objects could appear together
3. Provide detailed context interpretations for each object

INPUT FORMAT:
You will receive objects in this format:
Main Object: [main_object_name]
Secondary Objects: [list_of_secondary_objects]
Art Style: [style]

YOUR TASKS:

Task 1 - CONFLICT DETECTION:
- Identify ABSOLUTE conflicts only (physical/logical impossibility)
- Be very conservative - only mark if truly impossible to coexist

Task 2 - SCENE INFERENCE:
- Based on the PRIMARY object as the main anchor, infer 5-10 possible scene types
- Consider where the primary object typically appears (its natural habitat)
- Scenes should be GENERAL categories, not overly specific
- Examples: "modern kitchen", "children's playroom", "outdoor park", "art studio", "vintage shop"
- Prioritize scenes that make sense for the primary object
- Be creative: secondary objects can be toys, models, decorations, or artworks
- Consider the art style: Abstract/Surreal allows more creative interpretations

Task 3 - OBJECT INTERPRETATION:
- For EACH scene candidate, specify how EACH object appears
- Primary object should usually be in its natural/typical form
- Secondary objects can be adapted (toys, models, decorations) to fit
- Consider: form (real/toy/model/decoration), scale (miniature/toy/normal/large), role (primary/secondary/decorative)
- In "reasoning", describe the scene AND how each object exists in it (e.g., "kitchen with real oven, toy train on shelf, apple on counter")

RESPONSE FORMAT (Must be valid JSON):

If you CAN find plausible scenes:
{
    "success": true,
    "scenes_found": true,
    "scene_candidates": [
        {
            "scene_type": "modern kitchen",
            "plausibility": 0.9,
            "reasoning": "oven is primary appliance, train can be toy on shelf, apple is food",
            "object_interpretations": {
                "oven": {
                    "form": "real appliance",
                    "scale": "normal/full-size",
                    "role": "primary functional object"
                },
                "train": {
                    "form": "toy model",
                    "scale": "miniature (1:100)",
                    "role": "decorative element on shelf"
                },
                "apple": {
                    "form": "real fruit",
                    "scale": "normal",
                    "role": "food ingredient"
                }
            }
        }
    ]
}

If you CANNOT find ANY plausible scenes:
{
    "success": false,
    "scenes_found": false,
    "incompatibility_analysis": {
        "most_problematic_object": "object_name",
        "reason": "detailed explanation why this object cannot fit in any scene with the others",
        "conflicts": [
            {"object1": "name", "object2": "name", "reason": "why they conflict"}
        ]
    }
}

IMPORTANT RULES:
- If you can find ANY plausible scene, set success=true
- Only set success=false if NO scene is possible even with creative interpretation
- Provide at least 5 scene candidates when success=true
- For each scene, provide object_interpretations for ALL objects
- The primary object should drive the scene selection
- Be creative with secondary object adaptations (scale, form)
- Make "reasoning" field descriptive: explain the scene and how each object fits

CRITICAL REMINDER: Return ONLY the JSON object. Your response must start with { and end with }. No thinking, no explanation, no additional text."""

# Environment options for detailed environment specification
TIME_OF_DAY_OPTIONS = [
    "morning", "afternoon", "noon", "sunrise", "sunset",
    "evening", "night"
]

WEATHER_OPTIONS = [
    "sunny", "cloudy", "rainy", "snowy", "foggy",
    "windy", "duststorm", "drizzle", "showery"
]

PERIOD_OPTIONS = [
    "Ancient", "Post-classical", "Early modern",
    "Early Modern", "Modern", "Contemporary"
]

COMMON_PLACE365_CATEGORIES = {
    # Indoor - Residential
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

    # Indoor - Commercial & Public
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

    # Outdoor - Urban & Man-made
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

    # Outdoor - Natural
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
    "tundra": ["/t/tundra"],

    # Default
    "default": ["/a/art_studio"]
}

# Scene to environment mapping rules
SCENE_TIME_MAPPING = {
    "kitchen": ["morning", "afternoon", "evening"],
    "bedroom": ["morning", "evening", "night"],
    "office": ["morning", "afternoon"],
    "park": ["morning", "afternoon", "sunset"],
    "beach": ["morning", "afternoon", "sunset"],
    "cafe": ["morning", "afternoon"],
    "restaurant": ["evening", "afternoon"],
    "playroom": ["morning", "afternoon"],
    "studio": ["afternoon", "morning"],
    "garden": ["morning", "afternoon", "sunset"],
    "street": ["morning", "afternoon", "evening", "night"],
    "default": ["afternoon"]
}

SCENE_PERIOD_MAPPING = {
    "modern": "Contemporary",
    "contemporary": "Contemporary",
    "vintage": "Modern",
    "antique": "Early modern",
    "classical": "Post-classical",
    "medieval": "Post-classical",
    "renaissance": "Early modern",
    "ancient": "Ancient",
    "default": "Contemporary"
}

ENHANCEMENT_SYSTEM_PROMPT = """
You are an expert art object designer. Your task is to enhance basic objects with specific details including quantities, colors, states, and sizes for artistic composition.

Your task is to determine:
1. Specific quantities for each object (how many of each)
2. Detailed colors for each object
3. Current state/condition/appearance of each object
4. Relative sizes within the composition
5. Specific variations or types
6. Position preference
7. Importance level

RESPONSE FORMAT:
You MUST respond with a valid JSON object in exactly this format:

{
    "enhanced_objects": [
        {
            "name": "object_name",
            "quantity": number,
            "colors": ["color1", "color2"],
            "state": "detailed description of object state/condition",
            "size": "small/medium/large/tiny/huge",
            "specific_type": "more specific description if applicable",
            "position_preference": "foreground/midground/background",
            "importance": "primary/secondary/accent"
        }
    ],
    "overall_notes": "general notes about the object arrangement"
}

IMPORTANT RULES:
- Provide realistic and harmonious color combinations
- States should be vivid and specific (e.g., "blooming sunflower turning toward light" not just "flower")
- Sizes should create interesting visual hierarchy
- Consider how objects might interact or relate to each other
- Always respond with valid JSON only
- Do not include any text outside the JSON response"""
    

# Spatial relations
SPATIAL_RELATIONS = [
    # Horizontal positioning
    "left_of", "right_of",
    # Vertical positioning
    "above", "below",
    # Distance
    "near", "far",
    # Contact/overlap (can be approximated with bbox)
    "touching", "overlapping",
    # Containment
    "inside", "contains",
    # Support (approximated with 2D)
    "on", "under",
    # Alignment
    "beside", "aligned_horizontally", "aligned_vertically",
    # Depth approximation (using overlap)
    "in_front_of", "behind"
]

# Semantic relations - categorized by type
SEMANTIC_RELATIONS = {
    "visual": [
        "looking_at", "watching", "gazing_at", "facing_toward", "staring_at"
    ],
    "communicative": [
        "talking_to", "listening_to", "conversing_with", "gesturing_to",
        "showing_to", "presenting_to"
    ],
    "action": [
        "using", "holding", "playing_with", "reading", "drinking_from",
        "eating", "touching", "reaching_for", "pointing_at", "carrying",
        "operating", "manipulating"
    ],
    "social": [
        "interacting_with", "accompanying", "following", "greeting",
        "guarding", "protecting", "chasing", "approaching"
    ],
    "state": [
        "containing", "filled_with", "decorated_with", "framing",
        "supporting", "resting_in"
    ]
}

# Object capability for semantic relations
OBJECT_CAPABILITIES = {
    "can_see": ["person", "animal", "cat", "dog", "bird", "horse", "elephant", "bear", "zebra", "giraffe"],
    "can_communicate": ["person", "animal", "cat", "dog", "bird"],
    "can_use_tools": ["person"],
    "can_hold": ["person"],
    "can_move": ["person", "animal", "cat", "dog", "bird", "horse", "car", "train", "bus", "bicycle", "motorcycle", "truck", "airplane", "boat"],
    "is_container": ["cup", "bowl", "bottle", "vase", "box", "backpack", "bag", "suitcase", "refrigerator", "oven", "microwave"]
}

# Prompt templates
PROMPT_TEMPLATES = {
    "chat_template": "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n",
    "json_parsing_patterns": [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    ],
    "thinking_pattern": r'<think>.*?</think>'
}
