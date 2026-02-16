"""Validation objects and prompt configurations for style analysis experiments"""

COCO_UNIVERSAL_OBJECTS = [
    # People
    'person',
    # Animals
    'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'duck', 'rabbit', 'fox', 'deer', 'lion',
    # Food & Dining
    'bottle', 'wine glass', 'cup', 'bowl',
    'banana', 'apple', 'orange', 'carrot',
    'fork', 'knife', 'spoon', 'plate',
    # Furniture
    'chair', 'couch', 'bed', 'dining table',
    'bench', 'desk', 'shelf', 'cabinet',
    # Household
    'potted plant', 'vase', 'book', 'clock',
    'mirror', 'candle', 'lamp', 'basket',
    # Outdoor/Transport
    'boat', 'umbrella', 'bicycle',
    # Personal
    'backpack', 'handbag', 'hat'
]

# Total: 50 objects
assert len(COCO_UNIVERSAL_OBJECTS) == 50, f"Expected 50 objects, got {len(COCO_UNIVERSAL_OBJECTS)}"

# Backward compatibility
ALL_VALIDATION_OBJECTS = COCO_UNIVERSAL_OBJECTS
TOP_20_OBJECTS = COCO_UNIVERSAL_OBJECTS[:20]

VALIDATION_STYLES = [
    'Romanticism',        
    'Baroque',            
    'Realism',             
    'Post_Impressionism',   
    'Impressionism',       
    'Art_Nouveau',        
    'Surrealism',          
    'Expressionism',        
    'Neoclassicism',        
    'Symbolism',         
    'Photorealism',         
    'Chinese_painting'      
]

STYLE_PROMPTS = {
    'Romanticism': 'romantic painting, dramatic lighting, emotional atmosphere, nature landscape, turbulent skies, sublime beauty, delacroix style',
    'Baroque': 'baroque painting, dramatic chiaroscuro lighting, rich deep colors, ornate details, dynamic composition, theatrical, caravaggio rubens style',
    'Realism': 'realistic oil painting, classical realism style, detailed brushwork, naturalistic lighting, everyday life, courbet millet style',
    'Post_Impressionism': 'post-impressionist painting, bold vibrant colors, expressive brushstrokes, structured composition, cezanne van gogh gauguin style',
    'Impressionism': 'impressionist painting, dappled light, soft colors, loose visible brushstrokes, outdoor scene, capturing moment, monet renoir style',
    'Art_Nouveau': 'art nouveau style, decorative, flowing curved lines, organic floral forms, elegant ornamental, alphonse mucha klimt style',
    'Surrealism': 'surrealist painting, dreamlike atmosphere, symbolic, unexpected juxtaposition, precise detail, subconscious, salvador dali magritte style',
    'Expressionism': 'expressionist painting, intense emotion, distorted exaggerated forms, bold contrasting colors, psychological, edvard munch kirchner style',
    'Neoclassicism': 'neoclassical painting, classical composition, idealized noble forms, clear contours, balanced harmony, ancient themes, jacques-louis david style',
    'Symbolism': 'symbolist painting, mystical dreamlike, allegorical symbolic, rich colors, literary mythological references, gustave moreau redon style',
    'Photorealism': 'photorealistic painting, hyperrealistic, extremely detailed, precise technique, looks exactly like photograph, sharp focus',
    'Chinese_painting': 'traditional chinese ink painting, brush and ink, minimal composition, elegant flowing lines, empty space, song dynasty literati style'
}

VALIDATION_TEST = {
    'objects': COCO_UNIVERSAL_OBJECTS,
    'styles': VALIDATION_STYLES,
    'images_per_combination': 1,
    'total_images': len(COCO_UNIVERSAL_OBJECTS) * len(VALIDATION_STYLES)
}

TIER1_TEST = VALIDATION_TEST
TIER2_TEST = VALIDATION_TEST
TIER3_TEST = VALIDATION_TEST

EXPECTED_DETECTABILITY = {
    'Photorealism': {'expected_detection_rate': 0.90, 'rationale': 'Photorealistic = highly recognizable, maximum detail'},
    'Realism': {'expected_detection_rate': 0.80, 'rationale': 'Realistic but painterly, clear forms'},
    'Neoclassicism': {'expected_detection_rate': 0.75, 'rationale': 'Clear idealized forms, well-defined objects'},
    'Art_Nouveau': {'expected_detection_rate': 0.70, 'rationale': 'Decorative but identifiable, flowing lines'},
    'Baroque': {'expected_detection_rate': 0.68, 'rationale': 'Dramatic lighting, ornate but recognizable'},
    'Romanticism': {'expected_detection_rate': 0.65, 'rationale': 'Dramatic atmosphere, recognizable subjects'},
    'Impressionism': {'expected_detection_rate': 0.55, 'rationale': 'Loose brushwork, less detail'},
    'Post_Impressionism': {'expected_detection_rate': 0.55, 'rationale': 'Bold stylized colors, abstract'},
    'Chinese_painting': {'expected_detection_rate': 0.50, 'rationale': 'Minimalist, simplified forms'},
    'Symbolism': {'expected_detection_rate': 0.45, 'rationale': 'Symbolic, less literal'},
    'Expressionism': {'expected_detection_rate': 0.40, 'rationale': 'Distorted forms, emotional'},
    'Surrealism': {'expected_detection_rate': 0.35, 'rationale': 'Dreamlike, unrealistic settings'}
}


def generate_prompt(object_name, style_name, additional_context=''):
    """Generate a prompt for image generation"""
    style_modifier = STYLE_PROMPTS.get(style_name, '')

    if additional_context:
        base_prompt = f"a {object_name} {additional_context}"
    else:
        base_prompt = f"a {object_name}"

    return f"{base_prompt}, {style_modifier}"


if __name__ == '__main__':
    print("Validation Experiment Configuration")
    print("-" * 50)
    print(f"Total images to generate: {VALIDATION_TEST['total_images']}")
    print(f"Styles: {len(VALIDATION_STYLES)}")
    print(f"Objects per style: {len(COCO_UNIVERSAL_OBJECTS)}")
    print(f"Images per combination: {VALIDATION_TEST['images_per_combination']}")

    print("\nStyles (sorted by expected detectability):")
    print(f"{'Style':<25} {'Rate':<10} {'Rationale'}")
    print("-" * 50)
    for style in VALIDATION_STYLES:
        rate = EXPECTED_DETECTABILITY[style]['expected_detection_rate']
        rationale = EXPECTED_DETECTABILITY[style]['rationale']
        print(f"{style:<25} {rate:<10.2f} {rationale}")
