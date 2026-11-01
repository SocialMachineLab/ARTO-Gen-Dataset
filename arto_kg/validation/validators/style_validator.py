
"""Style labels, descriptions and synonyms used by the VLM style classifier."""


ART_STYLES = [
    "Baroque",
    "Impressionism",
    "Chinese Ink Painting",
    "Post-Impressionism",
    "Neoclassicism",
]

STYLE_PROMPTS = {
    "Baroque": "a Baroque painting with dramatic lighting, rich ornamentation, and dynamic composition",
    "Impressionism": "an Impressionist painting with visible brushstrokes and light effects",
    "Chinese Ink Painting": "a traditional Chinese ink wash painting with brush calligraphy",
    "Post-Impressionism": "a Post-Impressionist painting with bold colors, expressive brushwork, and geometric forms",
    "Neoclassicism": "a Neoclassical painting with classical Greek and Roman themes, harmonious composition",
}

STYLE_SYNONYMS = {
    'Baroque': ['baroque', 'baroque style', 'baroque painting', 'baroque period', 'baroque art'],
    'Impressionism': ['impressionism', 'impressionist', 'impressionist painting', 'impressionist style'],
    'Chinese Ink Painting': ['chinese ink painting', 'ink painting', 'chinese painting',
                             'traditional chinese painting', 'sumi-e', 'ink wash', 'chinese'],
    'Post-Impressionism': ['post-impressionism', 'post impressionism', 'postimpressionism',
                           'post-impressionist', 'post impressionist', 'postimpressionist'],
    'Neoclassicism': ['neoclassicism', 'neoclassical', 'neoclassical style',
                      'neoclassical painting', 'neo-classical'],
}
