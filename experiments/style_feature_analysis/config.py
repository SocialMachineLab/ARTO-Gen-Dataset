"""
Configuration file for style feature extraction and analysis
"""

import os

# Styles
STYLES = [
    'Impressionism',
    'Realism',
    'Romanticism',
    'Post-Impressionism',
    'Expressionism',
    'Baroque',
    'Surrealism',
    'Art Nouveau Modern',
    'Symbolism',
    'Neoclassicism',
    'Photorealism',
    'Ink and wash painting'
]

# Paths
# Directory of this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Image directory
# Expected structure: IMAGE_DIR/Impressionism/*.jpg, IMAGE_DIR/Realism/*.jpg, etc.
# Can be overridden via environment variable STYLE_IMAGE_DIR
IMAGE_DIR = os.getenv('STYLE_IMAGE_DIR', os.path.join(PROJECT_ROOT, 'wiki_images_by_style'))

# Output directories
# Can be overridden via STYLE_ANALYSIS_OUTPUT_DIR
OUTPUT_DIR = os.getenv('STYLE_ANALYSIS_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs'))
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
METADATA_DIR = os.path.join(OUTPUT_DIR, 'metadata')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# Sampling
# Number of images to sample per style (if a style has fewer images, use all)
SAMPLE_SIZE = 500

# Random seed for reproducibility
RANDOM_SEED = 42

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# Models
MODELS = {
    'dinov3_vit': {
        'name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'feature_dim': 1024,
        'input_size': 224
    },
    'dinov3_convnext': {
        'name': 'facebook/dinov3-convnext-large-pretrain-lvd1689m',
        'feature_dim': 1536,
        'input_size': 224
    },
    'siglip': {
        'name': 'google/siglip2-so400m-patch14-384',
        'feature_dim': 1152,
        'input_size': 384
    },
    'convnextv2': {
        'name': 'facebook/convnextv2-huge-22k-384',
        'feature_dim': 2816,
        'input_size': 384
    }
}

# Processing
# Batch size for feature extraction
BATCH_SIZE = 32

# Device configuration (will be auto-detected in feature_extractors.py)
# Options: 'cuda', 'mps', 'cpu'
DEVICE = 'auto'  # Auto-detect best available device

NUM_WORKERS = 4

# Analysis
# Distance metric for inter-class analysis
DISTANCE_METRIC = 'cosine'  # Options: 'cosine', 'euclidean'

# The five ARTO dataset styles (WikiArt folder names)
TARGET_COMBINATION = [
    'Baroque',
    'Neoclassicism',
    'Impressionism',
    'Post-Impressionism',
    'Ink and wash painting'
]


# Visualization
# Figure DPI for saved plots
FIG_DPI = 300

# Colormap for heatmaps
HEATMAP_CMAP = 'viridis'

# Figure size
FIGSIZE = (12, 10)
