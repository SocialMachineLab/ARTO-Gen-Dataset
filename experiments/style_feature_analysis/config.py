"""
Configuration file for style feature extraction and analysis
"""

import os

# ==================== Styles Configuration ====================
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

# ==================== Path Configuration ====================
# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Image directory - CHANGE THIS TO YOUR IMAGE FOLDER PATH
# Expected structure: IMAGE_DIR/Impressionism/*.jpg, IMAGE_DIR/Realism/*.jpg, etc.
# Can be overridden via environment variable STYLE_IMAGE_DIR
IMAGE_DIR = os.getenv('STYLE_IMAGE_DIR', os.path.join(PROJECT_ROOT, 'wiki_images_by_style'))

# Output directories
# Allow overriding via environment variable for testing
OUTPUT_DIR = os.getenv('STYLE_ANALYSIS_OUTPUT_DIR', os.path.join(PROJECT_ROOT, 'outputs'))
FEATURES_DIR = os.path.join(OUTPUT_DIR, 'features')
METADATA_DIR = os.path.join(OUTPUT_DIR, 'metadata')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

# ==================== Sampling Configuration ====================
# Number of images to sample per style (if a style has fewer images, use all)
SAMPLE_SIZE = 500

# Random seed for reproducibility
RANDOM_SEED = 42

# Supported image extensions
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# ==================== Model Configuration ====================
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

# ==================== Processing Configuration ====================
# Batch size for feature extraction (adjust based on your GPU memory)
BATCH_SIZE = 32  # Reduce to 8 or 16 if running out of memory

# Device configuration (will be auto-detected in feature_extractors.py)
# Options: 'cuda', 'mps', 'cpu'
DEVICE = 'auto'  # Auto-detect best available device

# Number of workers for data loading
NUM_WORKERS = 4

# ==================== Analysis Configuration ====================
# Distance metric for inter-class analysis
DISTANCE_METRIC = 'cosine'  # Options: 'cosine', 'euclidean'

# Your target 5-style combination for comparison
TARGET_COMBINATION = [
    'Photorealism',
    'Realism',
    'Impressionism',
    'Post-Impressionism',
    'Ink and wash painting'
]



# ==================== Object Detection Configuration ====================
# New Approach: SAM 2.1 + OWL-ViT v2 for Detectability Assessment

# ========== SAM 2.1 Configuration ==========
SAM_CONFIG = {
    'model_type': 'hiera_large',  
    
    'points_per_side': 16,  # Grid points for automatic mask generation
    'pred_iou_thresh': 0.88,  # Quality threshold
    'stability_score_thresh': 0.95,  # Stability threshold
    'min_mask_region_area': 100,  # Minimum mask area (pixels)
}

# ========== OWL-ViT v2 Configuration ==========
OWL_CONFIG = {
    'model_name': 'google/owlv2-base-patch16-ensemble',
    'confidence_thresholds': [0.05, 0.15, 0.25],  # Multiple thresholds for analysis
    'default_threshold': 0.05,  # Low threshold to capture all detections
    'nms_threshold': 0.3,  # Non-maximum suppression
}

# ========== Query Lists for OWL-ViT ==========
OWL_QUERIES = {
    # Layer 1: Super generic (5 queries)
    'generic': [
        'object', 'thing', 'element', 'shape', 'form'
    ],

    # Layer 2: Basic categories (15 queries)
    'basic': [
        'person', 'face', 'human', 'animal', 'creature',
        'plant', 'tree', 'flower', 'building', 'structure',
        'vehicle', 'furniture', 'food', 'tool', 'container'
    ],

    # Layer 3: Art-specific (20 queries)
    'art_specific': [
        'portrait', 'landscape', 'still life', 'figure',
        'sky', 'cloud', 'water', 'mountain', 'forest',
        'house', 'church', 'bridge', 'boat', 'garden',
        'woman', 'man', 'child', 'horse', 'bird', 'flower'
    ],
}

# Combine all queries
ALL_OWL_QUERIES = (
    OWL_QUERIES['generic'] +
    OWL_QUERIES['basic'] +
    OWL_QUERIES['art_specific']
)  # Total: 40 queries

# ========== Detectability Scoring ==========
DETECTABILITY_WEIGHTS = {
    'sam': {
        'stability_score': 0.4,
        'mask_count': 0.3,
        'mask_size': 0.3,
    },
    'owl': {
        'max_confidence': 0.5,
        'detection_count': 0.3,
        'query_response_rate': 0.2,
    },
    'final': {
        'sam_weight': 0.5,
        'owl_weight': 0.5,
    }
}

# Ideal ranges for scoring
IDEAL_RANGES = {
    'mask_count': (3, 20),  # Ideal number of masks per image
    'detection_count': (1, 10),  # Ideal number of detections
    'small_mask_threshold': 0.05,  # Masks < 5% of image area
}

# Detection output directory
DETECTIONS_DIR = os.path.join(OUTPUT_DIR, 'detections')

# NMS (Non-Maximum Suppression) threshold for merging duplicate detections
NMS_THRESHOLD = 0.5

# Maximum detections per image
MAX_DETECTIONS_PER_IMAGE = 50



# ==================== Visualization Configuration ====================
# Figure DPI for saved plots
FIG_DPI = 300

# Colormap for heatmaps
HEATMAP_CMAP = 'viridis'

# Figure size
FIGSIZE = (12, 10)
