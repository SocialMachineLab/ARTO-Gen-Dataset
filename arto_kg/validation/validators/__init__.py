"""Validators for each Stage 3 validation dimension.

Object detection, size, spatial relations, colour and text-image alignment,
plus the VLM-based style, state, semantic and spatial validators.
"""

from .vlm_style_validator import VLMStyleValidator
from .vlm_state_validator import VLMStateValidator
from .vlm_semantic_validator import VLMSemanticValidator
from .vlm_spatial_validator import VLMSpatialValidator
from .object_detection_validator import ObjectDetectionValidator
from .size_validator import SizeValidator
from .spatial_validator import SpatialValidator
from .alignment_validator import AlignmentValidator
from .color_validator import ColorValidator

__all__ = [
    'VLMStyleValidator',
    'VLMStateValidator',
    'VLMSemanticValidator',
    'VLMSpatialValidator',
    'ObjectDetectionValidator',
    'SizeValidator',
    'SpatialValidator',
    'AlignmentValidator',
    'ColorValidator',
]
