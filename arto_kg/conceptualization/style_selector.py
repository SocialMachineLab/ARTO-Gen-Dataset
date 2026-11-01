import random
from typing import List, Optional
from arto_kg.conceptualization.utils import setup_logger
from arto_kg.config.model_config import (
    ARTO_KG_STYLES,
    STYLE_COCO_VARIETY,
)


class StyleSelectionError(ValueError):
    """Raised when a requested style cannot be used by the pipeline."""


class StyleSelector:
    """Selects the art style.

    Automatic sampling draws from ARTO_KG_STYLES. Any style with a
    STYLE_COCO_VARIETY table can be requested explicitly; it is used as given.
    """

    def __init__(self, sampling_pool: Optional[List[str]] = None):
        self.logger = setup_logger("style_selector")
        self.sampling_pool = list(sampling_pool) if sampling_pool else list(ARTO_KG_STYLES)
        self.logger.info(
            f"Initialized StyleSelector; sampling pool: {self.sampling_pool}"
        )

    def select_style(self, preference: Optional[str] = None,
                    exclude_styles: Optional[List[str]] = None) -> str:
        """Return the style to generate.

        A specified style is used if it has a variety table, otherwise rejected.
        """
        if preference:
            resolved = self._resolve_preference(preference)
            self.logger.info(f"Using specified style: {resolved}")
            return resolved

        candidates = list(self.sampling_pool)
        if exclude_styles:
            candidates = [style for style in candidates if style not in exclude_styles]

        if not candidates:
            raise StyleSelectionError(
                f"No style candidates left after excluding {exclude_styles}"
            )

        selected_style = random.choice(candidates)
        self.logger.info(f"Selected style: {selected_style}")
        return selected_style

    def _resolve_preference(self, preference: str) -> str:
        """Validate an explicitly requested style.

        A style is usable only if it has a two-layer object vocabulary table.
        """
        if preference in self.sampling_pool:
            return preference

        if preference in STYLE_COCO_VARIETY:
            self.logger.warning(
                f"Style '{preference}' is outside the dataset sampling pool "
                f"{self.sampling_pool} but has an object variety table; using it as requested."
            )
            return preference

        raise StyleSelectionError(
            f"Style '{preference}' has no STYLE_COCO_VARIETY table and cannot be "
            f"generated. Available: {sorted(STYLE_COCO_VARIETY.keys())}"
        )
