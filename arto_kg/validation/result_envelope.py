"""
Shape of a validation result.

Two gates decide whether an artwork is kept: style, then object presence.
`status` and `failed_at` report only these gates; all other scores are recorded
in `steps` and never discard an artwork.
"""

from typing import Any, Dict, Optional

RETAINED = "retained"
DISCARDED = "discarded"

GATE_STYLE = "style"
GATE_OBJECT = "object_presence"

# Step keys of the arto-validation/2 schema.
STEP_STYLE = "step1_style"
STEP_OBJECT = "step2_object"
STEP_COLOUR = "step3_color"
STEP_SIZE = "step4_size"
STEP_STATE = "step5_state"
STEP_SPATIAL = "step6_spatial_relations"      # spatial: geometry + VLM channels
STEP_SEMANTIC = "step6_semantic_relations"    # semantic: VLM
STEP_ALIGNMENT = "step7_alignment"

# Schema identifier written into every result.
SCHEMA = "arto-validation/2"


def discarded(artwork_id: str, image_path: str, failed_at: str,
              steps: Dict[str, Any], provenance: Optional[Dict[str, Any]] = None,
              measurements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A result that failed a gate.

    The steps that ran are kept as evidence for the discard.
    """
    return {
        "schema": SCHEMA,
        "artwork_id": artwork_id,
        "image_path": image_path,
        "status": DISCARDED,
        "failed_at": failed_at,
        "steps": steps,
        "measurements": measurements or {"objects": []},
        "provenance": provenance or {},
    }


def retained(artwork_id: str, image_path: str, steps: Dict[str, Any],
             measurements: Dict[str, Any],
             provenance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """A result that passed both gates. Scores are recorded; they do not decide retention."""
    return {
        "schema": SCHEMA,
        "artwork_id": artwork_id,
        "image_path": image_path,
        "status": RETAINED,
        "failed_at": None,
        "steps": steps,
        "measurements": measurements,
        "provenance": provenance or {},
    }


def is_complete(result: Any) -> bool:
    """Whether a loaded result holds validation steps and is not marked incomplete.

    An empty file, `{}`, or a result from a run without a VLM is not complete.
    """
    return (isinstance(result, dict) and isinstance(result.get("steps"), dict)
            and bool(result["steps"]) and not result.get("incomplete"))


def declares_schema(result: Dict[str, Any]) -> bool:
    """Whether a result carries an arto-validation/ schema identifier."""
    return str(result.get("schema", "")).startswith("arto-validation/")
