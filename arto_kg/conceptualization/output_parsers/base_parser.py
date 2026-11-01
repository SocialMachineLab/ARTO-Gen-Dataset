"""
Base Output Parser Abstract Class
Defines unified interface for all parsers
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple


def _top_level_objects(text: str) -> List[str]:
    """Every top-level {...} span in text, in order.

    Braces inside JSON strings are ignored, so a value like "a {b}" does not
    end an object early.
    """
    spans, depth, start = [], 0, None
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                spans.append(text[start:index + 1])
    return spans


def extract_json(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The answer object in a model reply, or None.

    The last top-level JSON object that parses is taken: a reply that sketches
    an object and then gives its answer ends with the answer. Nested objects
    are part of their parent and never returned on their own.
    """
    if not text:
        return None
    for candidate in reversed(_top_level_objects(text)):
        try:
            data = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict):
            return data
    return None


class BaseOutputParser(ABC):
    """Base Output Parser Abstract Class"""

    def __init__(self, model_name: str = "unknown"):
        self.model_name = model_name
        self.supported_formats = []

    @abstractmethod
    def can_parse(self, raw_output: str) -> bool:
        """Determine if the output format can be parsed"""
        pass

    @abstractmethod
    def parse(self, raw_output: str) -> Dict[str, Any]:
        """Parse raw output, return standardized result"""
        pass

    def _validate_json(self, text: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """Validate JSON format"""
        try:
            data = json.loads(text)
            return True, data, None
        except json.JSONDecodeError as e:
            return False, None, str(e)

    def _extract_json_from_text(self, text: str) -> Optional[Dict]:
        """Extract the answer JSON object from text (see extract_json)."""
        return extract_json(text)

    def _clean_text(self, text: str) -> str:
        """Remove <think> blocks and collapse whitespace."""
        if not text:
            return ""

        # Remove <think> tags and content (support multi-line)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

        # Collapse whitespace runs into single spaces
        text = re.sub(r'\s+', ' ', text.strip())

        return text

    def create_standard_result(self,
                             raw_output: str,
                             analysis: Optional[str] = None,
                             final: Optional[str] = None,
                             json_data: Optional[Dict] = None,
                             format_type: str = "unknown",
                             parsing_status: str = "unknown",
                             error: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized parsing result.

        The model's JSON stays under `json_data`; its keys are not merged into
        the result, so a field the model names `error` or `final` cannot be
        mistaken for the parser's own.
        """
        return {
            'raw_output': raw_output,
            'analysis': analysis,
            'final': final,
            'json_data': json_data,
            'format_type': format_type,
            'model_type': self.model_name,
            'parsing_details': {
                'status': parsing_status,
                'raw_content_length': len(raw_output),
                'parsing_error': error
            }
        }
