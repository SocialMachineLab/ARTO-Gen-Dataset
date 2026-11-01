"""
Unified Output Parser
Automatically detect model type and use corresponding parser
"""

import logging
from typing import Dict, Any, Optional, List
from .base_parser import BaseOutputParser
from .gpt_oss_parser import GPTOSSParser
from .qwen_parser import QwenParser


# Typographic punctuation the LLM emits, mapped to ASCII so specifications and
# the knowledge graph stay plain ASCII.
# Only punctuation is mapped; letters keep their diacritics.
TYPOGRAPHIC_REPLACEMENTS = {
    "\u2010": "-",    # hyphen
    "\u2011": "-",    # non-breaking hyphen
    "\u2012": "-",    # figure dash
    "\u2013": "-",    # en dash
    "\u2014": "-",    # em dash
    "\u2015": "-",    # horizontal bar
    "\u2018": "'",    # left single quote
    "\u2019": "'",    # right single quote
    "\u201a": "'",    # single low quote
    "\u201c": '"',    # left double quote
    "\u201d": '"',    # right double quote
    "\u201e": '"',    # double low quote
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",    # non-breaking space
    "\u2032": "'",    # prime
    "\u2033": '"',    # double prime
}


def normalise_typography(value: Any) -> Any:
    """Replace typographic punctuation with ASCII throughout a nested structure.

    `raw_output` is left untouched so it always records the model's exact reply.
    """
    if isinstance(value, str):
        for source, target in TYPOGRAPHIC_REPLACEMENTS.items():
            if source in value:
                value = value.replace(source, target)
        return value
    if isinstance(value, list):
        return [normalise_typography(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalise_typography(item) for item in value)
    if isinstance(value, dict):
        return {key: (item if key == "raw_output" else normalise_typography(item))
                for key, item in value.items()}
    return value


class UnifiedOutputParser:
    """Unified output parser, automatically adapts to different model formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all parsers
        self.parsers = {
            'gpt_oss': GPTOSSParser(),
            'qwen': QwenParser(),
        }

        # Priority order, limited to registered parsers so an unregistered
        # name is never looked up.
        self.parser_priority = [name for name in ('gpt_oss', 'qwen')
                                if name in self.parsers]

    def detect_model_type(self, raw_output: str, model_name: Optional[str] = None) -> str:
        """Detect the output's model type.

        The model name decides when given. Otherwise the first parser that
        recognises the format is used; a reply no parser recognises is read as
        a plain answer by the Qwen parser.
        """

        # If model name is explicitly specified, use it first
        if model_name:
            if 'gpt-oss' in model_name.lower() or 'gptoss' in model_name.lower():
                return 'gpt_oss'
            elif 'qwen' in model_name.lower():
                return 'qwen'

        # Auto-detect format, over registered parsers only
        for parser_type in self.parser_priority:
            parser = self.parsers[parser_type]
            if parser.can_parse(raw_output):
                return parser_type

        # Unrecognised format: a plain answer, read by the Qwen parser.
        return 'qwen'
    
    def parse(self, raw_output: str, model_name: Optional[str] = None, 
              fallback: bool = True) -> Dict[str, Any]:
        """Unified parsing interface"""
        
        if not raw_output or not raw_output.strip():
            return self._create_empty_result(raw_output, "Empty input")
        
        # Detect model type
        detected_type = self.detect_model_type(raw_output, model_name)

        # No parser for this type: return an error result; parsers of other
        # model types are not tried.
        if detected_type not in self.parsers:
            self.logger.error(
                f"No parser registered for '{detected_type}' "
                f"(have: {sorted(self.parsers)}). The output is left unparsed "
                "rather than read with a parser built for another model.")
            return self._create_error_result(
                raw_output,
                f"No parser registered for model type '{detected_type}'",
                detected_type)

        try:
            # Use corresponding parser
            parser = self.parsers[detected_type]
            result = parser.parse(raw_output)
            
            # Add detection info
            result['detected_model_type'] = detected_type
            result['specified_model_name'] = model_name
            
            # Check parsing quality
            if result['parsing_details']['status'] == 'failed' and fallback:
                return self._try_fallback_parsing(raw_output, detected_type, model_name)

            # All LLM text passes through here, so typography is normalised once.
            return normalise_typography(result)
            
        except Exception as e:
            self.logger.error(f"Parsing failed with {detected_type} parser: {e}")
            
            if fallback:
                return self._try_fallback_parsing(raw_output, detected_type, model_name)
            else:
                return self._create_error_result(raw_output, str(e), detected_type)
    
    def _try_fallback_parsing(self, raw_output: str, failed_type: str, 
                             model_name: Optional[str]) -> Dict[str, Any]:
        """Try using other parsers as fallback"""
        
        for parser_type in self.parser_priority:
            if parser_type == failed_type:
                continue
            # Only a parser that recognises the format may read it, so a reply
            # one parser judged unanswered is not mined by another.
            if not self.parsers[parser_type].can_parse(raw_output):
                continue

            try:
                parser = self.parsers[parser_type]
                result = parser.parse(raw_output)
                
                # Mark as fallback parsing
                result['parsing_details']['is_fallback'] = True
                result['parsing_details']['original_failed_type'] = failed_type
                result['detected_model_type'] = parser_type
                result['specified_model_name'] = model_name
                
                if result['parsing_details']['status'] in ['success', 'partial']:
                    self.logger.info(f"Fallback parsing successful with {parser_type}")
                    return normalise_typography(result)
                    
            except Exception as e:
                self.logger.warning(f"Fallback parsing failed with {parser_type}: {e}")
                continue
        
        # All parsers failed
        return self._create_error_result(raw_output, "All parsers failed", failed_type)
    
    def _create_empty_result(self, raw_output: str, error: str) -> Dict[str, Any]:
        """Create result for empty input"""
        return {
            'raw_output': raw_output,
            'analysis': None,
            'final': None,
            'json_data': None,
            'format_type': 'empty',
            'model_type': 'unknown',
            'detected_model_type': 'unknown',
            'parsing_details': {
                'status': 'failed',
                'raw_content_length': len(raw_output or ''),
                'parsing_error': error
            }
        }
    
    def _create_error_result(self, raw_output: str, error: str, 
                           attempted_type: str) -> Dict[str, Any]:
        """Create error result: no answer was found, so `final` is empty."""
        return {
            'raw_output': raw_output,
            'analysis': None,
            'final': None,
            'json_data': None,
            'format_type': 'error',
            'model_type': attempted_type,
            'detected_model_type': attempted_type,
            'parsing_details': {
                'status': 'error',
                'raw_content_length': len(raw_output),
                'parsing_error': error
            }
        }
    
    def get_parser_info(self) -> Dict[str, Any]:
        """Get parser info"""
        return {
            'available_parsers': list(self.parsers.keys()),
            'parser_priority': self.parser_priority,
            'supported_formats': {
                parser_type: parser.supported_formats 
                for parser_type, parser in self.parsers.items()
            }
        }