"""
Unified Output Parser
Automatically detect model type and use corresponding parser
"""

import logging
from typing import Dict, Any, Optional, List
from .base_parser import BaseOutputParser
from .gpt_oss_parser import GPTOSSParser


class UnifiedOutputParser:
    """Unified output parser, automatically adapts to different model formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize all parsers
        self.parsers = {
            'gpt_oss': GPTOSSParser()
        }
        
        # Parser priority (GPT-OSS format is more specific, check first)
        self.parser_priority = ['gpt_oss', 'qwen']
        
    def detect_model_type(self, raw_output: str, model_name: Optional[str] = None) -> str:
        """Detect output model type"""
        
        # If model name is explicitly specified, use it first
        if model_name:
            if 'gpt-oss' in model_name.lower() or 'gptoss' in model_name.lower():
                return 'gpt_oss'
            elif 'qwen' in model_name.lower():
                return 'qwen'
        
        # Auto-detect format
        for parser_type in self.parser_priority:
            parser = self.parsers[parser_type]
            if parser.can_parse(raw_output):
                return parser_type
        
        # Default to Qwen parser
        return 'qwen'
    
    def parse(self, raw_output: str, model_name: Optional[str] = None, 
              fallback: bool = True) -> Dict[str, Any]:
        """Unified parsing interface"""
        
        if not raw_output or not raw_output.strip():
            return self._create_empty_result(raw_output, "Empty input")
        
        # Detect model type
        detected_type = self.detect_model_type(raw_output, model_name)
        
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
            
            return result
            
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
                    return result
                    
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
        """Create error result"""
        return {
            'raw_output': raw_output,
            'analysis': None,
            'final': raw_output,
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