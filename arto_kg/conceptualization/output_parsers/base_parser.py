"""
Base Output Parser Abstract Class
Defines unified interface for all parsers
"""

import json
import re
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple


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
        """Extract JSON from text"""
        # Find first { to last }
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        # Use stack to find matching }
        stack = []
        end_idx = -1
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                stack.append(char)
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack:
                        end_idx = i
                        break
        
        if end_idx != -1:
            json_str = text[start_idx:end_idx + 1]
            is_valid, data, error = self._validate_json(json_str)
            if is_valid:
                return data
        
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean text, remove excess whitespace and control characters"""
        if not text:
            return ""
        
        # Remove <think> tags and content (support multi-line)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Remove invalid JSON fragments at the beginning
        text = re.sub(r'^{"visual[^}]*}', '', text.strip())
        
        # Clean control characters and excess whitespace
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
        """Create standardized parsing result"""
        
        result = {
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
        
        # If JSON data exists, merge it into the result
        if json_data and isinstance(json_data, dict):
            result.update(json_data)
        
        return result