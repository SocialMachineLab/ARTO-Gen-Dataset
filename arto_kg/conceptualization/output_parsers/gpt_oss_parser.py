"""
GPT-OSS Output Format Parser
Supports GPT-OSS multi-channel output format
"""

import re
import json
from typing import Dict, Any, Optional, List
from .base_parser import BaseOutputParser


class GPTOSSParser(BaseOutputParser):
    """GPT-OSS Output Parser"""
    
    def __init__(self):
        super().__init__("gpt_oss")
        self.supported_formats = ["complete_channels", "simplified", "assistantfinal"]
        
        # Regex patterns for GPT-OSS format
        self.patterns = {
            'analysis_channel': r'<\|start\|>assistant<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',
            'final_channel': r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>)',
            'tool_call': r'<\|start\|>assistant to=functions\.(\w+)<\|channel\|>commentary.*?<\|message\|>(.*?)<\|call\|>',
            'assistantfinal_split': r'(.*)assistantfinal(.*)',
        }
    
    def can_parse(self, raw_output: str) -> bool:
        """Determine if the output format can be parsed"""
        indicators = [
            '<|channel|>',
            '<|start|>assistant',
            'assistantfinal',
            '<|message|>',
            '<|end|>',
            'to=functions.'
        ]
        
        return any(indicator in raw_output for indicator in indicators)
    
    def parse(self, raw_output: str) -> Dict[str, Any]:
        """Parse GPT-OSS output"""
        
        # Clean input
        clean_output = self._clean_text(raw_output)
        
        # 1. Try complete channel format
        complete_result = self._parse_complete_channels(clean_output)
        if complete_result['parsing_details']['status'] == 'success':
            return complete_result
        
        # 2. Try simplified assistantfinal format
        simplified_result = self._parse_assistantfinal_format(clean_output)
        if simplified_result['parsing_details']['status'] == 'success':
            return simplified_result
        
        # 3. Fallback to raw text
        return self._parse_fallback(raw_output)
    
    def _parse_complete_channels(self, clean_output: str) -> Dict[str, Any]:
        """Parse complete multi-channel format"""
        
        analysis_matches = re.findall(self.patterns['analysis_channel'], clean_output, re.DOTALL)
        final_matches = re.findall(self.patterns['final_channel'], clean_output, re.DOTALL)
        tool_matches = re.findall(self.patterns['tool_call'], clean_output, re.DOTALL)
        
        if not (analysis_matches or final_matches):
            return self.create_standard_result(
                clean_output,
                format_type="complete_channels",
                parsing_status="failed",
                error="No valid channels found"
            )
        
        # Extract content
        analysis_content = analysis_matches[0].strip() if analysis_matches else None
        final_content = final_matches[0].strip() if final_matches else None
        
        # Parse JSON data
        json_data = None
        if final_content:
            json_data = self._extract_json_from_text(final_content)
        
        # Handle tool calls
        tool_calls = []
        if tool_matches:
            for tool_name, tool_args in tool_matches:
                try:
                    args_data = json.loads(tool_args)
                    tool_calls.append({
                        'name': tool_name,
                        'arguments': args_data
                    })
                except:
                    pass
        
        result = self.create_standard_result(
            clean_output,
            analysis=analysis_content,
            final=final_content,
            json_data=json_data,
            format_type="complete_channels",
            parsing_status="success"
        )
        
        if tool_calls:
            result['tool_calls'] = tool_calls
            
        return result
    
    def _parse_assistantfinal_format(self, clean_output: str) -> Dict[str, Any]:
        """Parse assistantfinal simplified format"""
        
        match = re.match(self.patterns['assistantfinal_split'], clean_output, re.DOTALL)
        if not match:
            return self.create_standard_result(
                clean_output,
                format_type="assistantfinal",
                parsing_status="failed",
                error="No assistantfinal delimiter found"
            )
        
        analysis_content = match.group(1).strip() if match.group(1) else None
        final_content = match.group(2).strip() if match.group(2) else None
        
        # Clean analysis content, remove format identifiers
        if analysis_content:
            analysis_content = re.sub(r'^analysis\s*', '', analysis_content)
        
        # Parse JSON data
        json_data = None
        if final_content and final_content.startswith('{'):
            json_data = self._extract_json_from_text(final_content)
        
        return self.create_standard_result(
            clean_output,
            analysis=analysis_content,
            final=final_content,
            json_data=json_data,
            format_type="assistantfinal",
            parsing_status="success"
        )
    
    def _parse_fallback(self, raw_output: str) -> Dict[str, Any]:
        """Fallback parsing method"""
        
        # Try to extract JSON directly
        json_data = self._extract_json_from_text(raw_output)
        
        return self.create_standard_result(
            raw_output,
            analysis=None,
            final=raw_output,
            json_data=json_data,
            format_type="fallback",
            parsing_status="fallback"
        )
    
    def _reconstruct_from_analysis(self, analysis_content: str, failed_final: str) -> Optional[Dict]:
        """Reconstruct incomplete JSON from analysis content"""
        if not analysis_content:
            return None
        
        # Look for JSON structure description in analysis
        json_fragments = re.findall(r'\{[^}]*\}', analysis_content)
        for fragment in json_fragments:
            try:
                data = json.loads(fragment)
                if isinstance(data, dict) and len(data) > 1:
                    return data
            except:
                continue
        
        return None