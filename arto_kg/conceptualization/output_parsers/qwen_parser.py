"""
Qwen Output Parser
Qwen3 replies: an optional <think>...</think> block, then the answer
"""

import re
from typing import Dict, Any

from .base_parser import BaseOutputParser


class QwenParser(BaseOutputParser):
    """Qwen3 Output Parser"""

    def __init__(self):
        super().__init__("qwen")
        self.supported_formats = ["think_then_answer", "answer"]

    def can_parse(self, raw_output: str) -> bool:
        """A reply with a <think> block, or one that starts with a JSON object or a code fence."""
        text = (raw_output or "").lstrip()
        return "<think>" in text or text.startswith("{") or text.startswith("```")

    def parse(self, raw_output: str) -> Dict[str, Any]:
        """Parse a Qwen3 reply.

        The answer is what follows the reasoning block. A reasoning block that
        is opened but never closed means the reply ended inside the reasoning,
        so there is no answer to read.
        """
        text = raw_output or ""
        analysis = None

        closed = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if closed:
            analysis = closed.group(1).strip()
            answer = text[closed.end():]
        elif "<think>" in text:
            return self.create_standard_result(
                raw_output,
                analysis=text.split("<think>", 1)[1].strip(),
                format_type="think_then_answer",
                parsing_status="failed",
                error="Reasoning without a final answer (the reply may have been cut off)"
            )
        else:
            answer = text

        answer = answer.strip()
        json_data = self._extract_json_from_text(answer)
        return self.create_standard_result(
            raw_output,
            analysis=analysis,
            final=answer,
            json_data=json_data,
            format_type="think_then_answer" if analysis is not None else "answer",
            parsing_status="success" if json_data is not None else "failed",
            error=None if json_data is not None else "No JSON object in the answer"
        )
