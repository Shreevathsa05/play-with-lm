from typing import List, Dict, Any
from .base import BaseAnalyzer
import re

class QualityAnalyzer(BaseAnalyzer):
    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {}
            
        total_chars = 0
        special_chars = 0
        uppercase_chars = 0
        empty_lines = 0
        
        # Simple token estimation (split by space)
        estimated_tokens = 0
        
        def process_text(text: str):
            nonlocal total_chars, special_chars, uppercase_chars, empty_lines, estimated_tokens
            if not text: return
            total_chars += len(text)
            special_chars += len(re.findall(r'[^a-zA-Z0-9\s.,!?\'"-]', text))
            uppercase_chars += sum(1 for c in text if c.isupper())
            empty_lines += text.count('\n\n')
            estimated_tokens += len(text.split())
            
        for item in dataset:
            if "conversations" in item:
                for msg in item["conversations"]:
                    process_text(msg.get("value", ""))
            elif "text" in item:
                process_text(item["text"])
            elif "anchor" in item:
                process_text(item["anchor"])
                process_text(item["positive"])
                process_text(item["negative"])
            elif "text1" in item:
                process_text(item["text1"])
                process_text(item["text2"])
                
        if total_chars == 0:
            return {}
            
        return {
            "estimated_total_tokens": estimated_tokens,
            "special_char_ratio": round(special_chars / total_chars, 4),
            "uppercase_ratio": round(uppercase_chars / total_chars, 4),
            "empty_lines_count": empty_lines
        }
