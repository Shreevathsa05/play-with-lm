import json
from typing import List, Dict, Any

class SFTStandardizer:
    @staticmethod
    def is_alpaca(data: List[Dict]) -> bool:
        if not data: return False
        keys = data[0].keys()
        return "instruction" in keys and "output" in keys

    @staticmethod
    def is_openai(data: List[Dict]) -> bool:
        if not data: return False
        keys = data[0].keys()
        if "messages" in keys and isinstance(data[0]["messages"], list):
            return True
        return False

    @staticmethod
    def is_sharegpt(data: List[Dict]) -> bool:
        if not data: return False
        keys = data[0].keys()
        if "conversations" in keys and isinstance(data[0]["conversations"], list):
            return True
        return False

    @classmethod
    def standardize(cls, data: List[Dict]) -> List[Dict]:
        """Converts any supported SFT format to ShareGPT format."""
        if cls.is_sharegpt(data):
            return data
        
        if cls.is_alpaca(data):
            return cls._from_alpaca(data)
            
        if cls.is_openai(data):
            return cls._from_openai(data)
            
        raise ValueError("Unsupported SFT Dataset Format. Supported formats: Alpaca, ShareGPT, OpenAI.")

    @staticmethod
    def _from_alpaca(data: List[Dict]) -> List[Dict]:
        standardized = []
        for item in data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            prompt = instruction
            if input_text:
                prompt += f"\n\n{input_text}"
                
            standardized.append({
                "conversations": [
                    {"from": "user", "value": prompt},
                    {"from": "assistant", "value": output}
                ]
            })
        return standardized

    @staticmethod
    def _from_openai(data: List[Dict]) -> List[Dict]:
        standardized = []
        role_map = {
            "user": "user",
            "system": "system",
            "assistant": "assistant",
            "model": "assistant"
        }
        
        for item in data:
            conv = []
            for msg in item.get("messages", []):
                role = role_map.get(msg.get("role", "user"), "user")
                conv.append({
                    "from": role,
                    "value": msg.get("content", "")
                })
            standardized.append({"conversations": conv})
        return standardized
