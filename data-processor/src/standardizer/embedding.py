import json
from typing import List, Dict

class EmbeddingStandardizer:
    @classmethod
    def is_triplet(cls, data: List[Dict]) -> bool:
        if not data: return False
        keys = data[0].keys()
        return "anchor" in keys and "positive" in keys and "negative" in keys
        
    @classmethod
    def is_pair(cls, data: List[Dict]) -> bool:
        if not data: return False
        keys = data[0].keys()
        return "text1" in keys and "text2" in keys

    @classmethod
    def standardize(cls, data: List[Dict]) -> List[Dict]:
        """Standardizes embedding dataset to either Triplet or Pair format."""
        if cls.is_triplet(data):
            return data
            
        if cls.is_pair(data):
            return data
            
        # Try to infer format
        if not data: return []
        keys = data[0].keys()
        
        # Infer Triplet
        if "query" in keys and "pos" in keys and "neg" in keys:
            return [{"anchor": item["query"], "positive": item["pos"], "negative": item["neg"]} for item in data]
            
        # Infer Pair with label
        if "sentence1" in keys and "sentence2" in keys:
            return [{"text1": item["sentence1"], "text2": item["sentence2"], "label": item.get("label", item.get("score", 1.0))} for item in data]
            
        raise ValueError("Unsupported Embedding Dataset Format. Need anchor/positive/negative or text1/text2.")
