from typing import List, Dict, Any
from .base import BaseAnalyzer
import json

class DuplicateAndMissingAnalyzer(BaseAnalyzer):
    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {}
            
        missing_count = 0
        seen = set()
        exact_duplicates = 0
        
        for item in dataset:
            # Check missing values
            has_missing = False
            if "conversations" in item:
                if not item["conversations"]:
                    has_missing = True
                for msg in item["conversations"]:
                    if not msg.get("value"):
                        has_missing = True
            elif "text" in item:
                if not item["text"]:
                    has_missing = True
            elif "anchor" in item:
                if not item["anchor"] or not item["positive"] or not item["negative"]:
                    has_missing = True
            elif "text1" in item:
                if not item["text1"] or not item["text2"]:
                    has_missing = True
                    
            if has_missing:
                missing_count += 1
                
            # Exact duplicate check (using JSON string repr)
            repr_str = json.dumps(item, sort_keys=True)
            if repr_str in seen:
                exact_duplicates += 1
            else:
                seen.add(repr_str)
                
        return {
            "total_records": len(dataset),
            "missing_values_count": missing_count,
            "exact_duplicates_count": exact_duplicates,
            "missing_percentage": round(missing_count / len(dataset) * 100, 2),
            "duplicate_percentage": round(exact_duplicates / len(dataset) * 100, 2)
        }
