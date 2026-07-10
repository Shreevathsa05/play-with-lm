from typing import List, Dict, Any
from .base import BaseAnalyzer
import numpy as np

class StatsAnalyzer(BaseAnalyzer):
    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {}
            
        lengths = []
        turn_counts = []
        
        for item in dataset:
            if "conversations" in item:
                turn_counts.append(len(item["conversations"]))
                lengths.append(sum(len(msg.get("value", "")) for msg in item["conversations"]))
            elif "text" in item:
                lengths.append(len(item["text"]))
            elif "anchor" in item:
                lengths.append(len(item["anchor"]) + len(item["positive"]) + len(item["negative"]))
            elif "text1" in item:
                lengths.append(len(item["text1"]) + len(item["text2"]))
                
        if not lengths:
            return {}
            
        return {
            "char_length_mean": round(float(np.mean(lengths)), 2),
            "char_length_p50": round(float(np.percentile(lengths, 50)), 2),
            "char_length_p95": round(float(np.percentile(lengths, 95)), 2),
            "char_length_min": int(np.min(lengths)),
            "char_length_max": int(np.max(lengths)),
            "avg_turns_per_dialogue": round(float(np.mean(turn_counts)), 2) if turn_counts else 0,
        }
