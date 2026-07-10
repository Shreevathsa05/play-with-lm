from typing import List, Dict, Any
from .base import BaseAnalyzer
import re

class PrivacyAnalyzer(BaseAnalyzer):
    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {}
            
        email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
        
        emails_found = 0
        ips_found = 0
        
        def check_text(text: str):
            nonlocal emails_found, ips_found
            if not text: return
            emails_found += len(email_pattern.findall(text))
            ips_found += len(ip_pattern.findall(text))

        for item in dataset:
            if "conversations" in item:
                for msg in item["conversations"]:
                    check_text(msg.get("value", ""))
            elif "text" in item:
                check_text(item["text"])
            elif "anchor" in item:
                check_text(item["anchor"])
                check_text(item["positive"])
                check_text(item["negative"])
            elif "text1" in item:
                check_text(item["text1"])
                check_text(item["text2"])
                
        return {
            "pii_emails_detected": emails_found,
            "pii_ips_detected": ips_found,
            "has_pii": (emails_found + ips_found) > 0
        }
