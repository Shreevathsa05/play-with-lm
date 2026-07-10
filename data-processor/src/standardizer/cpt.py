import json
import pandas as pd
import io
from typing import List, Dict

class CPTStandardizer:
    
    @classmethod
    def standardize_json(cls, data: List[Dict]) -> List[Dict]:
        """Ensures JSON data has a standard 'text' field."""
        if not data: return []
        keys = data[0].keys()
        if "text" in keys:
            return data
            
        # Try to infer text field
        possible_text_keys = ["content", "document", "doc", "body"]
        text_key = next((k for k in possible_text_keys if k in keys), None)
        
        if text_key:
            return [{"text": item[text_key]} for item in data if text_key in item]
            
        raise ValueError("Unsupported CPT JSON format. Need a 'text' or 'content' field.")

    @classmethod
    def standardize_raw_text(cls, raw_text: str) -> List[Dict]:
        """Splits raw text into manageable chunks (documents)."""
        # For simplicity, split by double newlines or a chunk size.
        docs = [chunk.strip() for chunk in raw_text.split('\n\n') if chunk.strip()]
        return [{"text": doc} for doc in docs]

    @classmethod
    def standardize_csv(cls, csv_bytes: bytes) -> List[Dict]:
        """Parses CSV and extracts 'text' column."""
        df = pd.read_csv(io.BytesIO(csv_bytes))
        columns = [c.lower() for c in df.columns]
        
        if "text" in columns:
            text_col = df.columns[columns.index("text")]
        elif "content" in columns:
            text_col = df.columns[columns.index("content")]
        else:
            raise ValueError("CSV must contain a 'text' or 'content' column for CPT datasets.")
            
        docs = df[text_col].dropna().tolist()
        return [{"text": str(doc)} for doc in docs]
