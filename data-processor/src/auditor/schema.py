from typing import List, Dict, Any
from .base import BaseAnalyzer

class SchemaAnalyzer(BaseAnalyzer):
    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {"schema_valid": False, "missing_keys": []}
            
        # We assume standardizer has run, so we expect specific schemas.
        # Let's detect schema based on first item
        keys = dataset[0].keys()
        
        schema_type = "unknown"
        if "conversations" in keys:
            schema_type = "sft_sharegpt"
        elif "text" in keys:
            schema_type = "cpt"
        elif "anchor" in keys:
            schema_type = "embedding_triplet"
        elif "text1" in keys:
            schema_type = "embedding_pair"
            
        inconsistent_rows = 0
        for i, row in enumerate(dataset):
            if set(row.keys()) != set(keys):
                inconsistent_rows += 1
                
        return {
            "schema_type": schema_type,
            "inconsistent_rows": inconsistent_rows,
            "schema_valid": inconsistent_rows == 0,
            "expected_keys": list(keys)
        }
