from typing import List, Dict, Any
from .base import BaseAnalyzer

class SchemaAnalyzer(BaseAnalyzer):
    def analyze(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not dataset:
            return {"schema_type": "unknown", "supported_schema": False,
                    "schema_valid": False, "total_records": 0,
                    "inconsistent_rows": 0, "invalid_rows": 0,
                    "valid_required_fields": False, "nonempty": False}

        def detect(row: Dict[str, Any]) -> str:
            keys = set(row)
            if "conversations" in keys or "messages" in keys:
                return "sft"
            if {"prompt", "chosen", "rejected"}.issubset(keys):
                return "paired_preference"
            if {"prompt", "completion", "label"}.issubset(keys):
                return "binary_preference"
            if "prompt" in keys:
                return "prompt_only"
            if {"anchor", "positive", "negative"}.issubset(keys):
                return "embedding_triplet"
            if {"anchor", "positive"}.issubset(keys) or {"text1", "text2"}.issubset(keys):
                return "embedding_pair"
            if "text" in keys:
                return "raw_text"
            return "unknown"

        schema_types = [detect(row) for row in dataset]
        schema_type = schema_types[0]
        inconsistent_rows = sum(kind != schema_type for kind in schema_types)
        invalid_rows = 0
        for row, kind in zip(dataset, schema_types):
            if kind != schema_type:
                invalid_rows += 1
                continue
            if kind == "sft":
                messages = row.get("conversations", row.get("messages"))
                if not isinstance(messages, list) or not messages:
                    invalid_rows += 1
                    continue
                if not all(isinstance(message, dict) and str(message.get("value", message.get("content", ""))).strip()
                           and str(message.get("from", message.get("role", ""))).strip() for message in messages):
                    invalid_rows += 1
            elif kind == "raw_text":
                if not isinstance(row.get("text"), str) or not row["text"].strip():
                    invalid_rows += 1
            elif kind == "paired_preference":
                if any(not str(row.get(key, "")).strip() for key in ("prompt", "chosen", "rejected")):
                    invalid_rows += 1
            elif kind == "binary_preference":
                if not str(row.get("prompt", "")).strip() or not str(row.get("completion", "")).strip() or not isinstance(row.get("label"), bool):
                    invalid_rows += 1
            elif kind == "prompt_only":
                if not str(row.get("prompt", "")).strip():
                    invalid_rows += 1
            elif kind == "embedding_triplet":
                if any(not str(row.get(key, "")).strip() for key in ("anchor", "positive", "negative")):
                    invalid_rows += 1
            elif kind == "embedding_pair":
                left, right = ("anchor", "positive") if "anchor" in row else ("text1", "text2")
                if any(not str(row.get(key, "")).strip() for key in (left, right)):
                    invalid_rows += 1

        supported = schema_type != "unknown"
        return {
            "schema_type": schema_type,
            "supported_schema": supported,
            "inconsistent_rows": inconsistent_rows,
            "invalid_rows": invalid_rows,
            "schema_valid": supported and inconsistent_rows == 0 and invalid_rows == 0,
            "valid_required_fields": supported and invalid_rows == 0,
            "nonempty": bool(dataset) and invalid_rows == 0,
            "total_records": len(dataset),
        }
