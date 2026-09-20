from typing import Dict, Any

class ScoringEngine:
    def evaluate(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        findings = []
        if not metrics.get("supported_schema", False):
            findings.append({"rule_id": "SUPPORTED_SCHEMA", "severity": "BLOCKER",
                             "message": "Rows do not match a supported training schema."})
        if metrics.get("inconsistent_rows", 0):
            findings.append({"rule_id": "CONSISTENT_SCHEMA", "severity": "BLOCKER",
                             "message": f"{metrics['inconsistent_rows']} rows use a different schema."})
        if metrics.get("invalid_rows", 0):
            findings.append({"rule_id": "REQUIRED_FIELDS", "severity": "BLOCKER",
                             "message": f"{metrics['invalid_rows']} rows have missing or invalid required fields."})
        if metrics.get("total_records", 0) == 0:
            findings.append({"rule_id": "NONEMPTY", "severity": "BLOCKER", "message": "Dataset has no records."})
        if metrics.get("has_pii"):
            findings.append({"rule_id": "PII_REVIEW", "severity": "REVIEW",
                             "message": "Potential email or IP data was detected; review handling policy."})

        schema = metrics.get("schema_type", "unknown")
        common = ["SUPPORTED_SCHEMA", "CONSISTENT_SCHEMA", "REQUIRED_FIELDS", "NONEMPTY"]
        required = {
            "raw_text": common,
            "sft": common,
            "paired_preference": common + ["PREFERENCE_COLUMNS"],
            "binary_preference": common + ["BOOLEAN_LABEL"],
            "prompt_only": common,
            "embedding_pair": common,
            "embedding_triplet": common,
        }.get(schema, ["SUPPORTED_SCHEMA", "NONEMPTY"])
        finding_ids = {finding["rule_id"] for finding in findings}
        objective_schemas = {
            "pretrain": {"raw_text"},
            "cpt": {"raw_text"},
            "sft": {"sft"},
            "dpo": {"paired_preference"},
            "kto": {"binary_preference"},
            "reward_model": {"paired_preference"},
            "ppo": {"prompt_only"},
            "grpo": {"prompt_only"},
            "embedding": {"embedding_pair", "embedding_triplet"},
        }
        readiness = {}
        for objective in ("pretrain", "cpt", "sft", "dpo", "kto", "reward_model", "ppo", "grpo", "embedding"):
            blockers = [rule for rule in required if rule in finding_ids]
            if schema not in objective_schemas[objective]:
                blockers.append("OBJECTIVE_SCHEMA")
            readiness[objective] = {
                "state": "BLOCKED" if blockers else "NEEDS_REVIEW" if metrics.get("has_pii") else "READY",
                "blockers": blockers,
                "reviews": ["PII_REVIEW"] if metrics.get("has_pii") else [],
            }
        metrics.pop("health_score", None)
        metrics.pop("grade", None)
        metrics["report_version"] = 2
        metrics["findings"] = findings
        metrics["readiness"] = readiness
        metrics["recommendations"] = [finding["message"] for finding in findings]
        return metrics
