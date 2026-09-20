"""Deterministic, dependency-free training plan selection.

This chooses an objective from an audited schema. It does not estimate model
quality and it never silently changes an explicit user choice.
"""

from __future__ import annotations

from typing import Any


OBJECTIVE_SCHEMAS = {
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


def choose_objective(schema_type: str, goal: str | None = None,
                     available: set[str] | None = None) -> str | None:
    """Return an objective only when its data shape and prerequisites agree."""
    schema = str(schema_type or "unknown").strip().lower()
    requested = str(goal or "auto").strip().lower()
    available = available or set()
    if requested in {"train_from_scratch", "pretrain"}:
        return "pretrain" if schema == "raw_text" else None
    if requested in {"train_reward_model", "reward_model"}:
        return "reward_model" if schema == "paired_preference" else None
    if requested == "ppo_rlhf":
        return "ppo" if schema == "prompt_only" and "ppo_models" in available else None
    if requested == "verifiable_rl":
        return "grpo" if schema == "prompt_only" and "verified_reward" in available else None
    if requested in {"adapt_domain", "cpt"}:
        return "cpt" if schema == "raw_text" else None
    if requested in {"follow_instructions", "sft"}:
        return "sft" if schema == "sft" else None
    if requested in {"align_preferences", "dpo"}:
        return "dpo" if schema == "paired_preference" else None
    if requested == "kto":
        return "kto" if schema == "binary_preference" else None
    if requested in {"retrieval", "embedding"}:
        return "embedding" if schema in OBJECTIVE_SCHEMAS["embedding"] else None
    if requested not in {"auto", ""}:
        return None
    for objective in ("cpt", "sft", "dpo", "kto", "embedding"):
        if schema in OBJECTIVE_SCHEMAS[objective]:
            return objective
    return None


def resolve_plan(*, audit_report: dict[str, Any], request: dict[str, Any],
                 available: set[str] | None = None) -> dict[str, Any]:
    """Build a stable plan result from report + request; no model calls."""
    schema = audit_report.get("schema_type", "unknown")
    goal = request.get("goal") or request.get("objective") or "auto"
    objective = choose_objective(schema, goal, available)
    if objective is None:
        return {"state": "NEEDS_INPUT", "plan": None, "reasons": [
            {"rule_id": "OBJECTIVE_SCHEMA", "message":
             f"No supported objective matches schema '{schema}' and goal '{goal}'."}
        ]}
    readiness = (audit_report.get("readiness") or {}).get(objective, {})
    if readiness.get("state") == "BLOCKED":
        return {"state": "BLOCKED", "plan": None, "reasons": [
            {"rule_id": key, "message": "Dataset readiness requirement is blocked."}
            for key in readiness.get("blockers", [])
        ]}
    recipe = str(request.get("recipe") or "qlora_4bit").strip().lower()
    plan = {
        "policy_version": "auto-v1",
        "mode": "manual" if request.get("mode") == "manual" else "auto",
        "objective": objective,
        "recipe": recipe,
        "model": {"id": request.get("base_model_id") or request.get("baseModelId")},
        "train": dict(request.get("train") or {}),
        "reasons": [{"rule_id": "SCHEMA_OBJECTIVE", "message":
                     f"Selected {objective} from audited schema {schema}."}],
    }
    return {"state": "READY", "plan": plan, "reasons": plan["reasons"]}


__all__ = ["OBJECTIVE_SCHEMAS", "choose_objective", "resolve_plan"]
