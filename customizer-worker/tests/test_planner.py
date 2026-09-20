import unittest

from src.planner import choose_objective, resolve_plan


class PlannerTest(unittest.TestCase):
    def test_schema_objective_rules(self):
        self.assertEqual(choose_objective("raw_text", "auto"), "cpt")
        self.assertEqual(choose_objective("sft", "auto"), "sft")
        self.assertEqual(choose_objective("paired_preference", "auto"), "dpo")
        self.assertEqual(choose_objective("binary_preference", "auto"), "kto")
        self.assertEqual(choose_objective("raw_text", "train_from_scratch"), "pretrain")
        self.assertIsNone(choose_objective("prompt_only", "auto"))
        self.assertEqual(choose_objective("prompt_only", "verifiable_rl", {"verified_reward"}), "grpo")
        self.assertIsNone(choose_objective("sft", "adapt_domain"))

    def test_blocked_readiness_never_launches(self):
        result = resolve_plan(
            audit_report={"schema_type": "raw_text", "readiness": {"cpt": {"state": "BLOCKED", "blockers": ["NONEMPTY"]}}},
            request={"goal": "adapt_domain"},
        )
        self.assertEqual(result["state"], "BLOCKED")
        self.assertIsNone(result["plan"])

    def test_ready_plan_is_explicit(self):
        result = resolve_plan(
            audit_report={"schema_type": "sft", "readiness": {"sft": {"state": "READY"}}},
            request={"base_model_id": "model", "recipe": "lora"},
        )
        self.assertEqual(result["state"], "READY")
        self.assertEqual(result["plan"]["objective"], "sft")
        self.assertEqual(result["plan"]["recipe"], "lora")


if __name__ == "__main__":
    unittest.main()
