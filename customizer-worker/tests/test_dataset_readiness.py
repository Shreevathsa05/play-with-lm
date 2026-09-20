import unittest

from src.auditor import auditor


class DatasetReadinessTest(unittest.TestCase):
    def test_unknown_schema_is_blocked_without_health_grade(self):
        report = auditor.audit([{"foo": "bar"}])
        self.assertNotIn("health_score", report)
        self.assertEqual(report["readiness"]["sft"]["state"], "BLOCKED")

    def test_empty_dataset_is_blocked(self):
        report = auditor.audit([])
        self.assertEqual(report["readiness"]["cpt"]["state"], "BLOCKED")
        self.assertIn("NONEMPTY", report["readiness"]["cpt"]["blockers"])

    def test_schema_controls_objective(self):
        report = auditor.audit([{"text": "hello"}])
        self.assertEqual(report["readiness"]["cpt"]["state"], "READY")
        self.assertEqual(report["readiness"]["sft"]["state"], "BLOCKED")

    def test_unicode_is_observed_not_penalized(self):
        report = auditor.audit([{"text": "你好世界 नमस्ते दुनिया"}])
        self.assertEqual(report["schema_type"], "raw_text")
        self.assertNotIn("SPECIAL_CHARACTERS", {x["rule_id"] for x in report["findings"]})

    def test_prompt_only_is_ready_for_online_objectives(self):
        report = auditor.audit([{"prompt": "What is 2 + 2?", "answer": "4"}])
        self.assertEqual(report["schema_type"], "prompt_only")
        self.assertEqual(report["readiness"]["grpo"]["state"], "READY")


if __name__ == "__main__":
    unittest.main()
