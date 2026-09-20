import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_modules.inference_testers import (
    BatchInferenceTester,
    build_eval_report,
    exact_match,
    nll_improvement,
    truncate_prompts,
)


class TestScoring(unittest.TestCase):
    def test_nll_improvement_positive(self):
        self.assertGreater(nll_improvement(2.0, 1.0), 0)

    def test_exact_match(self):
        self.assertTrue(exact_match(" yes ", "yes"))
        self.assertFalse(exact_match("a", "b"))

    def test_truncate(self):
        items, truncated = truncate_prompts(list(range(1500)), 1000)
        self.assertEqual(len(items), 1000)
        self.assertTrue(truncated)

    def test_build_report(self):
        report = build_eval_report(
            job_uuid="j1",
            n_prompts=2,
            truncated=False,
            base_losses=[2.0, 2.0],
            ft_losses=[1.0, 1.0],
            base_accuracy=0.0,
            ft_accuracy=1.0,
            samples=[{"prompt": "p"}],
        )
        self.assertEqual(report["job_uuid"], "j1")
        self.assertAlmostEqual(report["reference_nll"]["mean_improvement"], 0.5)
        self.assertAlmostEqual(report["task_accuracy"]["delta"], 1.0)


class TestBatchInferenceTester(unittest.TestCase):
    def test_run_with_references(self):
        tester = BatchInferenceTester(max_prompts=1000, max_samples=2)
        prompts = [
            {"prompt": "Q1", "reference": "A1"},
            {"prompt": "Q2", "reference": "A2"},
        ]
        report = tester.run(
            job_uuid="job-1",
            prompts=prompts,
            generate_base=lambda p: "wrong",
            generate_ft=lambda p: "A1" if p == "Q1" else "A2",
            reference_loss_base=lambda p, r: 2.0,
            reference_loss_ft=lambda p, r: 1.0,
        )
        self.assertEqual(report["n_prompts"], 2)
        self.assertFalse(report["truncated"])
        self.assertEqual(report["task_accuracy"]["ft"], 1.0)
        self.assertEqual(len(report["samples"]), 2)

    def test_caps_at_1000(self):
        tester = BatchInferenceTester()
        prompts = [{"prompt": f"p{i}"} for i in range(1005)]
        report = tester.run(
            job_uuid="job-2",
            prompts=prompts,
            generate_base=lambda p: "x",
            generate_ft=lambda p: "y",
        )
        self.assertEqual(report["n_prompts"], 1000)
        self.assertTrue(report["truncated"])
        self.assertIsNone(report["reference_nll"]["mean_loss_base"])


if __name__ == "__main__":
    unittest.main()
