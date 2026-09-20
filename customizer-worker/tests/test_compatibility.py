import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_modules.compatibility import (
    CompatibilityGate,
    GateDecision,
    estimate_capacity,
    get_recipe_spec,
    parse_param_billions,
    suggest_fallback,
)


class TestRecipesAndEstimates(unittest.TestCase):
    def test_recipe_spec_qlora(self):
        spec = get_recipe_spec("qlora_4bit")
        self.assertTrue(spec.load_in_4bit)
        self.assertTrue(spec.use_peft)

    def test_unknown_recipe(self):
        with self.assertRaises(ValueError):
            get_recipe_spec("dpo")

    def test_parse_params(self):
        self.assertAlmostEqual(parse_param_billions("270m"), 0.27)
        self.assertAlmostEqual(parse_param_billions("1b"), 1.0)
        self.assertAlmostEqual(parse_param_billions(0.5), 0.5)

    def test_estimate_scales_with_recipe(self):
        full = estimate_capacity("fullparams", 1.0)
        qlora = estimate_capacity("qlora_4bit", 1.0)
        self.assertGreater(full.peak_vram_mb, qlora.peak_vram_mb)

    def test_suggest_fallback(self):
        self.assertEqual(suggest_fallback("fullparams"), "embedding")
        self.assertEqual(suggest_fallback("qlora_4bit"), None)


class TestCompatibilityGate(unittest.TestCase):
    def setUp(self):
        self.gate = CompatibilityGate(machine_max_vram_mb=8192, machine_max_ram_mb=32768)

    def test_accept_small_qlora(self):
        result = self.gate.evaluate(
            base_model_id="google/gemma-3-270m-it",
            recipe="qlora_4bit",
            model_params_b="270m",
            free_vram_mb=8000,
            free_ram_mb=16000,
        )
        self.assertEqual(result.decision, GateDecision.ACCEPT)
        self.assertTrue(result.ok_to_spawn)

    def test_reject_vision(self):
        result = self.gate.evaluate(
            base_model_id="llava-hf/llava-1.5",
            recipe="lora",
            model_params_b=7,
        )
        self.assertEqual(result.decision, GateDecision.REJECT)

    def test_reject_unsupported_unsloth(self):
        result = self.gate.evaluate(
            base_model_id="some/model",
            recipe="lora",
            model_params_b=1,
            unsloth_supported=False,
        )
        self.assertEqual(result.decision, GateDecision.REJECT)

    def test_wait_when_busy(self):
        result = self.gate.evaluate(
            base_model_id="google/gemma-3-270m-it",
            recipe="qlora_4bit",
            model_params_b="270m",
            free_vram_mb=1,
            free_ram_mb=16000,
        )
        self.assertEqual(result.decision, GateDecision.WAIT)

    def test_reject_or_suggest_over_machine_max(self):
        tiny = CompatibilityGate(machine_max_vram_mb=64, machine_max_ram_mb=128)
        result = tiny.evaluate(
            base_model_id="meta-llama/Llama-3-8B",
            recipe="fullparams",
            model_params_b=8,
        )
        self.assertIn(result.decision, (GateDecision.REJECT, GateDecision.SUGGEST))

    def test_rejects_models_larger_than_one_billion_parameters(self):
        result = CompatibilityGate(machine_max_vram_mb=16384, machine_max_ram_mb=32768).evaluate(
            base_model_id="meta-llama/Llama-3.1-8B-Instruct",
            recipe="qlora_4bit",
            model_params_b="8b",
        )
        self.assertEqual(result.decision, GateDecision.REJECT)
        self.assertIn("1B limit", result.reason)

    def test_rejects_invalid_model_size_hint(self):
        result = self.gate.evaluate(
            base_model_id="unsloth/gemma-3-270m-it",
            recipe="qlora_4bit",
            model_params_b="smallish",
        )
        self.assertEqual(result.decision, GateDecision.REJECT)
        self.assertIn("model_params_b", result.reason)


if __name__ == "__main__":
    unittest.main()
