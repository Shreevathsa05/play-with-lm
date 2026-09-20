"""Contract: Manager queue payload keys match worker expectations."""

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.modules["unsloth"] = MagicMock()
sys.modules["unsloth.chat_templates"] = MagicMock()
sys.modules["trl"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["torch"] = MagicMock()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.finetuning_flows.orchestrator import run_recipe
from src.consumer import handle_job_message
from src.finetuning_modules.compatibility import GateDecision, GateResult
from src.supervisor import SpawnResult, SupervisedRun


def sample_manager_payload(**overrides):
    payload = {
        "job_uuid": "11111111-2222-3333-4444-555555555555",
        "base_model_id": "google/gemma-3-270m-it",
        "recipe": "qlora_4bit",
        "model_params_b": "270m",
        "dataset_minio_uri": "minio://datasets/user-upload.jsonl",
        "export": {
            "format": "lora",
            "minio_destination": "minio://jobs/11111111-2222-3333-4444-555555555555/exports",
            "private": True,
        },
        "eval": {
            "prompts": [{"prompt": "Hello", "reference": "Hello"}],
            "max_prompts": 1000,
        },
        "max_seq_length": 512,
        "batch_size": 1,
        "dry_run": True,
        "unsloth_supported": True,
    }
    payload.update(overrides)
    return payload


class TestManagerWorkerContract(unittest.TestCase):
    def test_dry_run_recipe_from_manager_payload(self):
        result = run_recipe(sample_manager_payload(), scratch_dir=tempfile.mkdtemp())
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["job_uuid"], sample_manager_payload()["job_uuid"])
        self.assertIsNotNone(result["eval"])
        self.assertTrue(result["export"]["dry_run"])

    def test_consumer_completed_webhook_shape(self):
        superv = MagicMock()
        superv.run_job.return_value = SupervisedRun(
            gate=GateResult(GateDecision.ACCEPT, "ok", "qlora_4bit"),
            spawn=SpawnResult(
                returncode=0,
                stdout='{"status": "COMPLETED", "job_uuid": "11111111-2222-3333-4444-555555555555"}\n',
                stderr="",
                pid=9,
            ),
        )
        notify = MagicMock()
        handle_job_message(sample_manager_payload(), supervisor=superv, notify=notify)
        args = notify.call_args[0]
        self.assertEqual(args[0], "11111111-2222-3333-4444-555555555555")
        self.assertEqual(args[1], "COMPLETED")
        self.assertIn("waited_sec", args[2])


if __name__ == "__main__":
    unittest.main()
