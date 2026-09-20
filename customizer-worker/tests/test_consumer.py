import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.consumer import handle_job_message
from src.finetuning_modules.compatibility import GateDecision, GateResult
from src.supervisor import SpawnResult, SupervisedRun


class TestConsumer(unittest.TestCase):
    def test_handle_reject_notifies_failed(self):
        superv = MagicMock()
        superv.run_job.return_value = SupervisedRun(
            gate=GateResult(GateDecision.REJECT, "nope", "lora")
        )
        notify = MagicMock()
        handle_job_message(
            {"job_uuid": "abc", "recipe": "lora", "base_model_id": "m"},
            supervisor=superv,
            notify=notify,
        )
        self.assertEqual(notify.call_args_list[0].args[1], "CHECKING")
        self.assertEqual(notify.call_args[0][1], "FAILED")

    def test_handle_success_notifies_completed(self):
        superv = MagicMock()
        superv.run_job.return_value = SupervisedRun(
            gate=GateResult(GateDecision.ACCEPT, "ok", "qlora_4bit"),
            spawn=SpawnResult(returncode=0, stdout="{}", stderr="", pid=1),
            waited_sec=0.5,
        )
        notify = MagicMock()
        handle_job_message(
            b'{"job_uuid":"abc","recipe":"qlora_4bit","base_model_id":"m"}',
            supervisor=superv,
            notify=notify,
        )
        statuses = [call.args[1] for call in notify.call_args_list]
        self.assertEqual(statuses[0], "CHECKING")
        self.assertEqual(notify.call_args[0][1], "COMPLETED")

    def test_missing_job_uuid(self):
        with self.assertRaises(ValueError):
            handle_job_message({"recipe": "lora"}, supervisor=MagicMock(), notify=MagicMock())

    def test_structured_runner_result_reaches_webhook(self):
        superv = MagicMock()
        stdout = '__CUSTOMIZER_RESULT__={"status":"COMPLETED","job_uuid":"abc","export":{"export_dest":"minio://jobs/abc/exports"},"eval":{"n_prompts":1}}'
        superv.run_job.return_value = SupervisedRun(
            gate=GateResult(GateDecision.ACCEPT, "ok", "lora"),
            spawn=SpawnResult(returncode=0, stdout=stdout, stderr="", pid=1),
        )
        notify = MagicMock()
        handle_job_message(
            {"job_uuid": "abc", "recipe": "lora", "base_model_id": "m"},
            supervisor=superv,
            notify=notify,
        )
        report = notify.call_args[0][2]
        self.assertEqual(report["eval"]["n_prompts"], 1)
        self.assertEqual(report["export"]["export_dest"], "minio://jobs/abc/exports")

    def test_structured_failed_result_reaches_webhook(self):
        superv = MagicMock()
        stdout = '__CUSTOMIZER_RESULT__={"status":"FAILED","job_uuid":"abc","export":{"hf_push":{"status":"failed","repo_id":"user/model","url":null,"error":"bad token"}},"eval":null}'
        superv.run_job.return_value = SupervisedRun(
            gate=GateResult(GateDecision.ACCEPT, "ok", "lora"),
            spawn=SpawnResult(returncode=1, stdout=stdout, stderr="", pid=1),
        )
        notify = MagicMock()
        handle_job_message(
            {"job_uuid": "abc", "recipe": "lora", "base_model_id": "m", "dry_run": True},
            supervisor=superv,
            notify=notify,
        )
        report = notify.call_args.args[2]
        self.assertEqual(notify.call_args.args[1], "FAILED")
        self.assertEqual(report["export"]["hf_push"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
