import io
import os
import sys
import unittest
import importlib.util
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

SPEC = importlib.util.spec_from_file_location("job_spawner", os.path.join(ROOT, "src", "supervisor", "job_spawner.py"))
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
JobSpawner = MODULE.JobSpawner


class JobSpawnerTest(unittest.TestCase):
    def test_cancellation_terminates_child(self):
        spawner = JobSpawner(job_runner_module="timeit")
        result = spawner.spawn(
            {"job_uuid": "cancel-test"},
            extra_args=["-n", "100000000", "pass"],
            heartbeat_interval_sec=0.01,
            should_cancel=lambda: True,
        )
        self.assertTrue(result.cancelled)

    def test_spawn_decodes_utf8_progress_bars(self):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.returncode = 0
        mock_proc.pid = 1
        mock_proc.stdout = io.BytesIO(b"")
        mock_proc.stderr = io.BytesIO("Loading weights: \u258d\n".encode("utf-8"))
        with patch.object(MODULE.subprocess, "Popen", return_value=mock_proc) as popen:
            result = JobSpawner().spawn({"job_uuid": "utf8-pipe"})
        env = popen.call_args.kwargs["env"]
        self.assertEqual(popen.call_args.kwargs.get("bufsize"), 0)
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["HF_HUB_DISABLE_PROGRESS_BARS"], "1")
        self.assertEqual(env["HF_HUB_DISABLE_XET"], "1")
        self.assertIn("Loading weights:", result.stderr)

    def test_pump_drains_cr_progress_without_newline(self):
        sink: list[str] = []
        payload = b"Uploading adapter: 1%\r" * 20 + b"Uploading adapter: 100%\n"
        MODULE._pump_stream(io.BytesIO(payload), sink, "stderr", 9)
        joined = "".join(sink)
        self.assertIn("Uploading adapter: 1%", joined)
        self.assertIn("Uploading adapter: 100%", joined)

    def test_latest_checkpoint_uses_highest_step(self):
        with patch.object(MODULE.os, "listdir", return_value=["checkpoint-2", "checkpoint-10"]), \
             patch.object(MODULE.os.path, "isdir", return_value=True):
            self.assertTrue(JobSpawner.latest_checkpoint({"job_uuid": "job-1"}).endswith("checkpoint-10"))


if __name__ == "__main__":
    unittest.main()
