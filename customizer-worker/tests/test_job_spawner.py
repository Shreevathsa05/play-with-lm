import os
import sys
import unittest
import importlib.util
from unittest.mock import patch

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

    def test_latest_checkpoint_uses_highest_step(self):
        with patch.object(MODULE.os, "listdir", return_value=["checkpoint-2", "checkpoint-10"]), \
             patch.object(MODULE.os.path, "isdir", return_value=True):
            self.assertTrue(JobSpawner.latest_checkpoint({"job_uuid": "job-1"}).endswith("checkpoint-10"))


if __name__ == "__main__":
    unittest.main()
