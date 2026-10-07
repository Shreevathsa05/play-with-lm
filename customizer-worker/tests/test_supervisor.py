import os
import sys
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.supervisor import FCFSSupervisor, JobSpawner, ResourceMonitor, SpawnResult
from src.finetuning_modules.compatibility import GateDecision


class TestResourceMonitor(unittest.TestCase):
    def test_probes(self):
        mon = ResourceMonitor(
            vram_probe=lambda: (2048, 4096),
            ram_probe=lambda: (8192, 16384),
        )
        snap = mon.snapshot()
        self.assertEqual(snap.free_vram_mb, 2048)
        self.assertEqual(snap.total_ram_mb, 16384)


class TestFCFSSupervisor(unittest.TestCase):
    def test_reject_without_spawn(self):
        mon = ResourceMonitor(vram_probe=lambda: (8000, 8192), ram_probe=lambda: (16000, 32768))
        spawner = MagicMock()
        superv = FCFSSupervisor(
            monitor=mon,
            spawner=spawner,
            poll_interval_sec=0.01,
            max_wait_sec=0.05,
        )
        # Force reject via vision model id
        result = superv.run_job(
            {
                "job_uuid": "j-reject",
                "base_model_id": "llava-hf/llava",
                "recipe": "lora",
                "model_params_b": 1,
            }
        )
        self.assertEqual(result.gate.decision, GateDecision.REJECT)
        spawner.spawn.assert_not_called()

    def test_wait_then_accept_spawns(self):
        calls = {"n": 0}

        def vram_probe():
            calls["n"] += 1
            # first call busy, then free
            if calls["n"] < 3:
                return (1, 8192)
            return (8000, 8192)

        mon = ResourceMonitor(vram_probe=vram_probe, ram_probe=lambda: (16000, 32768))
        spawner = MagicMock()
        spawner.spawn.return_value = SpawnResult(returncode=0, stdout="", stderr="", pid=42)
        superv = FCFSSupervisor(
            monitor=mon,
            spawner=spawner,
            poll_interval_sec=0.01,
            max_wait_sec=2.0,
        )
        statuses = []
        result = superv.run_job(
            {
                "job_uuid": "j-wait",
                "base_model_id": "google/gemma-3-270m-it",
                "recipe": "qlora_4bit",
                "model_params_b": "270m",
            },
            status_callback=statuses.append,
        )
        self.assertEqual(result.gate.decision, GateDecision.ACCEPT)
        self.assertIsNotNone(result.spawn)
        self.assertEqual(result.spawn.returncode, 0)
        self.assertEqual(statuses, ["WAITING", "RUNNING"])
        spawner.spawn.assert_called_once()

    def test_dry_run_bypasses_capacity_wait(self):
        mon = ResourceMonitor(vram_probe=lambda: (1, 4096), ram_probe=lambda: (1, 16384))
        spawner = MagicMock()
        spawner.spawn.return_value = SpawnResult(returncode=0, stdout="", stderr="", pid=42)
        superv = FCFSSupervisor(monitor=mon, spawner=spawner)

        result = superv.run_job(
            {
                "job_uuid": "j-dry-run",
                "base_model_id": "google/gemma-3-270m-it",
                "recipe": "qlora_8bit",
                "model_params_b": "270m",
                "dry_run": True,
            }
        )

        self.assertEqual(result.gate.decision, GateDecision.ACCEPT)
        self.assertEqual(result.waited_sec, 0.0)
        spawner.spawn.assert_called_once()


class TestJobSpawnerHardKill(unittest.TestCase):
    def test_hard_kill_already_exited(self):
        proc = MagicMock()
        proc.poll.return_value = 0
        JobSpawner.hard_kill(proc)
        proc.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
