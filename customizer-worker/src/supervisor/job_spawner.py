"""Spawn / kill job_runner subprocesses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence


@dataclass
class SpawnResult:
    returncode: int
    stdout: str
    stderr: str
    pid: Optional[int] = None
    cancelled: bool = False
    checkpoint_path: Optional[str] = None


class JobSpawner:
    """One job = one OS subprocess running job_runner."""

    def __init__(
        self,
        python_executable: Optional[str] = None,
        job_runner_module: str = "src.job_runner",
        cwd: Optional[str] = None,
    ):
        self.python_executable = python_executable or sys.executable
        self.job_runner_module = job_runner_module
        self.cwd = cwd

    def spawn(
        self,
        job_payload: dict[str, Any],
        *,
        timeout_sec: Optional[float] = None,
        env: Optional[dict[str, str]] = None,
        extra_args: Optional[Sequence[str]] = None,
        heartbeat_callback: Optional[Callable[[], None]] = None,
        should_cancel: Optional[Callable[[], bool]] = None,
        heartbeat_interval_sec: float = 15.0,
    ) -> SpawnResult:
        fd, payload_path = tempfile.mkstemp(prefix="job_", suffix=".json")
        os.close(fd)
        try:
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(job_payload, f)

            cmd = [
                self.python_executable,
                "-m",
                self.job_runner_module,
                "--payload",
                payload_path,
            ]
            if extra_args:
                cmd.extend(extra_args)

            run_env = os.environ.copy()
            # Avoid tokenizer/thread pools fighting Windows process spawning during dataset.map.
            run_env.setdefault("TOKENIZERS_PARALLELISM", "false")
            # Torch 2.6 inductor expects triton_key; triton-windows 3.7 removed it.
            # Disable Unsloth/Torch compile so training uses eager kernels instead.
            if sys.platform.startswith("win"):
                run_env.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
                run_env.setdefault("TORCHDYNAMO_DISABLE", "1")
            if env:
                run_env.update(env)

            proc = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            started_at = time.monotonic()
            while True:
                remaining = None if timeout_sec is None else timeout_sec - (time.monotonic() - started_at)
                if remaining is not None and remaining <= 0:
                    self.hard_kill(proc)
                    stdout, stderr = proc.communicate()
                    return SpawnResult(
                        returncode=proc.returncode if proc.returncode is not None else -9,
                        stdout=stdout or "",
                        stderr=(stderr or "") + "\n[timeout] hard-killed job subprocess",
                        pid=proc.pid,
                    )

                wait_for = heartbeat_interval_sec if remaining is None else min(heartbeat_interval_sec, remaining)
                try:
                    stdout, stderr = proc.communicate(timeout=wait_for)
                    break
                except subprocess.TimeoutExpired:
                    if should_cancel is not None and should_cancel():
                        self.hard_kill(proc)
                        stdout, stderr = proc.communicate()
                        return SpawnResult(
                            returncode=proc.returncode if proc.returncode is not None else -15,
                            stdout=stdout or "", stderr=stderr or "", pid=proc.pid, cancelled=True,
                            checkpoint_path=self.latest_checkpoint(job_payload),
                        )
                    if heartbeat_callback is not None:
                        try:
                            heartbeat_callback()
                        except Exception:
                            # A deleted manager record must never leave an untracked trainer.
                            self.hard_kill(proc)
                            proc.communicate()
                            raise

            return SpawnResult(
                returncode=proc.returncode,
                stdout=stdout or "",
                stderr=stderr or "",
                pid=proc.pid,
            )
        finally:
            if os.path.exists(payload_path):
                try:
                    os.remove(payload_path)
                except OSError:
                    pass

    @staticmethod
    def hard_kill(proc: subprocess.Popen) -> None:
        """Terminate then kill — no orphan GPU processes."""
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    @staticmethod
    def latest_checkpoint(job_payload: dict[str, Any]) -> Optional[str]:
        """Find the last complete Trainer checkpoint in the persistent job workspace."""
        job_uuid = str(job_payload.get("job_uuid") or job_payload.get("jobId") or "")
        root = os.getenv("CUSTOMIZER_JOB_WORKSPACE_ROOT", os.path.join(tempfile.gettempdir(), "customizer-jobs"))
        checkpoints = os.path.join(root, job_uuid, "checkpoints")
        try:
            candidates = [os.path.join(checkpoints, name) for name in os.listdir(checkpoints)
                          if name.startswith("checkpoint-") and os.path.isdir(os.path.join(checkpoints, name))]
            return max(candidates, key=lambda path: int(os.path.basename(path).split("-", 1)[1])) if candidates else None
        except (OSError, ValueError):
            return None
