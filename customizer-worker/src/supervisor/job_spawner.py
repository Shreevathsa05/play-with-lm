"""Spawn / kill job_runner subprocesses."""

from __future__ import annotations

import codecs
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SpawnResult:
    returncode: int
    stdout: str
    stderr: str
    pid: Optional[int] = None
    cancelled: bool = False
    checkpoint_path: Optional[str] = None


def _flush_pipe_text(buf: str, sink: list[str], label: str, pid: Optional[int], *, final: bool = False) -> str:
    """Split on \\n and \\r so huggingface_hub/tqdm progress ticks cannot fill the pipe."""
    while True:
        n, r = buf.find("\n"), buf.find("\r")
        if n < 0 and r < 0:
            break
        i = n if r < 0 else r if n < 0 else min(n, r)
        line, sep, buf = buf[:i], buf[i], buf[i + 1 :]
        if sep == "\r" and buf.startswith("\n"):
            buf = buf[1:]
        text = line.rstrip()
        sink.append(text + "\n")
        if text:
            logger.info("job_subprocess pid=%s %s: %s", pid, label, text)
    if len(buf) >= 8192 or (final and buf):
        text = buf.rstrip()
        sink.append(buf if buf.endswith("\n") else buf + "\n")
        if text:
            logger.info("job_subprocess pid=%s %s: %s", pid, label, text)
        return ""
    return buf


def _pump_stream(stream, sink: list[str], label: str, pid: Optional[int]) -> None:
    """Drain bytes as they arrive. readline() deadlocks on tqdm '\\r' bars (Windows 64KiB pipe)."""
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    try:
        while True:
            try:
                chunk = stream.read(4096)
            except Exception:
                logger.exception("job_subprocess pid=%s %s read failed", pid, label)
                break
            if not chunk:
                break
            try:
                if isinstance(chunk, str):
                    # Defensive: some platforms wrap pipes as text; never use locale cp1252.
                    text = chunk.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
                else:
                    text = decoder.decode(chunk)
            except Exception:
                text = chunk.decode("utf-8", "replace") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            pending = _flush_pipe_text(pending + text, sink, label, pid)
        try:
            pending = _flush_pipe_text(pending + decoder.decode(b"", final=True), sink, label, pid, final=True)
        except Exception:
            pending = _flush_pipe_text(pending, sink, label, pid, final=True)
    except Exception:
        logger.exception("job_subprocess pid=%s %s pump failed", pid, label)
    finally:
        try:
            stream.close()
        except Exception:
            pass


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
            run_env.setdefault("PYTHONIOENCODING", "utf-8")
            run_env.setdefault("PYTHONUTF8", "1")
            # Hub/tqdm '\\r' bars + the Xet uploader stall inside a piped Windows job.
            run_env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            run_env.setdefault("TQDM_DISABLE", "1")
            run_env.setdefault("HF_HUB_DISABLE_XET", "1")
            # Transformers weight-load bars also emit Unicode to stderr on Windows.
            run_env.setdefault("TRANSFORMERS_VERBOSITY", "error")
            run_env.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            # Torch 2.6 inductor expects triton_key; triton-windows 3.7 removed it.
            # Disable Unsloth/Torch compile so training uses eager kernels instead.
            if sys.platform.startswith("win"):
                run_env.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
                run_env.setdefault("TORCHDYNAMO_DISABLE", "1")
            if env:
                run_env.update(env)

            # Unbuffered binary pipes: communicate() deadlocks once Torch fills ~64KiB,
            # and readline() deadlocks on huggingface_hub tqdm which only writes '\\r'.
            proc = subprocess.Popen(
                cmd,
                cwd=self.cwd,
                env=run_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            stdout_chunks: list[str] = []
            stderr_chunks: list[str] = []
            out_thread = threading.Thread(
                target=_pump_stream,
                args=(proc.stdout, stdout_chunks, "stdout", proc.pid),
                daemon=True,
            )
            err_thread = threading.Thread(
                target=_pump_stream,
                args=(proc.stderr, stderr_chunks, "stderr", proc.pid),
                daemon=True,
            )
            out_thread.start()
            err_thread.start()

            started_at = time.monotonic()
            next_heartbeat_at = started_at
            while True:
                now = time.monotonic()
                remaining = None if timeout_sec is None else timeout_sec - (now - started_at)
                if remaining is not None and remaining <= 0:
                    self.hard_kill(proc)
                    out_thread.join(timeout=2)
                    err_thread.join(timeout=2)
                    return SpawnResult(
                        returncode=proc.returncode if proc.returncode is not None else -9,
                        stdout="".join(stdout_chunks),
                        stderr="".join(stderr_chunks) + "\n[timeout] hard-killed job subprocess",
                        pid=proc.pid,
                    )

                if proc.poll() is not None:
                    out_thread.join(timeout=5)
                    err_thread.join(timeout=5)
                    break

                if should_cancel is not None and should_cancel():
                    self.hard_kill(proc)
                    out_thread.join(timeout=2)
                    err_thread.join(timeout=2)
                    return SpawnResult(
                        returncode=proc.returncode if proc.returncode is not None else -15,
                        stdout="".join(stdout_chunks),
                        stderr="".join(stderr_chunks),
                        pid=proc.pid,
                        cancelled=True,
                        checkpoint_path=self.latest_checkpoint(job_payload),
                    )

                if heartbeat_callback is not None and now >= next_heartbeat_at:
                    next_heartbeat_at = now + heartbeat_interval_sec
                    try:
                        heartbeat_callback()
                    except Exception:
                        # A deleted manager record must never leave an untracked trainer.
                        self.hard_kill(proc)
                        out_thread.join(timeout=2)
                        err_thread.join(timeout=2)
                        raise

                sleep_for = 0.5
                if remaining is not None:
                    sleep_for = min(sleep_for, max(remaining, 0.05))
                if heartbeat_callback is not None:
                    sleep_for = min(sleep_for, max(next_heartbeat_at - time.monotonic(), 0.05))
                time.sleep(sleep_for)

            return SpawnResult(
                returncode=proc.returncode if proc.returncode is not None else -1,
                stdout="".join(stdout_chunks),
                stderr="".join(stderr_chunks),
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
