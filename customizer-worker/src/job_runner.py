"""Subprocess entry: run one fine-tune job pipeline from a JSON payload."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import Any

# Must run before Unsloth/Torch imports. Torch 2.6 inductor + triton-windows 3.7
# crash with: cannot import name 'triton_key' from triton.compiler.compiler
if sys.platform.startswith("win"):
    os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from src.utils.logcarrier import LogCarrier
from src.utils.progress import attach_progress_sink


def load_payload(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    from src.finetuning_flows import run_recipe

    job_uuid = str(payload.get("job_uuid") or payload.get("jobId") or "")
    if not job_uuid:
        raise ValueError("job_uuid is required")

    log = LogCarrier(job_uuid)
    attach_progress_sink(log, job_uuid)
    workspace_root = os.getenv("CUSTOMIZER_JOB_WORKSPACE_ROOT", os.path.join(tempfile.gettempdir(), "customizer-jobs"))
    scratch = os.path.join(workspace_root, job_uuid)
    os.makedirs(scratch, exist_ok=True)
    completed = False
    log.info(
        "pipeline_start",
        scratch=scratch,
        recipe=payload.get("recipe"),
        unsloth_compile_disable=os.environ.get("UNSLOTH_COMPILE_DISABLE"),
    )

    try:
        result = run_recipe(payload, scratch_dir=scratch, log=log)
        completed = result.get("status") == "COMPLETED"
        log.info("pipeline_complete", status=result.get("status", "COMPLETED"))
        return result
    except Exception as exc:
        log.error("pipeline_failed", error=str(exc))
        raise
    finally:
        if completed:
            try:
                shutil.rmtree(scratch, ignore_errors=True)
                log.info("scratch_wiped")
            except Exception as wipe_exc:
                log.warning("scratch_wipe_failed", error=str(wipe_exc))
        else:
            log.info("checkpoint_workspace_retained", scratch=scratch)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single fine-tune job")
    parser.add_argument("--payload", required=True, help="Path to job JSON payload")
    args = parser.parse_args(argv)

    payload = load_payload(args.payload)
    result = run_pipeline(payload)
    print("__CUSTOMIZER_RESULT__=" + json.dumps(result, separators=(",", ":")))
    return 0 if result.get("status", "COMPLETED") == "COMPLETED" else 1


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    raise SystemExit(main())
