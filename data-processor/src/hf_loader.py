"""HuggingFace dataset loading with multi-config support."""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

logger = logging.getLogger(__name__)

# Prefer compact teaching/demo splits; avoid multi-hundred-GB shards like OpenThoughts 1.2M.
_PREFERRED_SPLITS = (
    "smoltalk_everyday_convs_reasoning_Qwen3_32B_think",
    "Mixture_of_Thoughts_science_no_think",
    "OpenHermes_2.5_no_think",
    "table_gpt_no_think",
    "train",
    "default",
)


def pick_hf_config(available: list[str], preferred: Optional[str] = None) -> str:
    if preferred and preferred in available:
        return preferred
    for name in ("SFT", "sft", "train", "default", "Mid"):
        if name in available:
            return name
    return available[0]


def pick_hf_split(available: list[str], preferred: Optional[str] = None) -> str:
    if preferred and preferred in available:
        return preferred
    for name in _PREFERRED_SPLITS:
        if name in available:
            return name

    def _is_huge(name: str) -> bool:
        lowered = name.lower()
        return any(token in lowered for token in ("1.2m", "openthoughts", "64k", "131k"))

    compact = [name for name in available if not _is_huge(name)]
    return compact[0] if compact else available[0]


def parse_hf_repo_and_config(
    hf_id: str, explicit_config: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """
    Accepts:
      - repo id: HuggingFaceTB/smoltalk2
      - repo:config: HuggingFaceTB/smoltalk2:SFT
    Explicit config wins over embedded :config, but the embedded suffix is still
    stripped from the repo id.
    """
    repo = (hf_id or "").strip()
    config = (explicit_config or "").strip() or None
    if ":" in repo:
        left, right = repo.rsplit(":", 1)
        if "/" in left and right and "/" not in right:
            repo = left.strip()
            if not config:
                config = right.strip()
    return repo, config


def parse_config_and_split(
    config: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Accepts:
      - SFT
      - SFT/OpenHermes_2.5_no_think
      - SFT:OpenHermes_2.5_no_think
    """
    value = (config or "").strip() or None
    if not value:
        return None, None
    for sep in ("/", ":"):
        if sep in value:
            left, right = value.split(sep, 1)
            left, right = left.strip(), right.strip()
            if left and right and "/" not in left:
                return left, right
    return value, None


def _rows_from_stream(dataset: Any, max_rows: int) -> list[Any]:
    rows: list[Any] = []
    for index, row in enumerate(dataset):
        if index >= max_rows:
            break
        rows.append(dict(row))
    return rows


def load_hf_dataset(
    hf_id: str,
    config: Optional[str] = None,
    max_rows: int = 10000,
    split: Optional[str] = None,
) -> list[Any]:
    """
    Stream a bounded sample for auditing.

    Important: do not call non-streaming load_dataset() first. For configs like
    HuggingFaceTB/smoltalk2:SFT that would download every parquet shard (e.g. 113
    OpenThoughts files) before max_rows can apply.
    """
    repo, config = parse_hf_repo_and_config(hf_id, config)
    config, embedded_split = parse_config_and_split(config)
    split = (split or "").strip() or embedded_split

    try:
        available_configs = get_dataset_config_names(repo)
    except Exception:
        available_configs = []
    if available_configs:
        if config and config not in available_configs:
            # UI often sends a split name in the config field for multi-split repos.
            if not split:
                split = config
            chosen = pick_hf_config(available_configs, preferred=None)
            logger.warning(
                "HF value %r is not a config in %s; using config=%r split=%r",
                config,
                available_configs,
                chosen,
                split,
            )
            config = chosen
        elif not config:
            config = pick_hf_config(available_configs, preferred=None)

    if not split:
        try:
            available_splits = (
                get_dataset_split_names(repo, config_name=config)
                if config
                else get_dataset_split_names(repo)
            )
        except Exception:
            available_splits = []
        if available_splits:
            split = pick_hf_split(available_splits)
            logger.info(
                "Selected HF split=%s from %s available split(s)",
                split,
                len(available_splits),
            )
        else:
            split = "train"

    logger.info(
        "Streaming HuggingFace sample repo=%s config=%s split=%s max_rows=%s",
        repo,
        config,
        split,
        max_rows,
    )

    kwargs: dict[str, Any] = {
        "path": repo,
        "split": split,
        "streaming": True,
    }
    if config:
        kwargs["name"] = config

    try:
        dataset = load_dataset(**kwargs)
    except Exception as first_err:
        msg = str(first_err)
        if (
            "Config name is missing" in msg
            or "BuilderConfig" in msg
            or "available configs" in msg.lower()
        ):
            available: list[str] = []
            if "available configs" in msg:
                after = msg.split("available configs", 1)[-1]
                available = re.findall(r"'([^']+)'", after)
            if not available:
                available = re.findall(r"'([^']+)'", msg)
            available = [c for c in available if c and c != repo and "/" not in c]
            if not available:
                raise
            chosen = pick_hf_config(available, preferred=config or "SFT")
            logger.warning(
                "HF config required for %s. Available=%s; selecting '%s'",
                repo,
                available,
                chosen,
            )
            kwargs["name"] = chosen
            dataset = load_dataset(**kwargs)
        else:
            raise

    rows = _rows_from_stream(dataset, max_rows)
    logger.info("Loaded %s HuggingFace sample rows (cap=%s)", len(rows), max_rows)
    return rows
