"""Small immutable normalized-dataset snapshot helper."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import csv
import io
from typing import Any, Iterable


def canonical_row(row: dict[str, Any]) -> str:
    return json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def snapshot_hash(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_row(row).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_jsonl(rows: list[dict[str, Any]], directory: str | None = None) -> tuple[str, str]:
    """Write deterministic JSONL and return (path, sha256)."""
    path = tempfile.mktemp(prefix="dataset_snapshot_", suffix=".jsonl", dir=directory)
    digest = hashlib.sha256()
    with open(path, "wb") as stream:
        for row in rows:
            line = (canonical_row(row) + "\n").encode("utf-8")
            digest.update(line)
            stream.write(line)
    return path, digest.hexdigest()


def parse_dataset_bytes(raw_data: bytes, object_name: str) -> list[dict[str, Any]]:
    text = raw_data.decode("utf-8")
    suffix = object_name.lower().rsplit(".", 1)[-1] if "." in object_name else "json"
    if suffix == "jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == "csv":
        return [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if suffix == "txt":
        return [{"text": line} for line in text.splitlines() if line.strip()]
    parsed = json.loads(text)
    return parsed if isinstance(parsed, list) else [parsed]


__all__ = ["canonical_row", "snapshot_hash", "write_jsonl", "parse_dataset_bytes"]
