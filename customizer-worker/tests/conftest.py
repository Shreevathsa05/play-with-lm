import os
import tempfile
import uuid
from pathlib import Path

SAFE_TMP = Path(__file__).resolve().parents[2] / ".test-tmp"
SAFE_TMP.mkdir(exist_ok=True)
tempfile.tempdir = str(SAFE_TMP)


def _mkdtemp(suffix=None, prefix=None, dir=None):
    base = Path(dir or SAFE_TMP)
    base.mkdir(exist_ok=True)
    prefix = "tmp" if prefix is None else prefix
    suffix = "" if suffix is None else suffix
    for _ in range(100):
        path = base / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            path.mkdir()
            return str(path)
        except FileExistsError:
            continue
    raise FileExistsError("could not create unique temp directory")


def _mkstemp(suffix=None, prefix=None, dir=None, text=False):
    base = Path(dir or SAFE_TMP)
    base.mkdir(exist_ok=True)
    prefix = "tmp" if prefix is None else prefix
    suffix = "" if suffix is None else suffix
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if not text:
        flags |= getattr(os, "O_BINARY", 0)
    for _ in range(100):
        path = base / f"{prefix}{uuid.uuid4().hex}{suffix}"
        try:
            fd = os.open(path, flags, 0o600)
            return fd, str(path)
        except FileExistsError:
            continue
    raise FileExistsError("could not create unique temp file")


tempfile.mkdtemp = _mkdtemp
tempfile.mkstemp = _mkstemp