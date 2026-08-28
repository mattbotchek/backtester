"""Atomic persistence helpers for production model artifacts."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any


def atomic_pickle_dump(value: Any, path: str | Path) -> None:
    """Write a pickle without exposing a partial artifact to readers."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            pickle.dump(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
