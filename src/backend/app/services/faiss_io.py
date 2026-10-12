"""Unicode-safe FAISS index I/O.

``faiss.read_index`` / ``faiss.write_index`` take a C++ ``const char*`` path.
On Windows that is interpreted with the ANSI code page, so any non-ASCII
character in the path (e.g. Vietnamese "Học máy") is mangled and the open
fails. To keep serving artifacts portable across machines and filesystems, we
serialize the index to a byte buffer in Python (Unicode-safe) and hand only the
raw bytes to FAISS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


def save_index(index: "faiss.Index", path: Path) -> None:
    """Serialize a FAISS index to ``path`` using Unicode-safe file I/O."""

    if faiss is None:
        raise RuntimeError("faiss is not installed; cannot save index.")
    # faiss.serialize_index returns a uint8 numpy array of the raw index bytes.
    data = faiss.serialize_index(index)
    Path(path).write_bytes(np.ascontiguousarray(data, dtype="uint8").tobytes())


def load_index(path: Path) -> "faiss.Index":
    """Load a FAISS index from ``path`` using Unicode-safe file I/O."""

    if faiss is None:
        raise RuntimeError("faiss is not installed; cannot load index.")
    raw = Path(path).read_bytes()
    arr = np.frombuffer(raw, dtype="uint8")
    return faiss.deserialize_index(arr)
