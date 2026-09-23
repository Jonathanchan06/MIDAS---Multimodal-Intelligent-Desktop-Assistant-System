"""Generic embedding + vector-retrieval primitive.

Deliberately knows nothing about wardrobes, chat, or SQLite tables — it
deals only in strings, numpy arrays, packed bytes, and opaque (id, vector)
pairs, so any future feature (chat memory, wardrobe search, anything else)
can call it the same way.

Uses a single module-level ollama.Client shared across threads. This is
different from core/db.py, where every caller opens its own SQLite
connection — that's needed because SQLite connections carry transactional
state that isn't safe to share across threads. An HTTP client has no such
state, so one shared client is fine here.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import ollama

import config

logger = logging.getLogger(__name__)

_client = ollama.Client(host=config.OLLAMA_HOST)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Embed a batch of strings in one /api/embed call. Returns an array
    of shape (len(texts), dim), dtype config.EMBED_DTYPE. Raises if the
    model isn't reachable/pulled — callers decide whether that's fatal."""
    if not texts:
        return np.empty((0, 0), dtype=config.EMBED_DTYPE)
    response = _client.embed(
        model=config.EMBED_MODEL,
        input=list(texts),
        keep_alive=config.EMBED_KEEP_ALIVE,
        options={"num_ctx": config.EMBED_NUM_CTX},
    )
    return np.asarray(response.embeddings, dtype=config.EMBED_DTYPE)


def embed_text(text: str) -> np.ndarray:
    """Single-string convenience. Returns shape (dim,)."""
    return embed_texts([text])[0]


def pack_vector(vec: np.ndarray) -> bytes:
    """numpy vector -> SQLite BLOB bytes."""
    return np.asarray(vec, dtype=config.EMBED_DTYPE).tobytes()


def unpack_vector(blob: bytes | None) -> np.ndarray | None:
    """SQLite BLOB bytes -> numpy vector. None/empty -> None, so a row
    stored before a model was available doesn't crash a ranking pass."""
    if not blob:
        return None
    return np.frombuffer(blob, dtype=config.EMBED_DTYPE)


def cosine_top_k(
    query: np.ndarray,
    candidates: Sequence[tuple[Any, np.ndarray | None]],
    k: int = 5,
) -> list[tuple[Any, float]]:
    """Rank (id, vector) pairs by cosine similarity to `query`. Returns
    [(id, score), ...] descending, at most k. Ids are opaque — this
    module never learns what they refer to.

    Candidates with a None vector, or a vector whose shape doesn't match
    `query`, are skipped (and counted in one warning) rather than raising
    — the visible symptom if EMBED_MODEL is ever swapped for a
    different-dimension model without re-embedding existing rows."""
    usable_ids = []
    usable_vecs = []
    skipped = 0
    for item_id, vec in candidates:
        if vec is None or vec.shape != query.shape:
            skipped += 1
            continue
        usable_ids.append(item_id)
        usable_vecs.append(vec)

    if skipped:
        logger.warning(
            "cosine_top_k: skipped %d candidate(s) with missing/mismatched embeddings",
            skipped,
        )
    if not usable_vecs:
        return []

    matrix = np.stack(usable_vecs).astype(np.float64)
    query = query.astype(np.float64)

    matrix_norms = np.linalg.norm(matrix, axis=1)
    query_norm = np.linalg.norm(query)
    eps = 1e-10

    scores = (matrix @ query) / (matrix_norms * query_norm + eps)
    order = np.argsort(-scores)[:k]
    return [(usable_ids[i], float(scores[i])) for i in order]
