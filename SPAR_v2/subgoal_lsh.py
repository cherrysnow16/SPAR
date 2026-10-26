"""Subgoal-string identity via MinHash LSH.

Free-text subgoals proposed independently per trajectory are merged into canonical
strings within one uid-group, so graph nodes keyed on (observation, subgoal strings)
merge consistently across rollouts:
  1. normalize (lowercase, drop punctuation, digits, stopwords)
  2. shingles = word unigrams + character 3-grams
  3. MinHash signatures, banded LSH -> candidate pairs
  4. keep a candidate pair only if its exact Jaccard >= threshold
  5. union-find over kept pairs; canonical = lexicographically smallest member
Steps 4-5 make the result deterministic and independent of input order.
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

_STOPWORDS = frozenset({
    "a", "an", "the", "in", "on", "into", "onto", "to", "of", "with", "from",
    "at", "for", "and", "it", "its", "inside", "up", "your", "you",
})
_P = (1 << 61) - 1


def normalize_subgoal(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    words = [w for w in text.split() if not w.isdigit() and w not in _STOPWORDS]
    return " ".join(words) if words else " ".join(text.split())


def _shingles(norm: str) -> set[str]:
    s = {"w:" + w for w in norm.split()}
    if len(norm) < 3:
        s.add("c:" + norm)
    else:
        s.update("c:" + norm[i:i + 3] for i in range(len(norm) - 2))
    return s


def _base_hash(x: str) -> int:
    return int.from_bytes(hashlib.blake2b(x.encode("utf-8"), digest_size=8).digest(), "little")


def _perm_params(num_perm: int):
    rng = np.random.RandomState(1234)
    a = [int(v) for v in rng.randint(1, 2 ** 31 - 1, size=num_perm)]
    b = [int(v) for v in rng.randint(0, 2 ** 31 - 1, size=num_perm)]
    return a, b


def _minhash(shingles: set[str], a: list[int], b: list[int]) -> tuple[int, ...]:
    hs = [_base_hash(x) for x in shingles]
    return tuple(min((ai * h + bi) % _P for h in hs) for ai, bi in zip(a, b))


def jaccard(x: set, y: set) -> float:
    return len(x & y) / len(x | y) if (x or y) else 1.0


def cluster_subgoals(texts, threshold: float = 0.6, num_perm: int = 128, bands: int = 32) -> dict[str, str]:
    """Map each raw subgoal text to its canonical (normalized) cluster representative."""
    assert num_perm % bands == 0, "num_perm must be divisible by bands"
    rows = num_perm // bands
    norms = sorted({normalize_subgoal(t) for t in texts})
    shingle_sets = {n: _shingles(n) for n in norms}
    a, b = _perm_params(num_perm)
    sigs = {n: _minhash(shingle_sets[n], a, b) for n in norms}

    parent = {n: n for n in norms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    candidates = set()
    for band in range(bands):
        buckets = {}
        for n in norms:
            buckets.setdefault(sigs[n][band * rows:(band + 1) * rows], []).append(n)
        for members in buckets.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidates.add((members[i], members[j]))

    for x, y in sorted(candidates):
        if jaccard(shingle_sets[x], shingle_sets[y]) >= threshold:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

    clusters = {}
    for n in norms:
        clusters.setdefault(find(n), []).append(n)
    canonical = {n: min(members) for members in clusters.values() for n in members}
    return {t: canonical[normalize_subgoal(t)] for t in texts}
