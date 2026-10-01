"""A pure-Python copy of PostgreSQL pg_trgm's similarity().

The fuzzy threshold was tuned on the dev set with pg_trgm scores, so this must
return the same numbers, or the same threshold would silently make different
decisions. Rules (checked against pg_trgm's show_trgm/similarity):
1. lowercase;
2. split into words at every non-alphanumeric character;
3. pad each word with two spaces in front and one at the end;
4. take every 3-character window, as a set (duplicates count once);
5. similarity = shared / (|A| + |B| - shared), as a float4 like PostgreSQL.
"""

import re

import numpy as np

_WORD = re.compile(r"[^\W_]+")  # letters and digits only (no underscore)


def trigrams(text: str) -> frozenset[str]:
    """pg_trgm's trigram set of a text."""
    grams: set[str] = set()
    for word in _WORD.findall(text.lower()):
        padded = f"  {word} "
        grams.update(padded[i : i + 3] for i in range(len(padded) - 2))
    return frozenset(grams)


def similarity_of(a: frozenset[str], b: frozenset[str]) -> float:
    """pg_trgm similarity of two precomputed trigram sets."""
    shared = len(a & b)
    union = len(a) + len(b) - shared
    if union == 0:
        return 0.0
    # PostgreSQL computes in float4; round the same way so threshold
    # comparisons near the boundary agree exactly.
    return float(np.float32(shared) / np.float32(union))


def similarity(a: str, b: str) -> float:
    """pg_trgm's similarity(a, b)."""
    return similarity_of(trigrams(a), trigrams(b))
