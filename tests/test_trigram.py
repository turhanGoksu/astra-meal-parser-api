"""pg_trgm parity: expected values were produced by PostgreSQL's pg_trgm."""

import numpy as np
import pytest

from astra_nutrition.trigram import similarity, trigrams


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("kase", {"  k", " ka", "ase", "kas", "se "}),
        ("Kase", {"  k", " ka", "ase", "kas", "se "}),  # lowercased
        (
            "tavuk gogsu",  # each word padded separately
            {"  g", "  t", " go", " ta", "avu", "gog", "gsu", "ogs", "su ", "tav",
             "uk ", "vuk"},
        ),
        ("70% dark", {"  7", "  d", " 70", " da", "70 ", "ark", "dar", "rk "}),
        (
            "hard-boiled egg",  # the hyphen separates words
            {"  b", "  e", "  h", " bo", " eg", " ha", "ard", "boi", "ed ", "egg",
             "gg ", "har", "ile", "led", "oil", "rd "},
        ),
        ("a", {"  a", " a "}),
        ("aaaa", {"  a", " aa", "aa ", "aaa"}),  # a set: duplicates once
        ("!!! ,,", set()),
    ],
)  # fmt: skip
def test_trigrams_match_pg_trgm(text: str, expected: set[str]) -> None:
    assert trigrams(text) == expected


@pytest.mark.parametrize(
    ("a", "b", "pg_value"),
    [
        ("kase", "kasa", 0.42857143),
        ("tavuk gosu", "tavuk gogsu", 0.64285713),
        ("yogurt", "yougurt", 0.5),
        ("elma", "elmas", 0.5714286),
        ("", "", 0.0),
        ("", "kase", 0.0),
    ],
)
def test_similarity_matches_pg_trgm_float4(a: str, b: str, pg_value: float) -> None:
    # PostgreSQL returns float4; compare after the same rounding, exactly.
    assert similarity(a, b) == float(np.float32(pg_value))
