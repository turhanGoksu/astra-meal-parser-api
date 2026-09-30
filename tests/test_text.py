"""Unit tests for Turkish-aware text folding."""

import pytest

from app.text import fold


def test_plain_lower_is_not_turkish_aware() -> None:
    # The trap fold() exists for: the dotted capital I gains a combining dot.
    assert "İki".lower() != "iki"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("İki", "iki"),
        ("KAŞIK", "kasik"),
        ("Tavuk Göğsü", "tavuk gogsu"),
        ("tavuk gogsu", "tavuk gogsu"),
        ("  Kuru   Kayısı ", "kuru kayisi"),
        ("kâse", "kase"),
        ("Black Tea", "black tea"),
    ],
)
def test_fold(text: str, expected: str) -> None:
    assert fold(text) == expected
