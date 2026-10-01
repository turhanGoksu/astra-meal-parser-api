"""Unit tests for Turkish-aware text folding."""

import pytest

from astra_nutrition.text import embedding_text, fold


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Tavuk Göğsü", "tavuk göğsü"),  # Turkish letters kept (not folded)
        ("İki  Bardak", "iki bardak"),  # no combining dot, whitespace collapsed
        ("Chicken Breast", "chicken breast"),
        ("Izgara Tavuk", "izgara tavuk"),  # documented limit: "I" -> "i", not "ı"
    ],
)
def test_embedding_text(text: str, expected: str) -> None:
    assert embedding_text(text) == expected


def test_embedding_text_can_keep_original_casing() -> None:
    assert embedding_text("  Tavuk  Göğsü ", lowercase=False) == "Tavuk Göğsü"
