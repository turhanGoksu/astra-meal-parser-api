"""Text normalization shared by amount parsing and food matching."""

_TURKISH_FOLD = str.maketrans(
    {
        "İ": "i",
        "I": "i",
        "ı": "i",
        "Ş": "s",
        "ş": "s",
        "Ğ": "g",
        "ğ": "g",
        "Ü": "u",
        "ü": "u",
        "Ö": "o",
        "ö": "o",
        "Ç": "c",
        "ç": "c",
        "Â": "a",
        "â": "a",
        "Î": "i",
        "î": "i",
        "Û": "u",
        "û": "u",
    }
)


def fold(text: str) -> str:
    """Lowercase, fold Turkish letters to ASCII and collapse whitespace.

    ``str.lower()`` is not Turkish-aware: ``"İ".lower()`` is "i" plus a
    combining dot, so ``"İki".lower() != "iki"``. Folding both user text and
    our own vocabularies to plain ASCII avoids that trap and also matches
    users who type without Turkish characters ("tavuk gogsu", "kasik").
    """
    return " ".join(text.translate(_TURKISH_FOLD).lower().split())


def embedding_text(text: str) -> str:
    """Normalize text before embedding: lowercase, but keep Turkish letters.

    The embedding model is case-sensitive for Turkish ("tavuk göğsü" vs
    "Tavuk göğsü": 0.82 similarity), so aliases and queries must share casing.
    ASCII folding is avoided here because it also lowered similarity (0.71).
    Known limit: a capital "I" becomes "i", although in Turkish words it
    stands for "ı" ("Izgara" -> "izgara", not "ızgara").
    """
    return " ".join(text.replace("İ", "i").lower().split())
