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
