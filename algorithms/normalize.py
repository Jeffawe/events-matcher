"""Name normalisation. Fixed code, not learned: the dictionary stores normalised keys,
so this must stay stable (change it and old dictionary keys stop matching)."""

import re
import unicodedata


def normalize(text: str) -> str:
    """Casefold, strip accents, turn punctuation into spaces, collapse whitespace.

    "Sofía  Núñez" -> "sofia nunez"      "J. O'Brien" -> "j o brien"
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[\W_]+", " ", text.casefold())
    return text.strip()


def tokens(text: str) -> list[str]:
    return normalize(text).split()
