"""Step 2 - Clean and preprocess text (lowercase, remove punctuation)."""
import re
import string

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation and digits, collapse whitespace."""
    text = str(text).lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\d+", " ", text)
    return _WS.sub(" ", text).strip()


def preprocess_series(series):
    return series.apply(clean_text)
