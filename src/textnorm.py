"""Общая нормализация для дедупликации, сплита и проверки контаминации."""

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_OPTION_NUMBER = re.compile(r"(?m)(?<=\n)\s*\d+\s*[.)]\s*")
_PUNCT_SPACES = re.compile(r"\s+([:;,.!?])")
_DASHES = str.maketrans({"—": "-", "–": "-", "‑": "-", " ": " "})


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_DASHES)
    text = _OPTION_NUMBER.sub("", text)
    text = _PUNCT_SPACES.sub(r"\1", text)
    return _SPACES.sub(" ", text).strip().lower()


def normalize_group(topic: str) -> str:
    return normalize_text(topic.split("|")[0])


def shingles(text: str, size: int) -> set[str]:
    words = re.findall(r"\w+", normalize_text(text))
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}
