"""Маскирование телефонов, email и дат рождения."""

import re

PHONE = re.compile(r"(?:\+7|\b8)[\s\-(]{0,3}\d{3}[\s\-)]{0,3}\d{3}[\s\-]?\d{2}[\s\-]?\d{2}\b")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
BIRTH_DATE = re.compile(
    r"(?P<pre>(?:дат[аеы]\s+рождения|год\s+рождения|родил(?:ся|ась)|"
    r"\bг\.\s?р\.|\bд\.\s?р\.)\s*[:\-—]?\s*)"
    r"(?:0?[1-9]|[12]\d|3[01])[.\-/](?:0?[1-9]|1[0-2])[.\-/](?:19|20)\d{2}\b",
    re.IGNORECASE,
)
PATTERNS = {"phone": PHONE, "email": EMAIL, "birth_date": BIRTH_DATE}
PLACEHOLDERS = {"phone": "[PHONE]", "email": "[EMAIL]", "birth_date": r"\g<pre>[DATE]"}


def scrub(text: str) -> tuple[str, dict[str, int]]:
    hits: dict[str, int] = {}
    for name, pattern in PATTERNS.items():
        text, count = pattern.subn(PLACEHOLDERS[name], text)
        if count:
            hits[name] = count
    return text, hits
