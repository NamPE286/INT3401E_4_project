from __future__ import annotations

import html
import re
from typing import List

_BR_TAG_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def clean(s: str) -> List[str]:
    if not s:
        return [""]

    parts = _BR_TAG_RE.split(s) if _BR_TAG_RE.search(s) else [s]
    cleaned_parts: List[str] = []

    for part in parts:
        text = html.unescape(part)
        text = _HTML_TAG_RE.sub(" ", text)
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        
        if text:
            cleaned_parts.append(text)

    return cleaned_parts or [""]
