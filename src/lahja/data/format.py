"""Instruction formatting and chat-template utilities.

Pure-python (no torch/transformers imports) so it is fast to unit-test
and reusable by both the training and serving paths.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

SYSTEM_PROMPT = (
    "أنت مساعد ذكي يفهم اللهجات الخليجية (القطرية، الإماراتية، السعودية، الكويتية) "
    "ويرد بنفس لهجة المستخدم بطلاقة واحترام."
)


@dataclass(frozen=True)
class Example:
    instruction: str
    output: str
    input: str = ""
    dialect: str = "gulf"

    @property
    def prompt(self) -> str:
        if self.input:
            return f"{self.instruction}\n\n{self.input}"
        return self.instruction


def is_arabic(text: str, threshold: float = 0.3) -> bool:
    """True if at least `threshold` fraction of non-space chars are Arabic."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    arabic = sum(1 for c in chars if ARABIC_RANGE.match(c))
    return arabic / len(chars) >= threshold

def normalize(text: str) -> str:
    """Light Arabic normalization: unify alef/teh-marbuta variants, strip tatweel."""
    text = re.sub("[إأآا]", "ا", text)
    text = text.replace("ـ", "")  # tatweel
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def to_chat_messages(ex: Example) -> list[dict]:
    """Convert an Example to OpenAI/HF chat-message format."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ex.prompt},
        {"role": "assistant", "content": ex.output},
    ]


def to_text(ex: Example) -> str:
    """Plain-text fallback template (ChatML-style) for models without a chat template."""
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\n{ex.prompt}\n"
        f"<|assistant|>\n{ex.output}"
    )


def dedup_key(ex: Example) -> str:
    return hashlib.sha256(normalize(ex.prompt).encode()).hexdigest()


def load_jsonl(path: str | Path) -> Iterator[Example]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield Example(
                instruction=row["instruction"],
                input=row.get("input", ""),
                output=row["output"],
                dialect=row.get("dialect", "gulf"),
            )


def prepare_examples(
    examples: Iterable[Example],
    min_output_chars: int = 2,
    require_arabic: bool = True,
) -> list[Example]:
    """Filter + deduplicate examples for training."""
    seen: set[str] = set()
    out: list[Example] = []
    for ex in examples:
        if len(ex.output.strip()) < min_output_chars:
            continue
        if require_arabic and not (is_arabic(ex.prompt) or is_arabic(ex.output)):
            continue
        key = dedup_key(ex)
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out
