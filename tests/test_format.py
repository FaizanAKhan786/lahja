"""Unit tests for the pure-python data formatting core."""
from lahja.data.format import (
    Example,
    dedup_key,
    is_arabic,
    normalize,
    prepare_examples,
    to_chat_messages,
    to_text,
)


def make(instr="شلونك؟", out="بخير الحمدلله", **kw) -> Example:
    return Example(instruction=instr, output=out, **kw)


def test_is_arabic_detects_gulf_text():
    assert is_arabic("شلونك؟ وش أخبارك")
    assert not is_arabic("hello world, plain english")


def test_normalize_unifies_alef_and_strips_tatweel():
    assert normalize("أإآا") == "اااا"
    assert normalize("مـــرحبا") == "مرحبا"
    assert normalize("  كيف   حالك  ") == "كيف حالك"


def test_chat_messages_structure():
    msgs = to_chat_messages(make(input="النص"))
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "النص" in msgs[1]["content"]


def test_text_template_contains_all_turns():
    txt = to_text(make())
    for tag in ("<|system|>", "<|user|>", "<|assistant|>"):
        assert tag in txt


def test_prepare_deduplicates_and_filters():
    rows = [
        make(),
        make(),                                   # duplicate
        make(instr="مرحبا", out=""),              # empty output
        Example(instruction="english only", output="also english"),  # not arabic
        make(instr="وش رايك بالمشروع؟", out="وايد زين"),
    ]
    out = prepare_examples(rows)
    assert len(out) == 2


def test_dedup_key_ignores_alef_variants():
    assert dedup_key(make(instr="أهلا")) == dedup_key(make(instr="اهلا"))
