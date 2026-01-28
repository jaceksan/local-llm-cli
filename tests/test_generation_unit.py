import torch

from app.generation import MaxThinkingTokensCriteria, _strip_redundant_answer_prefix


class _CharTokenizer:
    """Tiny deterministic tokenizer for unit tests (no HF downloads)."""

    def encode(self, text: str, *, add_special_tokens: bool = False):  # noqa: ARG002
        return [ord(ch) for ch in text]

    def decode(self, ids, *, skip_special_tokens: bool = False):  # noqa: ARG002
        return "".join(chr(i) for i in ids)


def _t(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long)


def test_strip_redundant_answer_prefix_removes_repeated_tags() -> None:
    raw = "</think>\n<answer>\n</think>\n<answer>\nHello"
    assert _strip_redundant_answer_prefix(raw) == "Hello"


def test_strip_redundant_answer_prefix_keeps_answer_content() -> None:
    raw = "\n\n<answer>\n  Hello\n"
    assert _strip_redundant_answer_prefix(raw) == "Hello\n"


def test_thinking_criteria_stops_at_cap_inside_open_think() -> None:
    tok = _CharTokenizer()
    cap = 5
    crit = MaxThinkingTokensCriteria(tok, prompt_length=0, max_thinking_tokens=cap)
    # gen_count == cap => full check runs
    ids = tok.encode("<think>abcde", add_special_tokens=False)
    assert bool(crit(_t(ids), scores=None)) is True  # type: ignore[arg-type]


def test_thinking_criteria_does_not_stop_when_think_closed() -> None:
    tok = _CharTokenizer()
    crit = MaxThinkingTokensCriteria(tok, prompt_length=0, max_thinking_tokens=1)
    ids = tok.encode("<think>a</think><answer>x", add_special_tokens=False)
    assert bool(crit(_t(ids), scores=None)) is False  # type: ignore[arg-type]


def test_thinking_criteria_uses_last_open_tag() -> None:
    tok = _CharTokenizer()
    crit = MaxThinkingTokensCriteria(tok, prompt_length=0, max_thinking_tokens=3)
    # closed first think; open second think and exceed cap there
    ids = tok.encode("<think>a</think><answer>x<think>abcd", add_special_tokens=False)
    assert bool(crit(_t(ids), scores=None)) is True  # type: ignore[arg-type]
