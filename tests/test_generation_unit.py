import torch

from app.generation import STOP_ANSWER_ON, STOP_THINKING_ON


class _CharTokenizer:
    """Tiny deterministic tokenizer for unit tests (no HF downloads)."""

    def encode(self, text: str, *, add_special_tokens: bool = False):  # noqa: ARG002
        return [ord(ch) for ch in text]

    def decode(self, ids, *, skip_special_tokens: bool = False):  # noqa: ARG002
        return "".join(chr(i) for i in ids)


def _t(ids: list[int]) -> torch.Tensor:
    return torch.tensor([ids], dtype=torch.long)


def test_stop_strings_constants() -> None:
    assert STOP_THINKING_ON == "</think>"
    assert STOP_ANSWER_ON == "</answer>"


def test_char_tokenizer_roundtrip() -> None:
    tok = _CharTokenizer()
    s = "<think>abc</think><answer>xyz</answer>"
    assert tok.decode(tok.encode(s, add_special_tokens=False)) == s


def test_tensor_helper() -> None:
    t = _t([1, 2, 3])
    assert isinstance(t, torch.Tensor)
    assert t.shape == (1, 3)
