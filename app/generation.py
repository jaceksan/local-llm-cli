"""Generation: streamer, thinking-cap criteria, and response generation."""

import time

import torch
from transformers import TextStreamer
from transformers.generation import StoppingCriteria, StoppingCriteriaList

# Check stopping criteria every N generated tokens to avoid GPU->CPU sync every step.
# .tolist() on a GPU tensor forces a sync; doing it every token killed throughput.
THINKING_CRITERIA_CHECK_EVERY = 32


class TimingStreamer(TextStreamer):
    """Streamer that tracks token generation timing."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.token_count = 0
        self.start_time = None
        self.first_token_time = None
        self.end_time = None

    def reset(self):
        self.token_count = 0
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.end_time = None

    def put(self, value):
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
        self.token_count += value.shape[-1] if hasattr(value, "shape") else 1
        super().put(value)

    def end(self):
        self.end_time = time.perf_counter()
        super().end()

    def get_metrics(self) -> dict:
        """Return generation metrics."""
        if self.start_time is None or self.end_time is None:
            return {}
        total_time = self.end_time - self.start_time
        time_to_first = self.first_token_time - self.start_time if self.first_token_time else 0
        generation_time = self.end_time - self.first_token_time if self.first_token_time else total_time
        return {
            "output_tokens": self.token_count,
            "total_time_s": total_time,
            "time_to_first_token_s": time_to_first,
            "generation_time_s": generation_time,
            "tokens_per_second": self.token_count / generation_time if generation_time > 0 else 0,
        }

    def get_generated_text(self) -> str:
        """Return the full generated text (assistant reply) from the last run."""
        decode_kwargs = getattr(self, "decode_kwargs", None) or {"skip_special_tokens": True}
        return self.tokenizer.decode(self.token_cache, **decode_kwargs)


class MaxThinkingTokensCriteria(StoppingCriteria):
    """Stop generation after max_thinking_tokens inside a <think>...</think> block."""

    THINK_OPEN = "<think>"
    THINK_CLOSE = "</think>"

    def __init__(self, tokenizer, prompt_length: int, max_thinking_tokens: int):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.max_thinking_tokens = max_thinking_tokens
        self._think_open_ids = tokenizer.encode(self.THINK_OPEN, add_special_tokens=False)
        self._think_close_ids = tokenizer.encode(self.THINK_CLOSE, add_special_tokens=False)

    def _find_last_occurrence(self, seq: list, sub: list) -> int | None:
        if not sub or len(sub) > len(seq):
            return None
        for i in range(len(seq) - len(sub), -1, -1):
            if seq[i : i + len(sub)] == sub:
                return i
        return None

    def _contains_sequence(self, seq: list, sub: list) -> bool:
        if not sub or len(sub) > len(seq):
            return False
        return any(seq[i : i + len(sub)] == sub for i in range(len(seq) - len(sub) + 1))

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.BoolTensor:
        gen_count = input_ids.shape[1] - self.prompt_length
        if (
            gen_count % THINKING_CRITERIA_CHECK_EVERY != 0
            and gen_count < self.max_thinking_tokens
        ):
            return torch.tensor([False], device=input_ids.device)
        gen = input_ids[0, self.prompt_length :].tolist()
        last_open = self._find_last_occurrence(gen, self._think_open_ids)
        if last_open is None:
            return torch.tensor([False], device=input_ids.device)
        think_start = last_open + len(self._think_open_ids)
        after_think = gen[think_start:]
        if self._contains_sequence(after_think, self._think_close_ids):
            return torch.tensor([False], device=input_ids.device)
        return torch.tensor([len(after_think) >= self.max_thinking_tokens], device=input_ids.device)


def generate_response(
    model,
    tokenizer,
    streamer: TimingStreamer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_thinking_tokens: int | None = None,
) -> dict:
    """Generate a response from a conversation (list of {role, content}) and return metrics.

    The last message in messages must be from the user. Previous messages provide context
    for follow-up questions. Use streamer.get_generated_text() after the call to get the
    assistant reply for appending to conversation history.
    """
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    input_token_count = input_ids.shape[-1]
    streamer.reset()

    gen_kwargs: dict = {
        "do_sample": True,
        "temperature": temperature,
        "top_k": 50,
        "top_p": 0.1,
        "repetition_penalty": 1.05,
        "max_new_tokens": max_tokens,
        "streamer": streamer,
    }
    if max_thinking_tokens is not None and max_thinking_tokens > 0:
        criteria = MaxThinkingTokensCriteria(
            tokenizer, input_token_count, max_thinking_tokens
        )
        gen_kwargs["stopping_criteria"] = StoppingCriteriaList([criteria])
        output_ids = model.generate(input_ids, **gen_kwargs)
        gen_text = tokenizer.decode(
            output_ids[0, input_token_count:], skip_special_tokens=False
        )
        if gen_text.count("<think>") > gen_text.count("</think>"):
            think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
            think_close_tensor = torch.tensor(
                [think_close_ids], device=output_ids.device, dtype=output_ids.dtype
            )
            continued_ids = torch.cat([output_ids, think_close_tensor], dim=-1)
            remaining = max_tokens - (output_ids.shape[-1] - input_token_count)
            if remaining > 0:
                model.generate(
                    continued_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.1,
                    repetition_penalty=1.05,
                    max_new_tokens=remaining,
                    streamer=streamer,
                )
    else:
        model.generate(input_ids, **gen_kwargs)

    metrics = streamer.get_metrics()
    metrics["input_tokens"] = input_token_count
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print generation metrics to stdout."""
    print("\n" + "=" * 50)
    print("Performance Metrics:")
    print(f"  Input tokens:         {metrics.get('input_tokens', 'N/A')}")
    print(f"  Output tokens:        {metrics.get('output_tokens', 'N/A')}")
    print(f"  Time to first token:  {metrics.get('time_to_first_token_s', 0):.3f}s")
    print(f"  Generation time:      {metrics.get('generation_time_s', 0):.3f}s")
    print(f"  Total time:           {metrics.get('total_time_s', 0):.3f}s")
    print(f"  Tokens/second:        {metrics.get('tokens_per_second', 0):.2f}")
    print("=" * 50)
