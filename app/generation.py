"""Generation: streamer, thinking-cap criteria, and response generation."""

import re
import time

import torch
from transformers import TextStreamer
from transformers.generation import StoppingCriteria, StoppingCriteriaList

# Check stopping criteria every N generated tokens to avoid GPU->CPU sync every step.
# .tolist() on a GPU tensor forces a sync; doing it every token killed throughput.
THINKING_CRITERIA_CHECK_EVERY = 32
_ANSWER_PREFIX_TAGS_RE = re.compile(r"^\s*(?:</think>|<answer>)\s*", re.DOTALL)


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

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.BoolTensor:
        gen_count = input_ids.shape[1] - self.prompt_length
        if (
            gen_count % THINKING_CRITERIA_CHECK_EVERY != 0
            and gen_count < self.max_thinking_tokens
        ):
            return torch.tensor([False], device=input_ids.device)

        # Robust: detect tags in decoded text (tokenization of "<think>" can vary).
        gen_ids = input_ids[0, self.prompt_length :].tolist()
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)

        last_open_pos = gen_text.rfind(self.THINK_OPEN)
        if last_open_pos == -1:
            return torch.tensor([False], device=input_ids.device)
        last_close_pos = gen_text.rfind(self.THINK_CLOSE)
        if last_close_pos > last_open_pos:
            return torch.tensor([False], device=input_ids.device)

        after_open = gen_text[last_open_pos + len(self.THINK_OPEN) :]
        if self.THINK_CLOSE in after_open:
            return torch.tensor([False], device=input_ids.device)

        # Count tokens *inside* the current open <think> block.
        think_token_count = len(
            self.tokenizer.encode(after_open, add_special_tokens=False)
        )
        return torch.tensor(
            [think_token_count >= self.max_thinking_tokens], device=input_ids.device
        )


def _strip_redundant_answer_prefix(text: str) -> str:
    """Strip repeated leading </think>/<answer> tags (model may emit them again in phase 2)."""
    out = text
    while True:
        m = _ANSWER_PREFIX_TAGS_RE.match(out)
        if not m:
            return out
        out = out[m.end() :]


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
            # We reached the thinking cap while still inside <think>.
            # Close it and explicitly start the answer section.
            insert_text = "</think>\n<answer>\n"
            insert_ids = tokenizer.encode(insert_text, add_special_tokens=False)

            # Streamer doesn't print "prompt"/input tokens, so we print and account for them manually.
            print(insert_text, end="", flush=True)
            if hasattr(streamer, "token_cache"):
                streamer.token_cache.extend(insert_ids)
            streamer.token_count += len(insert_ids)

            insert_tensor = torch.tensor(
                [insert_ids], device=output_ids.device, dtype=output_ids.dtype
            )
            continued_ids = torch.cat([output_ids, insert_tensor], dim=-1)
            remaining = max_tokens - (output_ids.shape[-1] - input_token_count) - len(insert_ids)
            if remaining > 0:
                # Phase 2: generate without streaming, then print a cleaned answer.
                # Token-level "bad words" is not reliable here because multiple tokenizations can decode
                # to the same string (so the model can still emit "</think><answer>" via alternate splits).
                phase2_start = time.perf_counter()
                phase2_ids = model.generate(
                    continued_ids,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.1,
                    repetition_penalty=1.05,
                    max_new_tokens=remaining,
                )
                phase2_end = time.perf_counter()
                new_ids = phase2_ids[0, continued_ids.shape[-1] :]
                new_text = tokenizer.decode(new_ids, skip_special_tokens=False)
                new_text = _strip_redundant_answer_prefix(new_text)

                # Print and record the cleaned continuation for history/metrics.
                print(new_text, end="", flush=True)
                new_text_ids = tokenizer.encode(new_text, add_special_tokens=False)
                if hasattr(streamer, "token_cache"):
                    streamer.token_cache.extend(new_text_ids)
                streamer.token_count += len(new_text_ids)
                # Extend timing to include phase 2 for overall throughput.
                if streamer.first_token_time is None:
                    streamer.first_token_time = streamer.start_time
                streamer.end_time = (streamer.end_time or streamer.start_time) + (phase2_end - phase2_start)
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
