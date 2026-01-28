"""Generation: streaming, thinking time cap, and response generation."""

import time

import torch
from transformers import TextStreamer
from transformers.generation.stopping_criteria import MaxTimeCriteria, StoppingCriteriaList, StopStringCriteria

# Stop strings for structured output
STOP_THINKING_ON = "<answer>"
STOP_ANSWER_ON = "</answer>"


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


def generate_response(
    model,
    tokenizer,
    streamer: TimingStreamer,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.1,
    max_thinking_seconds: float | None = 20.0,
    answer_max_chars: int | None = None,
) -> dict:
    """Generate a response from a conversation (list of {role, content}) and return metrics.

    The last message in messages must be from the user. Previous messages provide context
    for follow-up questions. Use streamer.get_generated_text() after the call to get the
    assistant reply for appending to conversation history.
    """
    if answer_max_chars is not None and answer_max_chars > 0:
        # Insert a strict-ish instruction without truncating output ourselves.
        # If the model can't fit, it should produce a shorter summary instead.
        #
        # Put it up-front as a system message (most chat templates prioritize earlier system content).
        constraint = (
            f"In <answer>, write at most {answer_max_chars} characters. "
            "If you cannot, write a shorter summary that fits. Do not add extra sections."
        )
        messages = [{"role": "system", "content": constraint}, *list(messages)]

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

    # Phase 1: allow thinking, but stop either when the model starts answering (<answer>)
    # or when we hit the thinking time cap.
    phase1_criteria = []
    if max_thinking_seconds is not None and max_thinking_seconds > 0:
        phase1_criteria.append(MaxTimeCriteria(max_time=float(max_thinking_seconds)))
    phase1_criteria.append(StopStringCriteria(tokenizer, STOP_THINKING_ON))
    gen_kwargs["stopping_criteria"] = StoppingCriteriaList(phase1_criteria)
    output_ids = model.generate(input_ids, **gen_kwargs)

    gen_text = tokenizer.decode(output_ids[0, input_token_count:], skip_special_tokens=False)
    if STOP_THINKING_ON not in gen_text:
        # Time cap hit before the model opened <answer>. Force-close and start answer.
        insert_text = "</think>\n<answer>\n"
        insert_ids = tokenizer.encode(insert_text, add_special_tokens=False)
        print(insert_text, end="", flush=True)
        if hasattr(streamer, "token_cache"):
            streamer.token_cache.extend(insert_ids)
        streamer.token_count += len(insert_ids)
        insert_tensor = torch.tensor([insert_ids], device=output_ids.device, dtype=output_ids.dtype)
        output_ids = torch.cat([output_ids, insert_tensor], dim=-1)

    # Phase 2: stream the answer, stopping cleanly at </answer>.
    remaining = max_tokens - (output_ids.shape[-1] - input_token_count)
    if remaining > 0:
        model.generate(
            output_ids,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.1,
            repetition_penalty=1.05,
            max_new_tokens=remaining,
            stopping_criteria=StoppingCriteriaList([StopStringCriteria(tokenizer, STOP_ANSWER_ON)]),
            streamer=streamer,
        )

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
