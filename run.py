#!/usr/bin/env python3
"""CLI for interacting with LiquidAI LFM2.5 model."""

import argparse
import platform
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def detect_device() -> tuple[str, str]:
    """Detect the best available device and appropriate dtype.

    Returns:
        Tuple of (device, dtype) - device name and recommended dtype.
    """
    if torch.cuda.is_available():
        return "cuda", "bfloat16"
    elif torch.backends.mps.is_available():
        # Apple Silicon (M1-M4) - MPS backend
        # float16 is more stable on MPS than bfloat16
        return "mps", "float16"
    else:
        return "cpu", "float32"


def get_device_info(device: str) -> str:
    """Get human-readable device information."""
    if device == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(0)})"
    elif device == "mps":
        return f"Apple Silicon MPS ({platform.processor()})"
    else:
        return "CPU"


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


def load_model(model_id: str, device: str, dtype: str):
    """Load model and tokenizer.

    Args:
        model_id: Hugging Face model identifier.
        device: Target device (cuda, mps, cpu, or auto).
        dtype: Data type (bfloat16, float16, float32).
    """
    device_info = get_device_info(device) if device != "auto" else "auto-detect"
    print(f"Loading model: {model_id}")
    print(f"Device: {device_info}, dtype: {dtype}")

    # Build model kwargs
    model_kwargs = {"dtype": dtype}

    if device == "auto":
        # Let transformers/accelerate decide
        model_kwargs["device_map"] = "auto"
    elif device == "mps":
        # MPS requires explicit device placement after loading
        model_kwargs["device_map"] = {"": "mps"}
    elif device == "cuda":
        model_kwargs["device_map"] = "auto"
        # Uncomment for flash attention on compatible GPUs:
        # model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        # CPU
        model_kwargs["device_map"] = {"": "cpu"}

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    actual_device = str(model.device)
    print(f"Model loaded on: {actual_device}")

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    streamer: TimingStreamer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.1,
) -> dict:
    """Generate a response and return metrics."""
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = inputs["input_ids"].to(model.device)
    input_token_count = input_ids.shape[-1]

    streamer.reset()

    model.generate(
        input_ids=input_ids,
        do_sample=True,
        temperature=temperature,
        top_k=50,
        top_p=0.1,
        repetition_penalty=1.05,
        max_new_tokens=max_tokens,
        streamer=streamer,
    )

    metrics = streamer.get_metrics()
    metrics["input_tokens"] = input_token_count
    return metrics


def print_metrics(metrics: dict):
    """Print generation metrics in a formatted way."""
    print("\n" + "=" * 50)
    print("Performance Metrics:")
    print(f"  Input tokens:         {metrics.get('input_tokens', 'N/A')}")
    print(f"  Output tokens:        {metrics.get('output_tokens', 'N/A')}")
    print(f"  Time to first token:  {metrics.get('time_to_first_token_s', 0):.3f}s")
    print(f"  Generation time:      {metrics.get('generation_time_s', 0):.3f}s")
    print(f"  Total time:           {metrics.get('total_time_s', 0):.3f}s")
    print(f"  Tokens/second:        {metrics.get('tokens_per_second', 0):.2f}")
    print("=" * 50)


def interactive_mode(model, tokenizer, streamer: TimingStreamer, args):
    """Run interactive chat mode."""
    print("\nInteractive mode. Type 'quit' or 'exit' to stop, 'help' for commands.\n")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        if prompt.lower() == "help":
            print("Commands:")
            print("  quit/exit/q  - Exit the program")
            print("  help         - Show this help")
            print("  Any other text will be sent to the model")
            continue

        print("\nAssistant: ", end="", flush=True)
        metrics = generate_response(
            model,
            tokenizer,
            streamer,
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print_metrics(metrics)


def main():
    # Detect best device before parsing args (for help text)
    default_device, default_dtype = detect_device()

    parser = argparse.ArgumentParser(
        description="CLI for LiquidAI LFM2.5 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s                           # Interactive mode (auto-detect device)
  %(prog)s "What is Python?"         # Single question
  %(prog)s -t 0.7 "Tell me a joke"   # With higher temperature
  %(prog)s --device mps              # Force Apple Silicon GPU
  %(prog)s --device cuda             # Force NVIDIA GPU

Detected device: {default_device} (dtype: {default_dtype})
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (if not provided, starts interactive mode)",
    )
    parser.add_argument(
        "-m", "--model",
        default="LiquidAI/LFM2.5-1.2B-Thinking",
        help="Model ID (default: LiquidAI/LFM2.5-1.2B-Thinking)",
    )
    parser.add_argument(
        "-d", "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default=None,
        help="Data type (default: auto based on device)",
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )

    args = parser.parse_args()

    # Resolve device and dtype
    if args.device == "auto":
        device, auto_dtype = detect_device()
    else:
        device = args.device
        # Set recommended dtype per device if not specified
        if args.device == "mps":
            auto_dtype = "float16"
        elif args.device == "cuda":
            auto_dtype = "bfloat16"
        else:
            auto_dtype = "float32"

    dtype = args.dtype if args.dtype else auto_dtype

    model, tokenizer = load_model(args.model, device, dtype)
    streamer = TimingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if args.question:
        # Single question mode
        print(f"\nQuestion: {args.question}\n")
        print("Assistant: ", end="", flush=True)
        metrics = generate_response(
            model,
            tokenizer,
            streamer,
            args.question,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print_metrics(metrics)
    else:
        # Interactive mode
        interactive_mode(model, tokenizer, streamer, args)


if __name__ == "__main__":
    main()
