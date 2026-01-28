"""CLI: argument parsing, main entry, and interactive mode."""

import argparse

from app.generation import TimingStreamer, generate_response, print_metrics
from app.model import detect_device, load_model


def build_parser(default_device: str, default_dtype: str) -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description="CLI for LiquidAI LFM2.5 model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s                           # Interactive mode (auto-detect device)
  %(prog)s "What is Python?"         # Single question
  %(prog)s -t 0.7 "Tell me a joke"   # With higher temperature
  %(prog)s --max-thinking-tokens 128 # Shorter reasoning cap (default: 256)
  %(prog)s --max-thinking-tokens 0   # No thinking cap
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
    parser.add_argument(
        "--max-thinking-tokens",
        type=int,
        default=256,
        metavar="N",
        help="Cap reasoning inside <think>...</think> at N tokens (default: 256). Use 0 for no cap.",
    )
    return parser


def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Parse args and resolve device/dtype."""
    args = parser.parse_args()
    if args.device == "auto":
        device, auto_dtype = detect_device()
    else:
        device = args.device
        auto_dtype = "float16" if args.device == "mps" else "bfloat16" if args.device == "cuda" else "float32"
    args._device = device
    args._dtype = args.dtype if args.dtype is not None else auto_dtype
    return args


def interactive_mode(model, tokenizer, streamer: TimingStreamer, args: argparse.Namespace) -> None:
    """Run interactive chat loop with conversation context (follow-up questions)."""
    print("\nInteractive mode. Type 'quit' or 'exit' to stop, 'help' for commands.\n")
    history: list[dict] = []
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
            print("  clear        - Reset conversation (forget previous Q/A)")
            print("  help         - Show this help")
            print("  Any other text will be sent to the model (context from previous turns kept)")
            continue
        if prompt.lower() == "clear":
            history = []
            print("Conversation cleared.")
            continue
        history.append({"role": "user", "content": prompt})
        print("\nAssistant: ", end="", flush=True)
        metrics = generate_response(
            model,
            tokenizer,
            streamer,
            history,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_thinking_tokens=args.max_thinking_tokens,
        )
        reply = streamer.get_generated_text()
        history.append({"role": "assistant", "content": reply})
        print_metrics(metrics)


def main() -> None:
    default_device, default_dtype = detect_device()
    parser = build_parser(default_device, default_dtype)
    args = parse_args(parser)

    model, tokenizer = load_model(args.model, args._device, args._dtype)
    streamer = TimingStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    if args.question:
        print(f"\nQuestion: {args.question}\n")
        print("Assistant: ", end="", flush=True)
        messages = [{"role": "user", "content": args.question}]
        metrics = generate_response(
            model,
            tokenizer,
            streamer,
            messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            max_thinking_tokens=args.max_thinking_tokens,
        )
        print_metrics(metrics)
    else:
        interactive_mode(model, tokenizer, streamer, args)
