# LiquidAI LFM2.5 Demo

CLI for testing the LiquidAI LFM2.5-1.2B-Thinking model using Hugging Face Transformers.

## Setup

```bash
make dev
```

## Usage

### Interactive mode

```bash
uv run python run.py
```

This starts an interactive chat where you can ask multiple questions. **Conversation context is kept**: follow-up questions see previous questions and answers, so you can ask things like "What is BAU?" and then "Explain that in one sentence." Type `clear` to reset the conversation and start fresh; `help` lists all commands.

### Single question

```bash
uv run python run.py "What is Python?"
```

### Options

```bash
uv run python run.py --help

# With custom temperature
uv run python run.py -t 0.7 "Tell me a joke"

# With more output tokens
uv run python run.py --max-tokens 1024 "Explain quantum computing"

# Cap thinking time (prevents “endless thinking”)
uv run python run.py --max-thinking-seconds 10 "Explain quantum computing"

# Ask the model to keep the answer short (no truncation; default is 256)
uv run python run.py --answer-max-chars 256 "Explain quantum computing"

# Force specific device
uv run python run.py --device mps "Hello"   # Apple Silicon GPU
uv run python run.py --device cuda "Hello"  # NVIDIA GPU
uv run python run.py --device cpu "Hello"   # CPU only
```

## Performance Metrics

After each response, the CLI displays:
- **Input/Output tokens**: Token counts for the prompt and response
- **Time to first token**: Latency before generation starts
- **Generation time**: Time spent generating tokens
- **Tokens/second**: Generation throughput

## Model

This demo uses [LiquidAI/LFM2.5-1.2B-Thinking](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking), a 1.2B parameter thinking model from LiquidAI.

The model will be downloaded automatically on first run (~2.4GB).

## Device Support

The CLI auto-detects the best available device:

| Platform | Device | dtype | Notes |
|----------|--------|-------|-------|
| Linux + NVIDIA GPU | `cuda` | bfloat16 | Fastest option |
| macOS + Apple Silicon (M1-M4) | `mps` | float16 | GPU acceleration via Metal |
| Linux/macOS (no GPU) | `cpu` | float32 | Slowest, but works everywhere |

### Apple Silicon Setup

On macOS with M1-M4 chips, the MPS (Metal Performance Shaders) backend is used automatically:

```bash
# Setup installs MPS-enabled PyTorch from PyPI
make dev

# Run - will auto-detect MPS
uv run python run.py "What is Python?"
```

### NVIDIA CUDA Setup

For Linux with NVIDIA GPU, edit `pyproject.toml` to use CUDA builds:

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cu124"  # Change to CUDA 12.4
explicit = true
```

Then reinstall:
```bash
rm uv.lock
make dev
```

## Project structure

Demo layout (single package, no PyPI publish):

```
app/
  __init__.py    # package entry
  __main__.py    # python -m app
  cli.py         # argument parsing, main(), interactive mode
  model.py       # device detection, load_model()
  generation.py   # TimingStreamer, time-capped thinking, stop-strings, generate_response()
run.py           # entry point: uv run python run.py
```

Run with `uv run python run.py` or `uv run python -m app`.

## Requirements

- Python 3.12+
- GPU recommended (CUDA or Apple Silicon MPS)
- ~5GB disk space for model cache
