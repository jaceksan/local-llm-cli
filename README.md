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

This starts an interactive chat where you can ask multiple questions.

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

## Requirements

- Python 3.12+
- GPU recommended (CUDA or Apple Silicon MPS)
- ~5GB disk space for model cache
