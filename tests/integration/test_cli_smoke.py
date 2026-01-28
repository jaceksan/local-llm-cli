import os
import subprocess

import pytest


def test_cli_smoke_offline_cached_model() -> None:
    """End-to-end smoke test (opt-in) that must not download anything.

    Run with:
      RUN_LLM_INTEGRATION_TESTS=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 make test-integration
    """
    if os.environ.get("RUN_LLM_INTEGRATION_TESTS") != "1":
        pytest.skip("Set RUN_LLM_INTEGRATION_TESTS=1 to enable integration tests.")

    env = dict(os.environ)
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"

    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "run.py",
            "--max-thinking-tokens",
            "16",
            "--max-tokens",
            "96",
            "Briefly explain what is BAU",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )

    if proc.returncode != 0:
        pytest.fail(f"CLI failed in offline mode.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

    assert "<answer>" in proc.stdout
