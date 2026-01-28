#!/usr/bin/env python3
"""Entry point for the Local LLM CLI demo.

Run with: python run.py [options] [question]
Or:       python -m app [options] [question]
"""

from app.cli import main

if __name__ == "__main__":
    main()
