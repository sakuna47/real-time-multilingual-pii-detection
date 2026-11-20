"""Validate JSONL files: python -m src.pii_data FILE.jsonl [...]"""
import sys

from .record import _main

sys.exit(_main(sys.argv[1:]))
