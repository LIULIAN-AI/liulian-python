#!/usr/bin/env python3
"""Compatibility wrapper for full doc updates.

This script is kept for backward compatibility; it delegates to
`update_doc.py --mode full`.
"""

from update_doc import main


if __name__ == "__main__":
    main(["--mode", "full"])
