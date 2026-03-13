#!/usr/bin/env python3
"""
CBZ to WebP Converter with TUI

This script converts images inside CBZ (Comic Book ZIP) files to WebP format
to reduce file size while maintaining quality. Features include:
- Beautiful Terminal UI with real-time progress tracking
- Multi-threaded parallel processing for speed
- Smart conversion (only converts if WebP is smaller)
- Memory system to skip already-converted files
- Automatic backups of conversion history

Created: 2026
License: Free for personal use
"""

import sys
from pathlib import Path

# Add src to path to allow importing the package
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(str(src_path))

from cbz_webp_converter.cli import run

if __name__ == "__main__":
    run()
