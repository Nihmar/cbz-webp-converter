# AGENTS.md - CBZ WebP Converter

## Project Overview

Python CLI tool that converts images inside CBZ files to WebP format. Features a terminal UI with real-time progress and multi-threaded processing.

- **Language**: Python 3.6+
- **Dependencies**: Pillow, Rich
- **Main entry**: `src/cbz_webp_converter_tui.py`

---

## Build / Run Commands

### Running the Converter

```bash
# Install dependencies
pip install -r requirements.txt

# Run the converter
python src/cbz_webp_converter_tui.py /path/to/comics [options]

# Options:
#   --delete        Delete original CBZ (default: rename to _OLD.cbz)
#   --quality N     WebP quality 1-100 (default: 90)
#   --threads N     Parallel threads 1-32 (default: 8)
#   --force         Re-convert all, ignoring memory
#   --verbose       Show detailed debug output
```

### Diagnostic Script

```bash
python src/diagnose_folders.py /path/to/comics
```

### Testing / Linting

**Currently there are NO tests, linting, or type checking configured.**

If added, typical commands:
```bash
pytest                    # Run all tests
pytest tests/test_converter.py::test_name  # Single test
pytest --cov=src --cov-report=html  # With coverage
ruff check src/          # Lint
ruff format src/         # Format
mypy src/                # Type check
```

---

## Code Style Guidelines

### Naming Conventions

- **Functions/variables**: `snake_case` (`convert_image_to_webp`, `cbz_path`)
- **Constants**: `UPPER_SNAKE_CASE` (`ZIP_DEFLATED`)
- **Classes**: `PascalCase` (`Console`, `Layout`)
- **Private functions**: `_leading_underscore`

### Type Hints

Minimal type hints exist. Add them when possible:
```python
def process_cbz_file(cbz_path: Path, quality: int) -> bool:
```

### Import Organization

Group imports (blank line between groups):
```python
# Standard library
import argparse
import json
import os

# Multi-threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ZIP/file handling
from zipfile import ZIP_DEFLATED, ZipFile
from pathlib import Path
from io import BytesIO

# Third-party
from PIL import Image
from rich.console import Console
from rich.progress import Progress, BarColumn
```

### Docstrings

Use Google-style with Args/Returns:
```python
def function_name(param1: str, param2: int) -> bool:
    """
    Short one-line description.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        True if successful, False otherwise.
    """
```

### Error Handling

- Use try/except for operations that can fail
- Log errors with Rich console output
- Never lose user data - keep original files on failure
- Use specific exception types:
  ```python
  except (json.JSONDecodeError, IOError):
  ```

### Thread Safety

- Use `threading.Lock()` for shared state
- Acquire locks before modifying:
  ```python
  with progress_lock:
      current_state["stats"]["total_processed"] += 1
  ```

### File Handling

- Use `pathlib.Path` for paths
- Use context managers (`with` statements)
- Use atomic writes (write to temp, then rename)
- Use `encoding="utf-8"` for text files

### Rich Terminal UI

- Use `console.print()` with colored markup
- Colors: `[red]error[/red]`, `[green]success[/green]`, `[yellow]warning[/yellow]`
- Use Panels, Tables, and Progress bars

---

## Project Patterns

### Conversion Memory
- File: `conversion_memory.json`
- Structure: `{"SeriesName": {"files": [...], "skip": false}}`
- Atomic writes with backup

### to_convert.txt
- One folder name per line; if missing/empty, processes all subdirectories

### Image Processing
- Convert to WebP only if smaller; handle alpha channels (RGBA, LA) separately

---

## Adding Features

1. Ensure thread safety for shared state
2. Test with --threads > 1
3. Preserve backward compatibility with conversion_memory.json
4. Use Rich for all user output
5. Add docstrings to new functions

---

## Quick Reference

| Task | Command |
|------|---------|
| Run | `python src/cbz_webp_converter_tui.py /path --threads 8` |
| Diagnose | `python src/diagnose_folders.py /path` |
| Install | `pip install -r requirements.txt` |

---

## File Structure

```
cbz-webp-converter/
├── src/
│   ├── cbz_webp_converter_tui.py   # Main converter
│   └── diagnose_folders.py         # Diagnostic
├── requirements.txt
├── README.md
└── .gitignore
```
