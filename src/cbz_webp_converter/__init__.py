"""
CBZ WebP Converter Package

A Python CLI tool that converts images inside CBZ files to WebP format.
Features a terminal UI with real-time progress and multi-threaded processing.
"""

__version__ = "1.0.0"
__author__ = "CBZ WebP Converter Team"

from .config import Config
from .converter import main
from .memory import (
    backup_memory_file,
    load_conversion_memory,
    save_conversion_memory,
    is_file_converted,
    mark_file_converted,
    is_series_skipped,
)
from .image_processing import (
    convert_image_to_webp,
    process_single_file,
)
from .file_operations import process_cbz_file
from .ui import (
    create_header,
    create_current_status,
    create_stats_panel,
    create_layout,
    is_image_file,
    is_webp_file,
    console,
)

__all__ = [
    # Config
    "Config",
    # Converter
    "main",
    # Memory
    "backup_memory_file",
    "load_conversion_memory",
    "save_conversion_memory",
    "is_file_converted",
    "mark_file_converted",
    "is_series_skipped",
    # Image Processing
    "convert_image_to_webp",
    "process_single_file",
    # File Operations
    "process_cbz_file",
    # UI
    "create_header",
    "create_current_status",
    "create_stats_panel",
    "create_layout",
    "is_image_file",
    "is_webp_file",
    "console",
]
