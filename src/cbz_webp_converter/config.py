"""
Configuration and settings for CBZ WebP Converter.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configuration settings for the converter."""
    directory: str
    delete: bool = False
    quality: int = 90
    threads: int = 8
    force: bool = False
    verbose: bool = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert images in CBZ files to WebP format with TUI"
    )
    
    # Required positional argument: directory containing comic folders
    parser.add_argument(
        "directory", help="Directory containing the folders with CBZ files"
    )

    # Optional flag: delete originals instead of creating backups
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete original CBZ files instead of renaming to _OLD.cbz",
    )

    # Optional parameter: WebP quality level
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="WebP quality level (1-100, default: 90)",
    )

    # Optional parameter: number of parallel processing threads
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of parallel threads for image conversion (default: 8)",
    )

    # Optional flag: force re-conversion of all files
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-conversion of all files, ignoring conversion memory",
    )

    # Optional flag: verbose output for debugging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging memory and file tracking",
    )

    args = parser.parse_args()
    
    # Validate arguments
    if not 1 <= args.quality <= 100:
        raise ValueError("Quality must be between 1 and 100")
    
    if not 1 <= args.threads <= 32:
        raise ValueError("Threads must be between 1 and 32")
    
    return Config(
        directory=args.directory,
        delete=args.delete,
        quality=args.quality,
        threads=args.threads,
        force=args.force,
        verbose=args.verbose,
    )
