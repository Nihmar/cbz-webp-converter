"""
Memory and file tracking operations for CBZ WebP Converter.

Handles conversion memory tracking, persistence, and backup operations.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from threading import Lock

from rich.console import Console

# Initialize Rich console for formatted output
console = Console()

# Global lock for memory file I/O operations
# This prevents multiple CBZ files from trying to save memory simultaneously
memory_save_lock = Lock()

def backup_memory_file(memory_file):
    """
    Create a timestamped backup of the memory file before loading it.

    This protects against data loss by creating a snapshot of the conversion
    history before any modifications are made. Backups are named with the
    current date and time (e.g., conversion_memory_20260130_143052.json).

    Args:
        memory_file (Path): Path to the memory JSON file to backup

    Returns:
        Path: Path to the created backup file, or None if backup failed

    Note:
        Failures are non-fatal - the script will continue even if backup fails
    """
    if memory_file.exists():
        try:
            # Create timestamp in format: yyyymmdd_HHMMSS
            # This ensures chronological sorting and uniqueness
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create backup filename by inserting timestamp before extension
            # Example: conversion_memory.json -> conversion_memory_20260130_143052.json
            backup_file = (
                memory_file.parent
                / f"{memory_file.stem}_{timestamp}{memory_file.suffix}"
            )

            # Copy the file with metadata preservation (timestamps, permissions)
            shutil.copy2(memory_file, backup_file)

            return backup_file
        except IOError as e:
            # Non-fatal error - warn user but continue processing
            console.print(
                f"[yellow]Warning: Could not create backup of memory file: {e}[/yellow]"
            )
            return None
    return None  # File doesn't exist, nothing to backup




def load_conversion_memory(memory_file):
    """
    Load the conversion memory from JSON file.

    The memory file tracks which files have been converted to avoid
    redundant processing. It also supports a 'skip' flag per series
    to permanently ignore certain folders.

    Args:
        memory_file (Path): Path to the conversion_memory.json file

    Returns:
        dict: Memory dictionary with structure:
              {
                  "Series Name": {
                      "files": ["file1.cbz", "file2.cbz"],
                      "skip": false
                  }
              }
              Returns empty dict {} if file doesn't exist or is corrupted

    Note:
        Automatically adds 'skip': false to old memory files for backward compatibility
        Attempts to restore from most recent backup if main file is corrupted
    """
    if memory_file.exists():
        try:
            # Read and parse JSON file
            with open(memory_file, "r", encoding="utf-8") as f:
                memory = json.load(f)

                # Ensure all series have the 'skip' property (backward compatibility)
                # Older versions of the script didn't include this property
                for series_name in memory:
                    if "skip" not in memory[series_name]:
                        memory[series_name]["skip"] = False
                return memory
        except (json.JSONDecodeError, IOError) as e:
            # If file is corrupted or unreadable, try to restore from backup
            console.print(
                f"[red]⚠ ERROR: Conversion memory file is corrupted![/red]"
            )
            console.print(f"[red]Reason: {type(e).__name__}: {e}[/red]")
            
            # Try to find and restore from most recent backup
            backup_dir = memory_file.parent
            backup_pattern = f"{memory_file.stem}_*{memory_file.suffix}"
            backups = sorted(
                backup_dir.glob(backup_pattern),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            
            if backups:
                latest_backup = backups[0]
                console.print(
                    f"[yellow]Attempting to restore from backup: {latest_backup.name}[/yellow]"
                )
                try:
                    with open(latest_backup, "r", encoding="utf-8") as f:
                        memory = json.load(f)
                        # Ensure backward compatibility
                        for series_name in memory:
                            if "skip" not in memory[series_name]:
                                memory[series_name]["skip"] = False
                        console.print(
                            f"[green]Successfully restored conversion memory from backup[/green]"
                        )
                        return memory
                except (json.JSONDecodeError, IOError) as backup_e:
                    console.print(
                        f"[red]✗ Failed to restore from backup: {type(backup_e).__name__}: {backup_e}[/red]"
                    )
            
            console.print(
                f"[yellow]Starting with empty conversion memory (all files will be reprocessed)[/yellow]"
            )
            return {}
    return {}  # File doesn't exist yet - first run




def save_conversion_memory(memory_file, memory):
    """
    Save the conversion memory to JSON file with atomic write.

    Called after each successful file conversion to persist progress.
    Uses atomic write pattern (write to temp, then rename) to prevent
    corruption if interrupted mid-write.

    CRITICAL: Uses global lock to prevent multiple concurrent writes
    which can cause data loss or corruption.

    Args:
        memory_file (Path): Path where to save conversion_memory.json
        memory (dict): Memory dictionary to save

    Note:
        Uses ensure_ascii=False to properly save Unicode characters in filenames
        Atomic writes ensure file is never in partially-written state
    """
    # CRITICAL: Acquire lock before any file operations
    # Multiple CBZ files finishing at the same time must wait in queue
    with memory_save_lock:
        try:
            # Write to temporary file first (atomic write pattern)
            temp_file = (
                memory_file.parent / f"{memory_file.stem}_temp{memory_file.suffix}"
            )

            with open(temp_file, "w", encoding="utf-8") as f:
                # indent=2 makes the JSON human-readable
                # ensure_ascii=False allows Unicode characters (e.g., Japanese titles)
                json.dump(memory, f, indent=2, ensure_ascii=False)
                f.flush()  # Ensure data is written to disk
                os.fsync(f.fileno())  # Force OS to write to disk (prevent buffer loss)

            # Atomic rename - ensures file is never corrupted
            # If process crashes during rename, we still have either old or new file
            if memory_file.exists():
                memory_file.unlink()  # Remove old file (Windows requires this)
            temp_file.rename(memory_file)  # Atomic on Unix, near-atomic on Windows

        except IOError as e:
            # Make errors VERY visible - this is critical data loss
            console.print(
                f"\n[bold red]⚠ ERROR: Could not save conversion memory![/bold red]"
            )
            console.print(f"[red]Reason: {e}[/red]")
            console.print(
                f"[yellow]Your conversion progress may not be saved![/yellow]"
            )
            console.print(f"[yellow]Check disk space and file permissions![/yellow]\n")
        except Exception as e:
            console.print(
                f"\n[bold red]⚠ UNEXPECTED ERROR saving memory: {e}[/bold red]\n"
            )




def is_file_converted(memory, series_name, file_name):
    """
    Check if a file has already been converted.

    Used to skip files that are already in the memory to avoid
    redundant processing on subsequent runs.

    Args:
        memory (dict): The conversion memory dictionary
        series_name (str): Name of the series (folder name)
        file_name (str): Name of the CBZ file

    Returns:
        bool: True if file exists in memory, False otherwise

    Example:
        >>> is_file_converted(memory, "Naruto", "Naruto_V001.cbz")
        True
    """
    if series_name in memory:
        # Use .get() with default [] to handle old memory files without 'files' key
        return file_name in memory[series_name].get("files", [])
    return False  # Series not in memory yet




def mark_file_converted(memory, series_name, file_name):
    """
    Mark a file as converted in memory (thread-safe).

    Adds the file to the conversion memory after successful processing.
    Creates the series entry if it doesn't exist.

    CRITICAL: Uses global lock to prevent race conditions when multiple
    threads try to modify the memory dictionary simultaneously.

    Args:
        memory (dict): The conversion memory dictionary (modified in-place)
        series_name (str): Name of the series (folder name)
        file_name (str): Name of the CBZ file that was converted

    Note:
        - Preserves existing 'skip' value if series already exists
        - Avoids duplicate entries in the files list
        - Memory is modified in-place and should be saved afterward
    """
    # Acquire lock to prevent concurrent modifications
    with memory_save_lock:
        if series_name not in memory:
            # Create new series entry with default values
            memory[series_name] = {"files": [], "skip": False}

        # Ensure skip property exists (for backward compatibility with old memory files)
        if "skip" not in memory[series_name]:
            memory[series_name]["skip"] = False

        # Add file to list if not already present (avoid duplicates)
        if file_name not in memory[series_name].get("files", []):
            memory[series_name]["files"].append(file_name)




def is_series_skipped(memory, series_name):
    """
    Check if a series is marked to be skipped.

    Users can manually edit conversion_memory.json and set "skip": true
    for series they want to permanently ignore.

    Args:
        memory (dict): The conversion memory dictionary
        series_name (str): Name of the series (folder name)

    Returns:
        bool: True if series should be skipped, False otherwise

    Use case:
        - Series already optimized manually
        - Series that don't compress well with WebP
        - Series user doesn't want to convert
    """
    if series_name in memory:
        # Default to False if 'skip' key doesn't exist (old memory files)
        return memory[series_name].get("skip", False)
    return False  # Series not in memory - don't skip




