"""
Main conversion orchestration for CBZ WebP Converter.

Handles folder processing and overall conversion workflow.
"""

# Standard library
import sys
from pathlib import Path

# Rich library for beautiful terminal UI
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# Local imports
from .memory import (
    backup_memory_file,
    load_conversion_memory,
    save_conversion_memory,
    is_file_converted,
    mark_file_converted,
    is_series_skipped,
    memory_save_lock,
)
from .file_operations import process_cbz_file
from .ui import (
    create_header,
    create_current_status,
    create_stats_panel,
    create_layout,
    console,
)
from .config import Config

# Global state for thread-safe updates
from threading import Lock

progress_lock = Lock()

current_state = {
    "folder": "Waiting...",
    "cbz": "Waiting...",
    "stats": {
        "total_processed": 0,
        "total_saved": 0,
        "images_converted": 0,
        "folders_processed": 0,
        "folders_skipped": 0,
        "files_skipped_cached": 0,
    },
}

def main(config: Config):
    """
    Main entry point for the CBZ to WebP converter.

    This function orchestrates the entire conversion process:
    1. Parse command-line arguments
    2. Load folder list and conversion memory
    3. Set up the Terminal UI
    4. Process each folder and its CBZ files
    5. Display final statistics

    The function uses Rich's Live display for a real-time updating TUI
    that shows progress at three levels: folders, CBZ files, and images.
    """
    # === COMMAND-LINE ARGUMENT PARSING ===
    # Set up argument parser with description
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

    # Optional parameter: WebP quality level (affects compression vs. image quality)
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

    # Optional flag: force re-conversion of all files (ignore memory)
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

    # Parse the command-line arguments
    # args are now passed via config

    # === VALIDATE ARGUMENTS ===
    # Ensure quality is in valid range (1-100)
    if not 1 <= config.quality <= 100:
        console.print("[red]Error: Quality must be between 1 and 100[/red]")
        sys.exit(1)

    # Ensure thread count is reasonable (1-32)
    # Too many threads can cause excessive context switching
    if not 1 <= config.threads <= 32:
        console.print("[red]Error: Threads must be between 1 and 32[/red]")
        sys.exit(1)

    # === LOAD FOLDER LIST ===
    # Get the directory where the script is located
    # This is where to_convert.txt and conversion_memory.json are stored
    script_dir = Path(__file__).parent
    folder_list_file = script_dir / "to_convert.txt"

    # Validate that the target directory exists
    base_dir = Path(config.directory)

    if not base_dir.exists():
        console.print(f"[red]Error: Directory {config.directory} does not exist![/red]")
        sys.exit(1)

    # Try to read folder names from to_convert.txt
    # If file doesn't exist or is empty, we'll scan all folders in the directory
    folder_names = []
    if folder_list_file.exists():
        with open(folder_list_file, "r", encoding="utf-8") as f:
            # Read lines, strip whitespace, skip empty lines
            folder_names = [line.strip() for line in f if line.strip()]

    # If no folders specified, auto-discover all folders in target directory
    if not folder_names:
        # Get all subdirectories, but filter out:
        # - Hidden folders (starting with '.')
        # - System folders like .Trash-1000
        folder_names = [
            f.name
            for f in sorted(base_dir.iterdir())
            if f.is_dir() and not f.name.startswith(".") and f.name != ".Trash-1000"
        ]

        # Ensure we found at least one folder
        if not folder_names:
            console.print(f"[red]Error: No folders found in {config.directory}![/red]")
            sys.exit(1)

        # Inform user we're using auto-discovery mode
        console.print(
            f"[yellow]to_convert.txt not found or empty - processing all {len(folder_names)} folders in directory[/yellow]"
        )
    else:
        # Inform user how many folders were loaded from file
        console.print(
            f"[cyan]Loaded {len(folder_names)} folders from to_convert.txt[/cyan]"
        )

    # === LOAD CONVERSION MEMORY ===
    # The conversion memory tracks which files have already been converted
    # This prevents redundant processing when re-running the script
    memory_file = script_dir / "conversion_memory.json"

    # Create timestamped backup before loading (protects against data loss)
    backup_file = backup_memory_file(memory_file)
    if backup_file:
        console.print(f"[dim cyan]Backup created: {backup_file.name}[/dim cyan]")

    # Load the memory file (always load it, even with --force flag)
    # This ensures we preserve existing entries when saving
    conversion_memory = load_conversion_memory(memory_file)

    # Track whether we're forcing re-conversion
    # When True: process all files regardless of memory (but still save to memory)
    # When False: skip files that are already in memory
    force_reconvert = config.force

    # Display memory status to user
    if config.force and memory_file.exists():
        console.print(
            "[yellow]--force flag set: Will re-convert all files (memory preserved)[/yellow]"
        )
    elif memory_file.exists():
        # Calculate total files tracked in memory
        total_cached = sum(
            len(series.get("files", [])) for series in conversion_memory.values()
        )
        console.print(
            f"[cyan]Loaded conversion memory: {len(conversion_memory)} series, {total_cached} files already converted[/cyan]"
        )

        # Verbose mode: show detailed memory contents
        if config.verbose:
            console.print("\n[bold]Memory contents:[/bold]")
            for series_name, data in sorted(conversion_memory.items()):
                file_count = len(data.get("files", []))
                skip_status = " [red](SKIP)[/red]" if data.get("skip", False) else ""
                console.print(f"  • {series_name}: {file_count} files{skip_status}")
            console.print()

    # Create the layout
    layout = create_layout(args, len(folder_names))

    # Create progress bars
    folder_progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        expand=True,
    )

    cbz_progress = Progress(
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        expand=True,
    )

    image_progress = Progress(
        TextColumn("[bold yellow]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        expand=True,
    )

    # Combine progress bars into a single panel
    progress_table = Table.grid(padding=(1, 2))
    progress_table.add_row(folder_progress)
    progress_table.add_row(cbz_progress)
    progress_table.add_row(image_progress)

    progress_panel = Panel(
        progress_table, title="[bold]Progress", border_style="cyan", box=box.ROUNDED
    )

    layout["progress"].update(progress_panel)

    # Start live display with reduced refresh rate
    with Live(layout, console=console, screen=False, refresh_per_second=4):
        # Create main folder task
        folder_task = folder_progress.add_task("Folders", total=len(folder_names))

        # === MAIN PROCESSING LOOP ===
        # Process each folder in the list
        for folder_name in folder_names:
            folder_path = base_dir / folder_name

            # Check if this series is marked to skip in the memory file
            # Users can manually set "skip": true in the JSON to ignore certain series
            if is_series_skipped(conversion_memory, folder_name):
                with progress_lock:
                    current_state["folder"] = folder_name
                    current_state["cbz"] = "Skipped (skip=true in memory)"
                    current_state["stats"]["folders_skipped"] += 1
                    folder_progress.update(folder_task, advance=1)
                layout["status"].update(create_current_status())
                layout["stats"].update(create_stats_panel())
                continue  # Skip to next folder

            # Update TUI to show which folder we're currently processing
            with progress_lock:
                current_state["folder"] = folder_name
                current_state["cbz"] = "Scanning folder..."
            layout["status"].update(create_current_status())

            # Validate that the folder exists
            if not folder_path.exists():
                with progress_lock:
                    current_state["stats"]["folders_skipped"] += 1
                    folder_progress.update(folder_task, advance=1)
                    # Show why folder was skipped (helpful for debugging)
                    current_state["cbz"] = f"Folder not found: {folder_path}"
                    layout["stats"].update(create_stats_panel())
                    layout["status"].update(create_current_status())
                continue  # Skip to next folder

            # Validate that it's actually a directory (not a file)
            if not folder_path.is_dir():
                with progress_lock:
                    current_state["stats"]["folders_skipped"] += 1
                    folder_progress.update(folder_task, advance=1)
                    current_state["cbz"] = f"Not a directory: {folder_path}"
                    layout["stats"].update(create_stats_panel())
                    layout["status"].update(create_current_status())
                continue  # Skip to next folder

            # --- PATCH 4: ensure series exists in memory only if it's a valid directory ---
            # Use mark_file_converted which handles the lock and creates series if missing
            with memory_save_lock:
                if folder_name not in conversion_memory:
                    conversion_memory[folder_name] = {"files": [], "skip": False}

            # Find all CBZ files in this folder
            # CBZ files are comic book archives (just ZIP files with images)
            all_cbz_files = list(folder_path.glob("*.cbz"))
            # Filter out backup files (ending with _OLD from previous runs)
            cbz_files_no_old = [f for f in all_cbz_files if not f.stem.endswith("_OLD")]

            # Apply conversion memory filtering (unless --force flag is set)
            # This is the "smart skip" feature that avoids re-converting files
            if force_reconvert:
                # Force mode: process everything regardless of memory
                cbz_files = cbz_files_no_old
                if config.verbose:
                    console.print(
                        f"[dim]--force: Processing all {len(cbz_files)} files in {folder_name}[/dim]"
                    )
            else:
                # Normal mode: skip files that are already in the conversion memory
                if config.verbose:
                    console.print(
                        f"[dim]Checking memory for {folder_name}: {len(conversion_memory.get(folder_name, {}).get('files', []))} files in cache[/dim]"
                    )

                # cbz_files = [
                #     f
                #     for f in cbz_files_no_old
                #     if not is_file_converted(conversion_memory, folder_name, f.name)
                # ]

                cbz_files = []
                for f in cbz_files_no_old:
                    if is_file_converted(conversion_memory, folder_name, f.name):
                        continue

                    # PATCH: check if CBZ already contains only WebP images (skip if so)
                    try:
                        with ZipFile(f, "r") as z:
                            names = z.namelist()
                            # Get all image files (including WebP)
                            image_files = [
                                name for name in names
                                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"}
                            ]
                            # Check if all image files are already WebP
                            if image_files and all(
                                Path(name).suffix.lower() == ".webp"
                                for name in image_files
                            ):
                                mark_file_converted(
                                    conversion_memory, folder_name, f.name
                                )
                                save_conversion_memory(memory_file, conversion_memory)
                                continue
                    except Exception as e:
                        # Log the error so user knows about potentially corrupt CBZ files
                        console.print(
                            f"[yellow]Warning: Could not check {f.name}: {type(e).__name__}: {e}[/yellow]"
                        )

                    cbz_files.append(f)

                # Track how many files we skipped due to cache
                skipped_count = len(cbz_files_no_old) - len(cbz_files)
                if skipped_count > 0:
                    with progress_lock:
                        current_state["stats"]["files_skipped_cached"] += skipped_count
                    if config.verbose:
                        console.print(
                            f"[dim green]Skipped {skipped_count} cached files in {folder_name}[/dim green]"
                        )
                        for f in cbz_files_no_old:
                            if is_file_converted(
                                conversion_memory, folder_name, f.name
                            ):
                                console.print(f"[dim]  ✓ {f.name} (in cache)[/dim]")

                if config.verbose and len(cbz_files) > 0:
                    console.print(
                        f"[dim yellow]Will process {len(cbz_files)} files in {folder_name}:[/dim yellow]"
                    )
                    for f in cbz_files:
                        console.print(f"[dim]  → {f.name}[/dim]")

            if not cbz_files:
                with progress_lock:
                    current_state["stats"]["folders_skipped"] += 1
                    folder_progress.update(folder_task, advance=1)
                    if all_cbz_files:
                        if len(cbz_files_no_old) == 0:
                            current_state["cbz"] = (
                                f"Only _OLD files found ({len(all_cbz_files)})"
                            )
                        else:
                            current_state["cbz"] = (
                                f"All files already converted ({len(cbz_files_no_old)})"
                            )
                    else:
                        current_state["cbz"] = "No CBZ files found"
                    layout["stats"].update(create_stats_panel())
                    layout["status"].update(create_current_status())
                continue

            # Create CBZ progress task for this folder
            cbz_task = cbz_progress.add_task(
                f"CBZ files in {folder_name}", total=len(cbz_files)
            )

            # Process each CBZ file
            for idx, cbz_file in enumerate(cbz_files, 1):
                process_cbz_file(
                    cbz_file,
                    config.delete,
                    config.quality,
                    config.threads,
                    folder_progress,
                    cbz_progress,
                    image_progress,
                    folder_task,
                    cbz_task,
                    conversion_memory,
                    folder_name,
                    memory_file,
                )

                # Update stats display only every 3 files or on last file to reduce flickering
                if idx % 3 == 0 or idx == len(cbz_files):
                    layout["stats"].update(create_stats_panel())
                    layout["status"].update(create_current_status())

            # Remove CBZ task for this folder
            cbz_progress.remove_task(cbz_task)

            # Update folder progress and stats
            with progress_lock:
                current_state["stats"]["folders_processed"] += 1
                folder_progress.update(folder_task, advance=1)
            layout["stats"].update(create_stats_panel())
            layout["status"].update(create_current_status())

        # Mark as complete
        with progress_lock:
            current_state["folder"] = "Complete"
            current_state["cbz"] = "Complete"
        layout["status"].update(create_current_status())

    # Print final summary
    console.print("\n[bold green]✓ Conversion complete![/bold green]")
    stats = current_state["stats"]
    console.print(f"[cyan]Total CBZ files processed:[/cyan] {stats['total_processed']}")
    console.print(
        f"[cyan]Total space saved:[/cyan] {stats['total_saved'] / (1024 * 1024):.1f} MB"
    )
    console.print(f"[cyan]Total images converted:[/cyan] {stats['images_converted']}")


if __name__ == "__main__":
    main()
