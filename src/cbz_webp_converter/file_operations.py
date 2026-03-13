"""
File operations for CBZ WebP Converter.

Handles CBZ file reading, writing, and processing.
"""

# Standard library
import os
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Rich console for warnings
from rich.console import Console

console = Console()

def process_cbz_file(
    cbz_path,
    delete_original,
    quality,
    max_workers,
    folder_progress,
    cbz_progress,
    image_progress,
    folder_task,
    cbz_task,
    memory,
    series_name,
    memory_file,
):
    """
    Process a single CBZ file: convert images to WebP and recompress.

    This is the main processing function that:
    1. Reads the CBZ file into memory
    2. Spawns worker threads to convert images in parallel
    3. Recompresses everything into a new CBZ with maximum compression
    4. Handles the original file (delete or rename to _OLD)
    5. Updates the conversion memory

    Args:
        cbz_path (Path): Path to the CBZ file to process
        delete_original (bool): If True, delete original; if False, rename to _OLD
        quality (int): WebP quality setting (1-100)
        max_workers (int): Number of parallel worker threads
        folder_progress (Progress): Progress bar for folder-level tracking
        cbz_progress (Progress): Progress bar for CBZ file tracking
        image_progress (Progress): Progress bar for image-level tracking
        folder_task: Task ID for folder progress bar
        cbz_task: Task ID for CBZ progress bar
        memory (dict): Conversion memory dictionary (tracks converted files)
        series_name (str): Name of the series/folder being processed
        memory_file (Path): Path to save updated memory

    Returns:
        bool: True if successful, False if error occurred
    """
    cbz_name = cbz_path.name

    # Update global state to show which file we're currently processing
    # This is displayed in the TUI status panel
    with progress_lock:
        current_state["cbz"] = cbz_name

    try:
        # === PHASE 1: Load CBZ file into memory ===
        # Get original file size for before/after comparison
        original_file_size = os.path.getsize(cbz_path)

        # Read the entire CBZ (ZIP) file into RAM
        # This avoids disk I/O during processing for better performance
        with open(cbz_path, "rb") as f:
            cbz_data = BytesIO(f.read())

        # === PHASE 2: Extract all files from CBZ ===
        # CBZ files are just ZIP archives with images
        # We need to extract everything before processing
        files_to_process = []
        with ZipFile(cbz_data, "r") as zip_read:
            file_list = zip_read.namelist()  # Get list of all files in archive
            for filename in file_list:
                # Read each file into memory as raw bytes
                file_data = zip_read.read(filename)
                files_to_process.append((filename, file_data))

        # Create a progress task for tracking image conversion within this CBZ
        # This shows real-time progress of "Image X of Y" in the TUI
        image_task = image_progress.add_task(
            f"[cyan]Images", total=len(files_to_process)
        )

        # === PHASE 3: Process all files in parallel ===
        # Use ThreadPoolExecutor to convert multiple images simultaneously
        # This significantly speeds up processing on multi-core systems
        processed_files = {}  # Will store results: {original_filename: (new_filename, data)}
        total_orig_size = 0  # Track total original size of images
        total_new_size = 0  # Track total size after WebP conversion
        converted_count = 0  # Count how many images were actually converted to WebP

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all files to the thread pool for parallel processing
                # Each file gets processed by a worker thread
                future_to_filename = {
                    executor.submit(
                        process_single_file,
                        filename,
                        file_data,
                        quality,
                        image_progress,
                        image_task,
                    ): filename
                    for filename, file_data in files_to_process
                }

                # Collect results as worker threads complete
                # as_completed returns futures as they finish (not in submission order)
                for future in as_completed(future_to_filename):
                    filename = future_to_filename[future]
                    try:
                        # Get result from worker thread
                        new_filename, converted_data, orig_size, new_size, was_converted = (
                            future.result()
                        )

                        # Store the processed file data
                        processed_files[filename] = (new_filename, converted_data)

                        # Accumulate statistics
                        total_orig_size += orig_size
                        total_new_size += new_size
                        if was_converted:
                            converted_count += 1  # Image was successfully converted to WebP

                    except Exception as e:
                        # Worker thread encountered an error processing this file
                        # Keep the original file data to avoid data loss
                        console.print(
                            f"[yellow]Warning: Error processing {filename}: {type(e).__name__}: {e}[/yellow]"
                        )
                        for orig_filename, orig_data in files_to_process:
                            if orig_filename == filename:
                                processed_files[filename] = (filename, orig_data)
                                break
        finally:
            # Always remove the image progress task, even if an error occurs
            image_progress.remove_task(image_task)

        # === PHASE 4: Create new CBZ with processed files ===
        # Write all processed files into a new ZIP archive
        new_cbz_data = BytesIO()
        with ZipFile(new_cbz_data, "w", ZIP_DEFLATED, compresslevel=9) as zip_write:
            # Write files in their original order (important for comic reading order)
            for original_filename, _ in files_to_process:
                if original_filename in processed_files:
                    new_filename, file_data = processed_files[original_filename]
                    # Use maximum ZIP compression (level 9) for smallest file size
                    zip_write.writestr(
                        new_filename,
                        file_data,
                        compress_type=ZIP_DEFLATED,
                        compresslevel=9,
                    )

        # === PHASE 5: Write new CBZ and handle original ===
        # Get the final compressed size
        new_cbz_data.seek(0)
        new_cbz_bytes = new_cbz_data.read()
        new_file_size = len(new_cbz_bytes)

        # Calculate space savings (negative = saved space, positive = increased size)
        size_diff = new_file_size - original_file_size

        # === PHASE 5: Write new CBZ and handle original ===
        # IMPORTANT: Write the file FIRST before marking as converted
        # This ensures memory and file system stay in sync if script crashes
        try:
            # Handle original file based on --delete flag
            if delete_original:
                # Delete mode: write to temp file first, then atomically replace original
                # This ensures we don't lose the original if the write fails
                temp_cbz_path = cbz_path.parent / f"{cbz_path.stem}_temp{cbz_path.suffix}"
                try:
                    with open(temp_cbz_path, "wb") as f:
                        f.write(new_cbz_bytes)
                        f.flush()
                        os.fsync(f.fileno())
                    # Atomic replace: delete original, rename temp to original
                    # Check if original exists before trying to delete (handles edge cases)
                    if cbz_path.exists():
                        cbz_path.unlink()
                    temp_cbz_path.rename(cbz_path)
                except Exception:
                    # Clean up temp file if any step failed
                    if temp_cbz_path.exists():
                        temp_cbz_path.unlink()
                    raise
            else:
                # Backup mode: rename original to *_OLD.cbz, then write new file
                # Python 3.6+ compatible: use parent / (stem + suffix) instead of with_stem()
                old_path = cbz_path.parent / f"{cbz_path.stem}_OLD{cbz_path.suffix}"
                # Handle existing _OLD file (from previous failed run)
                if old_path.exists():
                    console.print(
                        f"[yellow]Warning: Removing existing backup {old_path.name}[/yellow]"
                    )
                    old_path.unlink()
                # Write to temp file first
                temp_cbz_path = cbz_path.parent / f"{cbz_path.stem}_temp{cbz_path.suffix}"
                # Remove temp file if it exists (from previous failed run)
                if temp_cbz_path.exists():
                    temp_cbz_path.unlink()
                try:
                    with open(temp_cbz_path, "wb") as f:
                        f.write(new_cbz_bytes)
                        f.flush()
                        os.fsync(f.fileno())
                    # Rename original to _OLD, then temp to original
                    # Only rename if original exists
                    if cbz_path.exists():
                        os.rename(cbz_path, old_path)
                    temp_cbz_path.rename(cbz_path)
                except Exception:
                    # Clean up temp file if any step failed
                    if temp_cbz_path.exists():
                        temp_cbz_path.unlink()
                    raise
        except Exception as e:
            console.print(
                f"\n[bold red]⚠ ERROR: Failed to write new CBZ file![/bold red]"
            )
            console.print(f"[red]Reason: {e}[/red]")
            console.print(
                f"[yellow]Keeping original file unchanged. Memory NOT updated.[/yellow]\n"
            )
            raise  # Re-raise so the exception handler below catches it

        # === PHASE 6: Update statistics and memory ===
        # Update global statistics (thread-safe with lock)
        with progress_lock:
            current_state["stats"]["total_processed"] += 1
            if size_diff < 0:  # Only count actual space savings
                current_state["stats"]["total_saved"] += abs(size_diff)
            current_state["stats"]["images_converted"] += converted_count

        # Mark file as converted AFTER successful file write
        # This ensures memory only reflects successfully converted files
        mark_file_converted(memory, series_name, cbz_name)
        save_conversion_memory(memory_file, memory)

        # Update CBZ progress bar
        with progress_lock:
            cbz_progress.update(cbz_task, advance=1)

        return True  # Success!

    except Exception as e:
        # Fatal error processing this CBZ file
        console.print(
            f"[red]Error processing {cbz_name}: {type(e).__name__}: {e}[/red]"
        )
        # Update progress bar and continue to next file
        with progress_lock:
            cbz_progress.update(cbz_task, advance=1)
        return False  # Failure



