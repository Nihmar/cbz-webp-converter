# CBZ to WebP Converter – Terminal UI

A powerful, multi‑threaded tool to convert images inside CBZ (Comic Book ZIP) files to WebP format.  
It features a beautiful live terminal interface, smart caching, and parallel processing – all while ensuring you never lose a single byte of data.

![Demo](docs/demo.gif) *(screenshot placeholder – imagine a live progress display)*

---

## Features

- **Live TUI** – Real‑time progress at three levels: folders, CBZ files, and individual images.
- **Parallel conversion** – Uses a thread pool to convert multiple images at once.
- **Smart compression** – Only keeps the WebP version if it’s actually **smaller** than the original.
- **Memory system** – Remembers which files have already been converted, so repeated runs skip them.
- **Automatic backups** – A timestamped backup of the conversion memory is created before any changes.
- **Safe file handling** – Original files are either deleted or renamed to `_OLD.cbz` (your choice).
- **Skip series** – You can manually mark a series as “skip” in the memory file to ignore it permanently.
- **Force mode** – Re‑convert everything, ignoring the memory cache.
- **Verbose mode** – See exactly what the script is doing under the hood.

---

## Requirements

- **Python** 3.6 or later  
- **Pillow** (Python Imaging Library) – for image conversion  
- **Rich** – for the beautiful terminal UI

All Python dependencies are listed in `requirements.txt`.

The tool runs on **Windows, macOS, and Linux** – no special system libraries are required.

---

## Installation

1. **Clone or download** this repository.
2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   (It is recommended to use a virtual environment.)

That’s it! You’re ready to convert.

---

## Usage

```bash
python cbz_webp_converter_tui.py /path/to/comics [options]
```

### Arguments

| Argument          | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| `directory`       | **(required)** The root directory containing subfolders (each representing a comic series).     |
| `--delete`        | Delete original CBZ files after conversion (default: rename to `_OLD.cbz`).                      |
| `--quality`       | WebP quality (1–100, default: 90). Higher = better quality but larger files.                     |
| `--threads`       | Number of parallel worker threads (1–32, default: 8).                                            |
| `--force`         | Ignore conversion memory and process all files again.                                            |
| `--verbose`       | Print detailed information about memory lookups, skipped files, and folder checks.               |

### Examples

```bash
# Convert all series in ~/Comics, keep backups (_OLD.cbz)
python cbz_webp_converter_tui.py ~/Comics

# Convert, delete originals, use 75% quality and 16 threads
python cbz_webp_converter_tui.py ~/Comics --delete --quality 75 --threads 16

# Force re‑conversion of everything (ignore memory)
python cbz_webp_converter_tui.py ~/Comics --force

# See verbose output while processing
python cbz_webp_converter_tui.py ~/Comics --verbose
```

---

## How It Works

### 1. Folder Selection
The script looks for a file named `to_convert.txt` in the same directory as the script.  
If the file exists, it reads the list of folder names (one per line) – only those folders will be processed.  
If the file does not exist (or is empty), the script automatically scans **all** subdirectories of the given `directory` (excluding hidden folders and system trash).

### 2. Conversion Memory
The file `conversion_memory.json` (also in the script directory) stores the history of converted files:

```json
{
  "Naruto": {
    "files": ["Naruto_v01.cbz", "Naruto_v02.cbz"],
    "skip": false
  }
}
```

- `files`: list of already converted CBZ files.
- `skip`: if set to `true`, the whole series will be ignored on future runs (useful for series you don’t want to touch).

You can edit this file manually – just be careful with the JSON syntax.

### 3. Processing a CBZ File
- The CBZ (ZIP archive) is read entirely into memory.
- All image files (`.jpg`, `.png`, etc.) are converted to WebP in **parallel** threads.
- If the WebP version is **not smaller**, the original image is kept.
- All files (converted + unchanged) are written back into a **new** CBZ with maximum ZIP compression.
- The original file is either **deleted** or renamed to `_OLD.cbz`.
- Statistics are updated, and the conversion memory is saved immediately (atomic write with backup).

### 4. Smart Caching
When `--force` is **not** used, the script checks the memory for each CBZ file:
- If the file is already listed → skipped.
- If the CBZ contains **only** WebP images (and no other convertible formats) → it is automatically marked as converted and skipped.

This ensures you don’t waste time on files that are already optimised.

### 5. Backups
Every time the script starts, it creates a timestamped backup of the existing `conversion_memory.json` (e.g., `conversion_memory_20260226_143052.json`).  
This protects against accidental data loss.

---

## The `to_convert.txt` File

Place this file next to the script (`cbz_webp_converter_tui.py`) with one folder name per line.  
Only those folders will be processed – useful if you want to convert a subset of your collection.

**Example `to_convert.txt`:**
```
Naruto
One Piece
Berserk
```

If the file is missing or empty, the script processes **all** folders in the target directory.

---

## Diagnostic Script

A helper script `diagnose_folders.py` is included. It helps you verify that the folder names in `to_convert.txt` match the actual directory structure on disk. Run it with:

```bash
python diagnose_folders.py /path/to/comics
```

It will show you exactly what the script sees and highlight any mismatches (trailing spaces, hidden characters, etc.).

---

## License

Free for personal use. Created in 2026.

---

## Notes

- The script never modifies the original CBZ file until the new one is fully built in memory – your data is safe even if the conversion fails.
- Because everything is processed in RAM, you need enough free memory to hold the largest CBZ file you convert (plus working buffers). Most comic CBZ files are well under 500 MB, so this is rarely an issue.
- The `--threads` parameter can be increased on high‑core systems, but going above 16 often yields diminishing returns due to disk/CPU bottlenecks.

---

## Troubleshooting

**“No folders found”**  
- Make sure the target directory exists and contains subdirectories.
- Check `to_convert.txt` for typos or extra spaces.
- Run the diagnostic script to see what the script sees.

**“Permission denied” when saving memory**  
- Ensure the script has write permissions in its own directory (where `conversion_memory.json` lives).

**Conversion is slow**  
- Try increasing `--threads` (if you have a multi‑core CPU).
- The first run will be slower because all images are converted. Subsequent runs skip already processed files.

**Memory file grows very large**  
- This is normal – it tracks every converted CBZ. If you want to start fresh, you can delete `conversion_memory.json` (a backup is always created).

---

Enjoy smaller, space‑saving comic archives!
