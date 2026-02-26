#!/usr/bin/env python3
"""
Diagnostic script to check folder name handling
"""

import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python diagnose_folders.py /path/to/directory")
    sys.exit(1)

script_dir = Path(__file__).parent
folder_list_file = script_dir / 'to_convert.txt'

if not folder_list_file.exists():
    print(f"ERROR: {folder_list_file} not found!")
    sys.exit(1)

# Read folder names
with open(folder_list_file, 'r', encoding='utf-8') as f:
    folder_names = [line.strip() for line in f if line.strip()]

print(f"Found {len(folder_names)} folders in to_convert.txt:\n")

base_dir = Path(sys.argv[1])

if not base_dir.exists():
    print(f"ERROR: Directory {sys.argv[1]} does not exist!")
    sys.exit(1)

for i, folder_name in enumerate(folder_names, 1):
    print(f"{i}. Folder name from file: '{folder_name}'")
    print(f"   Repr: {repr(folder_name)}")
    
    folder_path = base_dir / folder_name
    print(f"   Full path: {folder_path}")
    print(f"   Exists: {folder_path.exists()}")
    print(f"   Is directory: {folder_path.is_dir() if folder_path.exists() else 'N/A'}")
    
    if folder_path.exists() and folder_path.is_dir():
        cbz_files = list(folder_path.glob('*.cbz'))
        cbz_no_old = [f for f in cbz_files if not f.stem.endswith('_OLD')]
        print(f"   Total CBZ files: {len(cbz_files)}")
        print(f"   CBZ files (excluding _OLD): {len(cbz_no_old)}")
        if cbz_no_old:
            for cbz in cbz_no_old[:3]:  # Show first 3
                print(f"      - {cbz.name}")
            if len(cbz_no_old) > 3:
                print(f"      ... and {len(cbz_no_old) - 3} more")
    
    print()

# Also list what's actually in the directory
print("\n" + "="*60)
print("Actual folders in the directory:")
print("="*60)
for folder in sorted(base_dir.iterdir()):
    if folder.is_dir():
        print(f"  - {folder.name}")
        print(f"    Repr: {repr(folder.name)}")
