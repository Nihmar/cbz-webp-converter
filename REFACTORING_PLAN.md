# CBZ WebP Converter Refactoring Plan

## Goal
Break the monolithic `cbz_webp_converter_tui.py` file into a structured package with multiple modules for better maintainability and extensibility.

## Package Structure

```
src/
└── cbz_webp_converter/
    ├── __init__.py                 # Package initialization
    ├── memory.py                   # Memory/file tracking operations
    ├── ui.py                       # Terminal UI components
    ├── image_processing.py         # Image conversion logic
    ├── file_operations.py          # CBZ file handling
    ├── converter.py                # Main conversion logic
    ├── config.py                   # Configuration and settings
    └── cli.py                      # Command-line interface
```

## Module Breakdown

### 1. memory.py
**Purpose**: Handle conversion memory tracking and persistence
**Functions**:
- `backup_memory_file()` - Create timestamped backups
- `load_conversion_memory()` - Load memory from JSON with restoration
- `save_conversion_memory()` - Atomic save with locking
- `is_file_converted()` - Check if file is in memory
- `mark_file_converted()` - Mark file as converted (thread-safe)
- `is_series_skipped()` - Check if series should be skipped

**Dependencies**: `pathlib`, `json`, `datetime`, `shutil`, `threading`

### 2. ui.py
**Purpose**: Terminal UI components and layout
**Functions**:
- `create_header()` - Configuration header panel
- `create_current_status()` - Current processing status panel
- `create_stats_panel()` - Statistics display panel
- `create_layout()` - Main TUI layout
- `Console` instance and related UI helpers

**Dependencies**: `rich` library

### 3. image_processing.py
**Purpose**: Image conversion and format handling
**Functions**:
- `is_image_file()` - Check if file is an image
- `is_webp_file()` - Check if file is WebP
- `convert_image_to_webp()` - Convert image to WebP format
- `process_single_file()` - Process individual file from CBZ

**Dependencies**: `PIL`, `pathlib`, `io`

### 4. file_operations.py
**Purpose**: CBZ file handling and operations
**Functions**:
- `process_cbz_file()` - Main CBZ processing logic
- File path validation and manipulation helpers
- Atomic file write operations

**Dependencies**: `pathlib`, `zipfile`, `os`, `shutil`

### 5. converter.py
**Purpose**: High-level conversion orchestration
**Functions**:
- `convert_folder()` - Process a single folder
- `process_all_folders()` - Process multiple folders
- Progress tracking integration

**Dependencies**: All other modules

### 6. config.py
**Purpose**: Configuration and settings
**Classes/Functions**:
- `Config` dataclass or class for settings
- Command-line argument parsing
- Default values and validation

**Dependencies**: `argparse`, `dataclasses`

### 7. cli.py
**Purpose**: Command-line interface entry point
**Functions**:
- `main()` - Entry point
- Argument parsing and validation
- Package initialization

**Dependencies**: `sys`, `argparse`, all other modules

## Migration Strategy

1. **Create new package structure** (done)
2. **Extract memory.py** - Move all memory-related functions
3. **Extract ui.py** - Move all UI-related functions
4. **Extract image_processing.py** - Move image conversion functions
5. **Extract file_operations.py** - Move CBZ file handling
6. **Extract converter.py** - Move high-level conversion logic
7. **Extract config.py** - Move configuration
8. **Extract cli.py** - Move main entry point
9. **Update imports** - Fix all cross-module imports
10. **Test and verify** - Ensure functionality is preserved

## Benefits

1. **Separation of Concerns**: Each module has a clear, single responsibility
2. **Testability**: Individual modules can be tested in isolation
3. **Extensibility**: Easy to add new features (e.g., different output formats)
4. **Maintainability**: Smaller files are easier to understand and modify
5. **Reusability**: Modules can be imported and used independently

## Considerations

1. **Circular Imports**: Need to carefully manage dependencies between modules
2. **Global State**: `current_state` and locks need to be handled carefully
3. **Backward Compatibility**: Original `cbz_webp_converter_tui.py` should still work
4. **Error Handling**: Ensure error propagation works across modules
