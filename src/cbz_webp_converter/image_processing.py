"""
Image processing and conversion for CBZ WebP Converter.

Handles image format detection and WebP conversion.
"""

# Standard library
from pathlib import Path
from io import BytesIO

# Image processing
from PIL import Image

# Rich console for warnings
from rich.console import Console

console = Console()

def convert_image_to_webp(image_data, filename, quality=90):
    """
    Convert image data to WebP format.
    Returns WebP data if smaller, otherwise returns original data.
    """
    try:
        # Store original file size for comparison
        # We only want to use WebP if it actually saves space
        original_size = len(image_data)

        # Open image from raw bytes using PIL
        # BytesIO creates a file-like object from the byte data
        img = Image.open(BytesIO(image_data))

        # Handle different image color modes for WebP compatibility
        # RGBA = RGB with alpha (transparency) channel - keep as-is for WebP
        # LA = Luminance (grayscale) with alpha - keep as-is for WebP
        # LA with alpha is also supported natively by WebP
        if img.mode in ("RGBA", "LA", "RGB"):
            # Keep these modes as-is - WebP supports them natively
            pass
        elif img.mode == "P":
            # Palette mode: convert to RGBA if image has transparency, otherwise RGB
            # WebP supports both, but RGBA preserves transparency information
            if "transparency" in img.info:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
        elif img.mode in ("L", "1"):
            # Grayscale or bitmap: keep as L (WebP supports grayscale)
            # WebP supports L mode directly, no need to convert to RGB
            pass
        elif img.mode == "CMYK":
            # CMYK is not supported by WebP, convert to RGB
            img = img.convert("RGB")
        else:
            # For any other unsupported modes, convert to RGB as fallback
            img = img.convert("RGB")

        # Convert to WebP format in memory (no disk I/O)
        webp_buffer = BytesIO()
        # Parameters:
        # - format='WEBP': Output format
        # - quality=90: Compression quality (1-100, higher=better quality/larger file)
        # - method=6: Compression method (0-6, higher=slower but better compression)
        # - exif=b'': Strip EXIF metadata to save space
        img.save(webp_buffer, format="WEBP", quality=quality, method=6, exif=b"")
        webp_data = webp_buffer.getvalue()

        # Calculate the new size after WebP conversion
        webp_size = len(webp_data)

        # Only use WebP if it's actually smaller
        # This is critical for files that don't compress well (already optimized JPEGs, etc.)
        if webp_size < original_size:
            # WebP is smaller - use it!
            # Change file extension from .jpg/.png/etc to .webp
            new_filename = str(Path(filename).with_suffix(".webp"))
            return webp_data, new_filename, original_size, webp_size, True
        else:
            # WebP is larger or same size - keep original format
            # This prevents file size increases
            return image_data, filename, original_size, webp_size, False

    except Exception as e:
        # Image conversion failed (corrupted image, unsupported format, etc.)
        # Keep original image data to avoid data loss
        console.print(
            f"[yellow]Warning: Could not convert {filename}: {type(e).__name__}: {e}[/yellow]"
        )
        return image_data, filename, len(image_data), len(image_data), False



def process_single_file(filename, file_data, quality, image_progress, image_task):
    """
    Process a single file from the CBZ archive.

    This function is called by worker threads in parallel to convert images.
    It determines whether a file needs conversion and handles the conversion process.

    Args:
        filename (str): Name of the file within the CBZ archive
        file_data (bytes): Raw binary data of the file
        quality (int): WebP quality setting (1-100)
        image_progress (Progress): Rich progress bar object for image processing
        image_task: Task ID in the progress bar to update

    Returns:
        tuple: (new_filename, converted_data, original_size, new_size, was_converted)
    """
    result = None

    # Check if this is an image file that needs conversion
    # Skip already-WebP images and non-image files
    if is_image_file(filename) and not is_webp_file(filename):
        # This is a convertible image (JPG, PNG, etc.)
        # Convert it to WebP and get conversion results
        converted_data, new_filename, orig_size, new_size, converted = (
            convert_image_to_webp(file_data, filename, quality)
        )
        result = (new_filename, converted_data, orig_size, new_size, converted)
    else:
        # File is either:
        # - Already a WebP image (no need to reconvert)
        # - A non-image file (metadata, XML, etc.)
        # Keep it as-is without modification
        result = (filename, file_data, len(file_data), len(file_data), False)

    # Update the progress bar to show this file has been processed
    # Thread-safe: uses a lock to prevent race conditions
    with progress_lock:
        image_progress.update(image_task, advance=1)

    return result



