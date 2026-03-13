"""
Command-line interface for CBZ WebP Converter.
"""

import sys
from pathlib import Path

# Local imports
from .config import parse_args, Config
from .converter import main


def run():
    """Entry point for the CLI."""
    try:
        # Parse command-line arguments and create config
        config = parse_args()
        
        # Run the main conversion process
        main(config)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("[red]Conversion cancelled by user.[/red]")
        sys.exit(130)
    except Exception as e:
        print(f"[bold red]Unexpected error: {e}[/bold red]", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run()
