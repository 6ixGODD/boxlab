from __future__ import annotations

import argparse
import logging
import sys
import textwrap

from boxlab import __version__
from boxlab.cli.helper import display
from boxlab.exceptions import BoxlabError


class BoxlabHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom help formatter for BoxLab CLI.

    Provides better formatting with:
    - Proper indentation
    - Colored headers
    - Preserved formatting for examples
    - Better spacing
    """

    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 32,
        width: int | None = None,
    ):
        # Use terminal width if available
        if width is None:
            try:
                import shutil

                width = shutil.get_terminal_size().columns - 2
            except (AttributeError, ValueError):
                width = 80

        super().__init__(prog, indent_increment, max_help_position, width)

    def _format_action(self, action: argparse.Action) -> str:
        """Format a single action."""
        # Get the base formatting
        result = super()._format_action(action)

        # Add extra spacing between arguments
        if action.option_strings:
            result += "\n"

        return result

    def _format_usage(self, usage: str, actions: list, groups: list, prefix: str | None) -> str:
        """Format usage string with better styling."""
        if prefix is None:
            prefix = "Usage: "

        # Get the formatted usage
        return super()._format_usage(usage, actions, groups, prefix)

    def add_usage(self, usage: str, actions: list, groups: list, prefix: str | None = None) -> None:
        """Add usage with custom prefix."""
        if prefix is None:
            prefix = "Usage: "
        super().add_usage(usage, actions, groups, prefix)


def main() -> int:
    """Main entry point for BoxLab CLI.

    This is the primary entry point for the command-line interface. It handles:
    - Argument parsing
    - Command routing
    - Error handling and display
    - Exit code management

    Returns:
        Exit code:
            0 - Success
            1 - General error
            2+ - Specific error codes (see boxlab.exceptions)
            130 - Interrupted by user (Ctrl+C)

    Examples:
        Run from command line:

        ```bash
        boxlab --help
        boxlab dataset info data/coco --format coco
        boxlab annotator
        ```

        Run as module:

        ```bash
        python -m boxlab --help
        ```
    """
    try:
        args = parse_args()

        # Setup logging if verbose mode is enabled
        if hasattr(args, "verbose") and args.verbose:
            logging.basicConfig(
                level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            logging.debug("Verbose mode enabled")

        # Execute command
        args.func(args)
        return 0

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print()
        display.warning("Cancelled by user")
        return 130

    except BoxlabError as e:
        # Handle BoxLab-specific errors with proper error display
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        display.show_error(e, verbose=verbose)
        return e.code

    except Exception as e:
        # Handle unexpected errors
        verbose = "--verbose" in sys.argv or "-v" in sys.argv
        display.show_error(e, verbose=verbose)
        return 1


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register all available CLI commands.

    This function imports and registers all subcommands for the CLI.
    Each subcommand module provides a `register()` function that adds
    its parser to the subparsers.

    Args:
        subparsers: The subparsers action from the main argument parser.

    Note:
        Commands are imported lazily to avoid import overhead when
        only showing help or version information.
    """
    from boxlab.cli import annotator
    from boxlab.cli import dataset

    dataset.register(subparsers)
    annotator.register(subparsers)


def print_help(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Print help message with additional usage information.

    This is called when no command is specified or when an invalid
    command is provided. It displays the main help and suggests
    common commands.

    Args:
        parser: The main argument parser.
        args: Parsed arguments (may be incomplete).
    """
    parser.print_help()

    if args.command is None:
        print()
        display.info("Quick Start:")
        print("  boxlab dataset info <path> --format coco      # View dataset information")
        print("  boxlab dataset convert <input> <output>       # Convert between formats")
        print("  boxlab annotator                              # Launch GUI application")
        print()
        display.info("For detailed help on a command:")
        print("  boxlab <command> --help")
        print()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments with enhanced formatting.

    Creates the main argument parser with custom help formatting,
    registers all subcommands, and parses the command line arguments.

    Returns:
        Parsed arguments as a Namespace object.

    Raises:
        SystemExit: If parsing fails or help/version is requested.
    """
    # Create main parser with custom formatter
    parser = argparse.ArgumentParser(
        description=textwrap.dedent("""
            BoxLab - Object Detection Dataset Management Toolkit

            A comprehensive toolkit for managing, converting, and annotating
            object detection datasets with support for COCO and YOLO formats.
        """),
        prog="boxlab",
        formatter_class=BoxlabHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              # View dataset information
              boxlab dataset info data/coco/annotations.json --format coco

              # Convert COCO to YOLO
              boxlab dataset convert input.json -if coco output -of yolo

              # Merge multiple datasets
              boxlab dataset merge -i ds1.json coco -i ds2.json coco -o merged

              # Launch annotation GUI
              boxlab annotator

            Documentation: https://github.com/6ixGODD/boxlab
            Report issues: https://github.com/6ixGODD/boxlab/issues
        """),
        add_help=True,
    )

    # Global options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed tracebacks",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"BoxLab v{__version__}",
        help="Show version information and exit",
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="Available Commands",
        dest="command",
        help="Command to execute",
        metavar="<command>",
        required=False,  # Allow running without command to show help
    )

    # Register all subcommands
    register_commands(subparsers)

    # Set default function to print help
    parser.set_defaults(func=lambda args: print_help(parser, args))

    # Parse arguments
    return parser.parse_args()


def cli() -> None:
    """Entry point for installed command-line script.

    This function is called when running `boxlab` as an installed command.
    It's registered in pyproject.toml as a console script entry point.

    Example:
        After installing with pip:

        ```bash
        boxlab --help
        ```
    """
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
