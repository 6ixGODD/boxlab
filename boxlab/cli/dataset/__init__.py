from __future__ import annotations

import argparse
import typing as t

if t.TYPE_CHECKING:
    from argparse import _SubParsersAction


def register(subparser: _SubParsersAction[argparse.ArgumentParser]) -> None:
    from boxlab.cli.dataset import convert
    from boxlab.cli.dataset import info
    from boxlab.cli.dataset import merge
    from boxlab.cli.dataset import visualize

    parser = subparser.add_parser(
        "dataset",
        help="Dataset management operations",
        description="Manage, convert, merge, and analyze datasets",
    )

    dataset_subparser = parser.add_subparsers(
        dest="dataset_command",
        help="Dataset operation to perform",
        required=True,
    )

    # Register subcommands
    convert.register(dataset_subparser)
    info.register(dataset_subparser)
    merge.register(dataset_subparser)
    visualize.register(dataset_subparser)
