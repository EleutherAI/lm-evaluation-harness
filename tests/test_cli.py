import argparse

import pytest

import lm_eval.__main__


def test_cli_parse_error():
    """
    Assert error raised if cli args argument doesn't have type
    """
    with pytest.raises(ValueError):
        parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument(
            "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
        )
        parser.add_argument(
            "--tasks",
            "-t",
            default=None,
            metavar="task1,task2",
            help="To get full list of tasks, use the command lm-eval --tasks list",
        )
        lm_eval.__main__.check_argument_types(parser)


def test_cli_parse_no_error():
    """
    Assert typed arguments are parsed correctly
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="hf", help="Name of model e.g. `hf`"
    )
    parser.add_argument(
        "--tasks",
        "-t",
        type=str,
        default=None,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    lm_eval.__main__.check_argument_types(parser)
