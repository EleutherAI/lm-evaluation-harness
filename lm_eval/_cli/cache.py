import argparse

from lm_eval._cli.base import SubCommand


class CacheCommand(SubCommand):
    """Command for cache management."""

    def __init__(self, subparsers: argparse._SubParsersAction, *args, **kwargs):
        # Create and configure the parser
        super().__init__(*args, **kwargs)
        parser = subparsers.add_parser(
            "cache",
            help="Manage evaluation cache",
            description="Manage evaluation cache files and directories.",
            epilog="""
Examples:
  lm-eval cache clear --cache_path ./cache.db     # Clear cache file
  lm-eval cache info --cache_path ./cache.db      # Show cache info
  lm-eval cache clear --cache_path ./cache_dir/   # Clear cache directory
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Add command-specific arguments
        self._add_args(parser)

        # Set the function to execute for this subcommand
        parser.set_defaults(func=self.execute)

    def _add_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "action",
            choices=["clear", "info"],
            help="Action to perform: clear or info",
        )
        parser.add_argument(
            "--cache_path",
            type=str,
            default=None,
            help="Path to cache directory or file",
        )

    def execute(self, args: argparse.Namespace) -> None:
        """Execute the cache command."""
        import os

        if args.action == "clear":
            if args.cache_path:
                if os.path.exists(args.cache_path):
                    if os.path.isdir(args.cache_path):
                        import shutil

                        shutil.rmtree(args.cache_path)
                    else:
                        os.remove(args.cache_path)
                    print(f"✅ Cache cleared: {args.cache_path}")
                else:
                    print(f"❌ Cache path not found: {args.cache_path}")
            else:
                print("❌ Please specify --cache_path")
        elif args.action == "info":
            if args.cache_path and os.path.exists(args.cache_path):
                import os

                size = os.path.getsize(args.cache_path)
                print(f"Cache: {args.cache_path}")
                print(f"Size: {size} bytes")
            else:
                print("❌ Cache path not found or not specified")
