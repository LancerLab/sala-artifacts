#!/usr/bin/env python3
"""CLI entry point for HB analysis tool.

Usage:
  # Analyze a single kernel pattern
  python -m hb_analyzer pattern examples/choreo_1p1c_gemm.json

  # Scan PTX for sync structure
  python -m hb_analyzer ptx kernel.ptx

  # Cross-framework comparison
  python -m hb_analyzer compare examples/*.json

  # Run all examples
  python -m hb_analyzer all-examples
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analyzer import (
    analyze_pattern_file,
    analyze_ptx_file,
    cross_framework_comparison,
)
from .visualize import pattern_to_dot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hb_analyzer",
        description="Happens-Before analysis for pipelined GPU kernels",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_pattern = sub.add_parser("pattern", help="Analyze a kernel sync pattern (JSON)")
    p_pattern.add_argument("file", type=Path, help="JSON pattern file")

    p_ptx = sub.add_parser("ptx", help="Scan PTX for sync/memory structure")
    p_ptx.add_argument("file", type=Path, help="PTX file")

    p_compare = sub.add_parser("compare", help="Cross-framework comparison")
    p_compare.add_argument("files", type=Path, nargs="+", help="JSON pattern files")

    p_dot = sub.add_parser("dot", help="Generate Graphviz DOT for a pattern")
    p_dot.add_argument("file", type=Path, help="JSON pattern file")
    p_dot.add_argument("-o", "--output", type=Path, help="Output file (default: stdout)")

    sub.add_parser("all-examples", help="Run all built-in examples")

    args = parser.parse_args(argv)

    if args.command == "pattern":
        print(analyze_pattern_file(args.file))
    elif args.command == "ptx":
        print(analyze_ptx_file(args.file))
    elif args.command == "compare":
        print(cross_framework_comparison(args.files))
    elif args.command == "dot":
        from .pattern import load_pattern
        pattern = load_pattern(args.file)
        dot = pattern_to_dot(pattern)
        if args.output:
            args.output.write_text(dot)
            print(f"DOT written to {args.output}")
        else:
            print(dot)
    elif args.command == "all-examples":
        examples_dir = Path(__file__).parent / "examples"
        patterns = sorted(examples_dir.glob("*.json"))
        if not patterns:
            print(f"No examples found in {examples_dir}")
            return 1
        print(cross_framework_comparison(patterns))
        print()
        for p in patterns:
            print(analyze_pattern_file(p))
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
