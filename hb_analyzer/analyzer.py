"""Main analysis pipeline combining pattern analysis and PTX scanning."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .pattern import load_pattern, analyze_pattern, format_report
from .ptx_scanner import scan_ptx_file, format_ptx_summary


def analyze_pattern_file(path: str | Path) -> str:
    """Load a JSON pattern and run full HB analysis. Returns formatted report."""
    pattern = load_pattern(path)
    result = analyze_pattern(pattern)
    return format_report(result)


def analyze_ptx_file(path: str | Path) -> str:
    """Scan a PTX file and produce a sync/memory summary."""
    kernels = scan_ptx_file(path)
    if not kernels:
        return f"No kernels found in {path}"
    parts = []
    for k in kernels:
        parts.append(format_ptx_summary(k))
    return "\n\n".join(parts)


def cross_framework_comparison(
    pattern_files: list[str | Path],
) -> str:
    """Run HB analysis on multiple patterns and produce a comparison table."""
    rows: list[dict[str, Any]] = []
    for pf in pattern_files:
        pattern = load_pattern(pf)
        result = analyze_pattern(pattern)
        rows.append(result)

    lines = [
        "=" * 78,
        "Cross-Framework HB Analysis Comparison",
        "=" * 78,
        "",
        f"{'Kernel':<40} {'Framework':<10} {'Baseline':>8} {'Optimized':>9} "
        f"{'Saved':>8} {'Pct':>6}",
        "-" * 78,
    ]
    for r in rows:
        lines.append(
            f"{r['kernel']:<40} {r['framework']:<10} "
            f"{r['smem_baseline']:>8} {r['smem_optimized']:>9} "
            f"{r['savings_bytes']:>8} {r['savings_pct']:>5.1f}%"
        )

    lines.extend(["", "-" * 78, ""])

    for r in rows:
        overlaps = [(a, b) for a, b, ok in r["overlaps"] if ok]
        kernel = r["kernel"]
        if overlaps:
            pairs = ", ".join(f"({a},{b})" for a, b in overlaps)
            lines.append(f"{kernel}: overlappable = {pairs}")
        else:
            lines.append(f"{kernel}: no overlappable buffer pairs")

    lines.extend(["", "=" * 78])
    return "\n".join(lines)
