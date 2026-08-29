"""JSON-based kernel sync pattern description.

Allows manual or auto-generated description of a kernel's
warp-group structure, pipeline events, and buffer accesses.

Schema:
{
  "kernel": "kernel_name",
  "framework": "choreo|cutlass|tilelang|triton",
  "arch": "sm_90a",
  "warp_groups": [
    {
      "wg_id": 0,
      "role": "producer",
      "phases": [
        {
          "label": "P0",
          "signal_in": "empty",
          "signal_out": null,
          "buffers": ["lhs_s", "rhs_s"]
        },
        ...
      ]
    }
  ],
  "shared_buffers": {
    "lhs_s": {"size_bytes": 16384, "description": "LHS tile"},
    ...
  },
  "notes": "optional description"
}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hb_graph import Phase, HBGraph, build_hb_graph


def load_pattern(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def pattern_to_phases(pattern: dict[str, Any]) -> list[Phase]:
    """Convert a JSON pattern description to a list of Phase objects."""
    phases: list[Phase] = []
    phase_id = 0
    for wg in pattern["warp_groups"]:
        wg_id = wg["wg_id"]
        for p in wg["phases"]:
            phase = Phase(
                wg_id=wg_id,
                phase_id=phase_id,
                signal_in=p.get("signal_in"),
                signal_out=p.get("signal_out"),
                buffers_accessed=set(p.get("buffers", [])),
                label=p.get("label", f"WG{wg_id}_P{phase_id}"),
            )
            phases.append(phase)
            phase_id += 1
    return phases


def analyze_pattern(pattern: dict[str, Any]) -> dict[str, Any]:
    """Run full HB analysis on a kernel pattern.

    Returns a dict with:
      - graph: the HBGraph object
      - overlaps: list of (buf_a, buf_b, can_overlap)
      - smem_baseline: total size without overlaps
      - smem_optimized: size with maximum overlap (greedy estimate)
      - savings_bytes: baseline - optimized
      - savings_pct: percentage saved
    """
    phases = pattern_to_phases(pattern)
    graph = build_hb_graph(phases)

    overlaps = graph.overlap_report()
    shared_bufs = pattern.get("shared_buffers", {})

    baseline = sum(b["size_bytes"] for b in shared_bufs.values())

    # Greedy overlap estimation: merge overlappable buffers
    # into groups that can share the same physical memory.
    # Each group's allocation = max(sizes in group).
    buf_names = sorted(shared_bufs.keys())
    can_overlap_set = {(a, b) for a, b, ok in overlaps if ok}

    groups = _greedy_overlap_groups(buf_names, can_overlap_set)
    optimized = 0
    for group in groups:
        optimized += max(shared_bufs[b]["size_bytes"] for b in group)

    return {
        "kernel": pattern.get("kernel", "unknown"),
        "framework": pattern.get("framework", "unknown"),
        "graph": graph,
        "overlaps": overlaps,
        "smem_baseline": baseline,
        "smem_optimized": optimized,
        "savings_bytes": baseline - optimized,
        "savings_pct": (
            100.0 * (baseline - optimized) / baseline if baseline > 0 else 0.0
        ),
        "groups": groups,
    }


def _greedy_overlap_groups(
    bufs: list[str], can_overlap: set[tuple[str, str]]
) -> list[list[str]]:
    """Greedily merge buffers that can pairwise overlap into groups.

    A buffer can join a group only if it can overlap with ALL
    existing members of the group (clique requirement).
    """
    groups: list[list[str]] = []
    for buf in bufs:
        placed = False
        for group in groups:
            if all(
                (buf, m) in can_overlap or (m, buf) in can_overlap
                for m in group
            ):
                group.append(buf)
                placed = True
                break
        if not placed:
            groups.append([buf])
    return groups


def format_report(result: dict[str, Any]) -> str:
    """Format analysis results as human-readable text."""
    lines = [
        f"{'=' * 60}",
        f"Kernel:    {result['kernel']}",
        f"Framework: {result['framework']}",
        f"{'=' * 60}",
        "",
    ]

    graph: HBGraph = result["graph"]
    lines.append(graph.dump())
    lines.append("")

    lines.append("Buffer Overlap Analysis:")
    overlaps = result["overlaps"]
    safe = [(a, b) for a, b, ok in overlaps if ok]
    unsafe = [(a, b) for a, b, ok in overlaps if not ok]

    if safe:
        lines.append(f"  OVERLAPPABLE ({len(safe)} pairs):")
        for a, b in safe:
            lines.append(f"    ({a}, {b})")
    if unsafe:
        lines.append(f"  NON-OVERLAPPABLE ({len(unsafe)} pairs):")
        for a, b in unsafe:
            lines.append(f"    ({a}, {b})")

    lines.append("")
    lines.append(f"Overlap Groups (clique-based):")
    for i, group in enumerate(result["groups"]):
        lines.append(f"  Group {i}: {group}")

    lines.append("")
    lines.append(f"Shared Memory:")
    lines.append(f"  Baseline:  {result['smem_baseline']:>8} bytes")
    lines.append(f"  Optimized: {result['smem_optimized']:>8} bytes")
    lines.append(
        f"  Savings:   {result['savings_bytes']:>8} bytes "
        f"({result['savings_pct']:.1f}%)"
    )

    return "\n".join(lines)
