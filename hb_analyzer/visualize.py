"""Visualize HB graphs as DOT (Graphviz) or ASCII diagrams.

Produces publication-quality HB graph diagrams showing:
  - Phases as nodes (colored by warp group)
  - Sequential edges (solid, within WG)
  - Signal edges (dashed, cross-WG)
  - Buffer access annotations
  - Overlap result annotations
"""
from __future__ import annotations

from .hb_graph import HBGraph, Phase, build_sequential_edges, build_signal_edges

WG_COLORS = [
    "#4E79A7",  # WG0 blue
    "#F28E2B",  # WG1 orange
    "#E15759",  # WG2 red
    "#76B7B2",  # WG3 teal
    "#59A14F",  # WG4 green
    "#EDC948",  # WG5 yellow
    "#B07AA1",  # WG6 purple
    "#FF9DA7",  # WG7 pink
]

WG_FILL_COLORS = [
    "#D6E4F0",
    "#FDE5C9",
    "#F5C4C6",
    "#D4ECEA",
    "#C7E3C1",
    "#F9EFC4",
    "#DFD1E5",
    "#FFE4E8",
]


def hb_graph_to_dot(
    graph: HBGraph,
    title: str = "HB Graph",
    show_transitive: bool = False,
    show_buffers: bool = True,
    overlap_results: list[tuple[str, str, bool]] | None = None,
) -> str:
    """Convert an HBGraph to Graphviz DOT format.

    Args:
        graph: The HB graph to visualize.
        title: Title for the graph.
        show_transitive: If True, show all transitive edges. If False,
            show only direct (non-transitive) edges for clarity.
        show_buffers: Show buffer names in phase nodes.
        overlap_results: Optional overlap results to annotate.
    """
    lines = [
        f'digraph "{title}" {{',
        '  rankdir=TB;',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [fontname="Helvetica", fontsize=8];',
        f'  label="{title}";',
        '  labelloc=t;',
        '  fontsize=14;',
        '  fontname="Helvetica-Bold";',
        '',
    ]

    wg_ids = sorted(set(p.wg_id for p in graph.phases))

    for wg_id in wg_ids:
        wg_phases = [(i, p) for i, p in enumerate(graph.phases)
                     if p.wg_id == wg_id]
        if not wg_phases:
            continue

        color = WG_COLORS[wg_id % len(WG_COLORS)]
        fill = WG_FILL_COLORS[wg_id % len(WG_FILL_COLORS)]

        role = ""
        if wg_id == 0:
            role = "Producer"
        elif len(wg_ids) == 2:
            role = "Consumer"
        else:
            role = f"Consumer {wg_id - 1}"

        lines.append(f'  subgraph cluster_wg{wg_id} {{')
        lines.append(f'    label="WG{wg_id} ({role})";')
        lines.append(f'    style=dashed;')
        lines.append(f'    color="{color}";')
        lines.append(f'    fontcolor="{color}";')
        lines.append('')

        for idx, phase in wg_phases:
            label = phase.label or f"P{phase.phase_id}"
            parts = [label]
            if phase.signal_in:
                parts.append(f"wait {phase.signal_in}")
            if phase.signal_out:
                parts.append(f"trigger {phase.signal_out}")
            if show_buffers and phase.buffers_accessed:
                bufs = ", ".join(sorted(phase.buffers_accessed))
                parts.append(f"[{bufs}]")
            node_label = "\\n".join(parts)

            lines.append(
                f'    p{idx} [label="{node_label}", '
                f'fillcolor="{fill}", color="{color}"];'
            )

        lines.append('  }')
        lines.append('')

    # Edges
    seq_edges = _get_sequential_edges(graph)
    sig_edges = _get_signal_edges(graph)

    if show_transitive:
        if not graph._closed:
            graph.compute_transitive_closure()
        for i in range(graph.n):
            for j in range(graph.n):
                if graph._matrix[i][j] and i != j:
                    style = "solid" if (i, j) in seq_edges else "dashed"
                    lines.append(f'  p{i} -> p{j} [style={style}];')
    else:
        for i, j in seq_edges:
            lines.append(f'  p{i} -> p{j} [style=solid, color="#666666"];')
        for i, j in sig_edges:
            ev = graph.phases[i].signal_out or "?"
            lines.append(
                f'  p{i} -> p{j} [style=dashed, color="#CC0000", '
                f'label="{ev}", fontcolor="#CC0000"];'
            )

    if overlap_results:
        lines.append('')
        lines.append('  // Overlap annotations')
        safe = [(a, b) for a, b, ok in overlap_results if ok]
        if safe:
            overlap_text = "Overlappable: " + ", ".join(
                f"({a},{b})" for a, b in safe
            )
            lines.append(
                f'  overlap_note [shape=note, style=filled, '
                f'fillcolor="#E8F5E9", '
                f'label="{overlap_text}"];'
            )

    lines.append('}')
    return '\n'.join(lines)


def _get_sequential_edges(graph: HBGraph) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    by_wg: dict[int, list[int]] = {}
    for idx, phase in enumerate(graph.phases):
        by_wg.setdefault(phase.wg_id, []).append(idx)
    for wg_phases in by_wg.values():
        for i in range(len(wg_phases) - 1):
            edges.add((wg_phases[i], wg_phases[i + 1]))
    return edges


def _get_signal_edges(graph: HBGraph) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    triggers: dict[str, list[int]] = {}
    waits: dict[str, list[int]] = {}
    for idx, phase in enumerate(graph.phases):
        if phase.signal_out:
            triggers.setdefault(phase.signal_out, []).append(idx)
        if phase.signal_in:
            waits.setdefault(phase.signal_in, []).append(idx)
    for event, trigger_idxs in triggers.items():
        wait_idxs = waits.get(event, [])
        for t in trigger_idxs:
            for w in wait_idxs:
                if graph.phases[t].wg_id != graph.phases[w].wg_id:
                    edges.add((t, w))
    return edges


def pattern_to_dot(pattern: dict, show_buffers: bool = True) -> str:
    """Convert a JSON pattern to DOT via HB graph construction."""
    from .pattern import pattern_to_phases, analyze_pattern
    from .hb_graph import build_hb_graph

    phases = pattern_to_phases(pattern)
    graph = build_hb_graph(phases)
    result = analyze_pattern(pattern)

    title = f"{pattern.get('kernel', 'unknown')} ({pattern.get('framework', '')})"
    return hb_graph_to_dot(
        graph,
        title=title,
        show_buffers=show_buffers,
        overlap_results=result["overlaps"],
    )
