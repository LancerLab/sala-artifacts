"""Core Happens-Before graph for pipelined GPU kernels.

Implements the formal model from hb_model.md:
  - Phase: maximal contiguous stmt sequence bounded by signals
  - HBGraph: partial order on phases with CanOverlap query
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Phase:
    """A maximal contiguous sequence of statements within one warp group,
    bounded by signal operations (wait/trigger)."""
    wg_id: int
    phase_id: int
    signal_in: Optional[str] = None
    signal_out: Optional[str] = None
    buffers_accessed: set[str] = field(default_factory=set)
    label: str = ""

    def __repr__(self) -> str:
        tag = self.label or f"WG{self.wg_id}_P{self.phase_id}"
        parts = [tag]
        if self.signal_in:
            parts.append(f"in={self.signal_in}")
        if self.signal_out:
            parts.append(f"out={self.signal_out}")
        if self.buffers_accessed:
            parts.append(f"bufs={{{','.join(sorted(self.buffers_accessed))}}}")
        return f"Phase({', '.join(parts)})"


class HBGraph:
    """Happens-Before partial order on phases.

    Edges encode two rules:
      1. Sequential: consecutive phases in the same WG
      2. Signal: trigger(E) in WG_x --> wait(E) in WG_y
    Transitive closure is computed via Floyd-Warshall.
    """

    def __init__(self, phases: list[Phase] | None = None):
        self.phases: list[Phase] = phases or []
        self._matrix: list[list[bool]] = []
        self._closed = False

    @property
    def n(self) -> int:
        return len(self.phases)

    def add_phase(self, phase: Phase) -> int:
        idx = len(self.phases)
        self.phases.append(phase)
        self._closed = False
        return idx

    def _ensure_matrix(self) -> None:
        n = self.n
        if len(self._matrix) != n:
            self._matrix = [[False] * n for _ in range(n)]
            self._closed = False

    def add_edge(self, from_idx: int, to_idx: int) -> None:
        self._ensure_matrix()
        self._matrix[from_idx][to_idx] = True
        self._closed = False

    def compute_transitive_closure(self) -> None:
        """Floyd-Warshall. O(n^3), n typically < 20."""
        self._ensure_matrix()
        n = self.n
        m = self._matrix
        for k in range(n):
            for i in range(n):
                if not m[i][k]:
                    continue
                for j in range(n):
                    if m[k][j]:
                        m[i][j] = True
        self._closed = True

    def reaches(self, from_idx: int, to_idx: int) -> bool:
        if not self._closed:
            self.compute_transitive_closure()
        return self._matrix[from_idx][to_idx]

    def can_overlap(self, buf_a: str, buf_b: str) -> bool:
        """True iff lifetimes of buf_a and buf_b are totally ordered by HB.

        CanOverlap(A, B) := forall p_a in lifetime(A), p_b in lifetime(B):
            p_a -->hb p_b  OR  p_b -->hb p_a
        """
        if not self._closed:
            self.compute_transitive_closure()
        phases_a = [i for i, p in enumerate(self.phases)
                    if buf_a in p.buffers_accessed]
        phases_b = [i for i, p in enumerate(self.phases)
                    if buf_b in p.buffers_accessed]
        if not phases_a or not phases_b:
            return False
        for ia in phases_a:
            for ib in phases_b:
                if ia == ib:
                    return False
                if not (self.reaches(ia, ib) or self.reaches(ib, ia)):
                    return False
        return True

    def buffer_lifetime(self, buf: str) -> list[Phase]:
        return [p for p in self.phases if buf in p.buffers_accessed]

    def all_buffers(self) -> set[str]:
        result: set[str] = set()
        for p in self.phases:
            result.update(p.buffers_accessed)
        return result

    def overlap_report(self) -> list[tuple[str, str, bool]]:
        """Check all buffer pairs, return (a, b, can_overlap)."""
        bufs = sorted(self.all_buffers())
        results = []
        for i, a in enumerate(bufs):
            for b in bufs[i + 1:]:
                results.append((a, b, self.can_overlap(a, b)))
        return results

    def dump(self, title: str = "HB Graph") -> str:
        lines = [f"=== {title} ===", f"Phases ({self.n}):"]
        for i, p in enumerate(self.phases):
            lines.append(f"  [{i}] {p}")
        if not self._closed:
            self.compute_transitive_closure()
        lines.append("HB edges (transitive closure):")
        for i in range(self.n):
            for j in range(self.n):
                if self._matrix[i][j]:
                    pi = self.phases[i].label or f"P{i}"
                    pj = self.phases[j].label or f"P{j}"
                    lines.append(f"  {pi} --> {pj}")
        return "\n".join(lines)


def build_sequential_edges(graph: HBGraph) -> None:
    """Add HB edges between consecutive phases in the same WG."""
    by_wg: dict[int, list[int]] = {}
    for idx, phase in enumerate(graph.phases):
        by_wg.setdefault(phase.wg_id, []).append(idx)
    for wg_phases in by_wg.values():
        for i in range(len(wg_phases) - 1):
            graph.add_edge(wg_phases[i], wg_phases[i + 1])


def build_signal_edges(graph: HBGraph) -> None:
    """Add HB edges from trigger(E) to wait(E) across different WGs.

    Acyclic constraint: only add edges that do not create cycles.
    Back-edges (e.g., consumer triggers empty -> producer waits empty)
    represent cross-iteration ordering and are excluded to prevent
    false within-iteration happens-before orderings.
    """
    triggers: dict[str, list[int]] = {}
    waits: dict[str, list[int]] = {}
    for idx, phase in enumerate(graph.phases):
        if phase.signal_out:
            triggers.setdefault(phase.signal_out, []).append(idx)
        if phase.signal_in:
            waits.setdefault(phase.signal_in, []).append(idx)

    # Compute initial reachability from sequential edges.
    graph.compute_transitive_closure()

    # Collect candidates sorted by trigger phase_id to prioritize
    # init/forward edges over back-edges.
    candidates: list[tuple[int, int, str]] = []
    for event, trigger_idxs in triggers.items():
        wait_idxs = waits.get(event, [])
        for t in trigger_idxs:
            for w in wait_idxs:
                if graph.phases[t].wg_id != graph.phases[w].wg_id:
                    candidates.append((t, w, event))
    candidates.sort(key=lambda e: graph.phases[e[0]].phase_id)

    for t, w, event in candidates:
        # HBA_NO_ACYCLIC=1 disables the acyclic constraint (Table 4 ablation:
        # "+HB w/o acyclic" — back-edges are added, which can create false
        # within-iteration orderings, e.g. FA K_s/V_s appear ordered).
        if os.environ.get("HBA_NO_ACYCLIC") == "1" or not graph.reaches(w, t):
            graph.add_edge(t, w)
            graph.compute_transitive_closure()
        # else: skip — would create cycle (cross-iteration back-edge)


def build_hb_graph(phases: list[Phase]) -> HBGraph:
    """Construct a complete HB graph from a list of phases."""
    graph = HBGraph(phases)
    graph._ensure_matrix()
    build_sequential_edges(graph)
    build_signal_edges(graph)
    return graph
