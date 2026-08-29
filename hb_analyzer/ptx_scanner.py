"""PTX scanner for extracting synchronization and shared memory patterns.

Scans PTX assembly to identify:
  - Shared memory declarations (.shared)
  - mbarrier operations (init, arrive, try_wait)
  - TMA bulk copy operations (cp.async.bulk)
  - CTA barriers (bar.sync)
  - Shared memory access instructions (ld.shared, st.shared)
  - Warp group structure from predicated execution
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SharedDecl:
    """A .shared memory declaration in PTX."""
    name: str
    size_bytes: int
    alignment: int
    is_extern: bool = False


@dataclass
class MBarrierOp:
    """An mbarrier operation."""
    op_type: str  # "init", "arrive", "arrive_expect_tx", "try_wait"
    line_no: int
    raw: str
    address_reg: str = ""
    predicate: str = ""


@dataclass
class TMAOp:
    """A TMA bulk copy operation."""
    line_no: int
    raw: str
    dst_reg: str = ""
    mbar_reg: str = ""
    predicate: str = ""


@dataclass
class SmemAccess:
    """A shared memory load or store."""
    op_type: str  # "load" or "store"
    line_no: int
    raw: str
    address: str = ""


@dataclass
class BarrierOp:
    """A CTA-level barrier."""
    line_no: int
    raw: str


@dataclass
class PTXKernelInfo:
    """Extracted sync/memory info from one PTX kernel."""
    name: str = ""
    shared_decls: list[SharedDecl] = field(default_factory=list)
    mbarrier_ops: list[MBarrierOp] = field(default_factory=list)
    tma_ops: list[TMAOp] = field(default_factory=list)
    smem_accesses: list[SmemAccess] = field(default_factory=list)
    barriers: list[BarrierOp] = field(default_factory=list)
    total_shared_bytes: int = 0
    has_extern_shared: bool = False


# Regex patterns for PTX instructions
_RE_KERNEL = re.compile(r'\.entry\s+(\w+)')
_RE_SHARED = re.compile(
    r'\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[(\d+)\]'
)
_RE_EXTERN_SHARED = re.compile(
    r'\.extern\s+\.shared\s+\.align\s+(\d+)\s+\.b8\s+(\w+)\[\]'
)
_RE_MBARRIER_INIT = re.compile(
    r'mbarrier\.init\.shared\S*\s+\[([^\]]+)\]'
)
_RE_MBARRIER_ARRIVE = re.compile(
    r'mbarrier\.arrive(?:\.expect_tx)?\.shared\S*\s+'
)
_RE_MBARRIER_WAIT = re.compile(
    r'mbarrier\.try_wait\S*\.shared\S*\s+'
)
_RE_TMA_COPY = re.compile(
    r'cp\.async\.bulk(?:\.tensor)?\S*\.shared\S*\.global\S*'
)
_RE_BAR_SYNC = re.compile(r'bar\.sync\s+')
_RE_LD_SHARED = re.compile(r'ld\.shared\S*\s+')
_RE_ST_SHARED = re.compile(r'st\.shared\S*\s+')
_RE_PREDICATE = re.compile(r'@(!?\%p\d+)\s+')


def scan_ptx(ptx_text: str) -> list[PTXKernelInfo]:
    """Scan PTX text and extract kernel sync/memory info."""
    kernels: list[PTXKernelInfo] = []
    current: PTXKernelInfo | None = None

    for line_no, line in enumerate(ptx_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue

        m = _RE_KERNEL.search(stripped)
        if m:
            if current:
                _finalize(current)
                kernels.append(current)
            current = PTXKernelInfo(name=m.group(1))
            continue

        if current is None:
            continue

        pred = ""
        pm = _RE_PREDICATE.match(stripped)
        if pm:
            pred = pm.group(1)

        m = _RE_SHARED.search(stripped)
        if m:
            current.shared_decls.append(SharedDecl(
                name=m.group(2),
                alignment=int(m.group(1)),
                size_bytes=int(m.group(3)),
            ))
            continue

        m = _RE_EXTERN_SHARED.search(stripped)
        if m:
            current.shared_decls.append(SharedDecl(
                name=m.group(2),
                alignment=int(m.group(1)),
                size_bytes=0,
                is_extern=True,
            ))
            current.has_extern_shared = True
            continue

        m = _RE_MBARRIER_INIT.search(stripped)
        if m:
            current.mbarrier_ops.append(MBarrierOp(
                op_type="init", line_no=line_no, raw=stripped,
                address_reg=m.group(1), predicate=pred,
            ))
            continue

        if _RE_MBARRIER_ARRIVE.search(stripped):
            op_type = ("arrive_expect_tx"
                       if "expect_tx" in stripped else "arrive")
            current.mbarrier_ops.append(MBarrierOp(
                op_type=op_type, line_no=line_no, raw=stripped,
                predicate=pred,
            ))
            continue

        if _RE_MBARRIER_WAIT.search(stripped):
            current.mbarrier_ops.append(MBarrierOp(
                op_type="try_wait", line_no=line_no, raw=stripped,
                predicate=pred,
            ))
            continue

        if _RE_TMA_COPY.search(stripped):
            current.tma_ops.append(TMAOp(
                line_no=line_no, raw=stripped, predicate=pred,
            ))
            continue

        if _RE_BAR_SYNC.search(stripped):
            current.barriers.append(BarrierOp(
                line_no=line_no, raw=stripped,
            ))
            continue

        if _RE_LD_SHARED.search(stripped):
            current.smem_accesses.append(SmemAccess(
                op_type="load", line_no=line_no, raw=stripped,
            ))
            continue

        if _RE_ST_SHARED.search(stripped):
            current.smem_accesses.append(SmemAccess(
                op_type="store", line_no=line_no, raw=stripped,
            ))
            continue

    if current:
        _finalize(current)
        kernels.append(current)

    return kernels


def _finalize(info: PTXKernelInfo) -> None:
    info.total_shared_bytes = sum(
        d.size_bytes for d in info.shared_decls if not d.is_extern
    )


def scan_ptx_file(path: str | Path) -> list[PTXKernelInfo]:
    with open(path) as f:
        return scan_ptx(f.read())


def format_ptx_summary(info: PTXKernelInfo) -> str:
    lines = [
        f"Kernel: {info.name}",
        f"  Static shared memory: {info.total_shared_bytes} bytes "
        f"({len(info.shared_decls)} declarations)",
        f"  Extern shared: {'yes' if info.has_extern_shared else 'no'}",
        f"  mbarrier ops: {len(info.mbarrier_ops)} "
        f"(init={sum(1 for o in info.mbarrier_ops if o.op_type == 'init')}, "
        f"arrive={sum(1 for o in info.mbarrier_ops if 'arrive' in o.op_type)}, "
        f"wait={sum(1 for o in info.mbarrier_ops if o.op_type == 'try_wait')})",
        f"  TMA bulk copies: {len(info.tma_ops)}",
        f"  Shared loads/stores: {len(info.smem_accesses)} "
        f"(ld={sum(1 for a in info.smem_accesses if a.op_type == 'load')}, "
        f"st={sum(1 for a in info.smem_accesses if a.op_type == 'store')})",
        f"  CTA barriers: {len(info.barriers)}",
    ]

    wg_indicator = any(
        "arrive" in o.op_type and o.predicate for o in info.mbarrier_ops
    )
    lines.append(f"  Warp specialization detected: "
                 f"{'likely' if wg_indicator else 'unlikely'}")

    return "\n".join(lines)
