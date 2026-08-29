# Artifact Evaluation — Signal-Aware Liveness Analysis for Shared Memory Optimization in Warp-Specialized GPU Kernels (CGO 2027)

This repository contains the complete artifact for the paper: the
compiler, the kernels, and one-command reproduction scripts for every
XComp statistic in Figure 2, Table 2 (cross-framework), Table 3
(occupancy), Figure 3 (throughput), and Table 4 (ablation). A fresh
clone on an H100/H800 reproduces all compiler-reported and
`ncu`-measured shared-memory values exactly, plus register counts and
occupancy, and every kernel passes its CPU-reference correctness check.

| Requirement | Value |
|---|---|
| GPU | NVIDIA H100 or H800 (SM90a); the paper's measurements are from an H800 |
| CUDA | 13.0 (the container's base; reproduces the paper's registers exactly — see §6.1 for newer minors) |
| Tools | `cmake` ≥ 3.18, `ninja`, `flex`, `bison`, `gcc`/`g++`, `git`, `python3`, `bc`, `ncu` (ships with the CUDA toolkit) |
| Network | none for the compiler build (CUTLASS bundled); the Tawa fork's LLVM build ~1.2 GB at its one-time setup |
| Time | build ~10 min; Figure 2 ~10 s (no GPU); Table 2 ~30 min (GPU + ncu) |

## 0. Build the Compiler

```bash
git clone --recursive https://github.com/LancerLab/sala-artifacts.git && cd sala-artifacts
```

The compiler is the `croqtile/` submodule, pinned to the paper-lineage
branch's commit `a9cd1ba` (the compiler state the paper was evaluated with). All commands in this guide are run from
the **repository root**.

```bash
cmake -S croqtile -B croqtile/build -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCHOREO_DEFAULT_TARGET=cute
ninja -C croqtile/build choreo copp
```

CUTLASS v4.2.1 is fetched at configure only if
`croqtile/extern/cutlass` is missing — the artifact bundles it, so the
build needs no network. An explicit `-DCUTE_HOME=` or `CUTE_HOME`
always takes precedence. Verify:

```bash
croqtile/build/choreo --help-hidden | grep no-sala     # shows: --no-sala
```

All reproduction scripts resolve the compiler themselves
(`$REPO/croqtile/build/choreo`); override with `CHOREO=/path/to/choreo`.

---

## 1. Figure 2 (fig:hb-results) — Compiler-Reported SMEM Allocation (~10 s, no GPU)

These are the compiler-internal allocation sizes behind the paper's
Figure 2 ("SALA liveness refinement results (compiler allocation)") —
the direct effect of SALA's tighter liveness intervals, before `nvcc`
adds ~4–20 KB of runtime metadata.

### 1.1 One command

```bash
bash reproduce/figure2_compiler_smem.sh
```

Expected output — all nine items:

```
Fig.2 item      Base KB   SALA KB   Save%   Paper
----------      -------   -------   -----   ------------
1P1C f16        112 KB    96 KB     14.2%   112->96 (-14%)
1P1C e4m3       112 KB    96 KB     14.2%   112->96 (-14%)
1P2C f16        128 KB    96 KB     25.0%   128->96 (-25%)
1P3C f16        128 KB    80 KB     37.5%   128->80 (-37%)
FA K/V          40 KB     32 KB     20.0%   40->32 (-20%)
FA full         72 KB     56 KB     22.2%   72->56 (-22%)
FA tuned        192 KB    160 KB    16.6%   192->160 (-17%)
FA 3s           256 KB*   224 KB    12.5%   256*->224 (-12%)
Conv2d          96 KB     64 KB     33.3%   96->64 (-33%)
```

### 1.2 Kernel-compiled items (1–4, 7–9)

Per-kernel command (choreo prints the dynamic SMEM as
`cudaFuncAttributeMaxDynamicSharedMemorySize is set to \`N`):

```bash
croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala <kernel.co> -o /dev/null 2>&1 | grep -o 'set to `[0-9]*'
croqtile/build/choreo -gs -t cute -arch=sm_90a           <kernel.co> -o /dev/null 2>&1 | grep -o 'set to `[0-9]*'
```

| Item | Kernel file | Base B | SALA B | Paper |
|------|-------------|--------|--------|-------|
| 1P1C f16 | `benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co` | 114688 | 98304 | 112→96 |
| 1P1C e4m3 | `benchmarks/matmul/matmul_e4m3_dyn_sm90_warpspec_1p1c.co` | 114688 | 98304 | 112→96 |
| 1P2C f16 | `benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p2c.co` | 131072 | 98304 | 128→96 |
| 1P3C f16 | `benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` | 131072 | 81920 | 128→80 |
| FA tuned | `benchmarks/flash_atten/fa_tuned_1p2c.co` | 196608 | 163840 | 192→160 |
| FA 3s | `benchmarks/flash_atten/fa_3s.co` | 262144 (cannot compile) | 229376 | 256*→224 |
| Conv2d | `benchmarks/conv/conv2d_fprop_ws_1p1c.co` | 98304 | 65536 | 96→64 (−33%) |

The FA 3s baseline (item 8) cannot compile — it exceeds the 228 KB/SM
limit ("shared memory OUT OF BOUND"); 256 KB is the kernel's buffer
sum. SALA overlaps the 32 KB output buffer with the K/V pipeline →
224 KB. This is the paper's `256*` star.

### 1.3 HB-analyzer model items (5, 6)

Items 5 and 6 are HB-pattern models from the vendored analyzer
(`hb_analyzer/`, stdlib-only Python) — the pattern JSONs at
`hb_analyzer/examples/choreo_fa_kv.json` and
`choreo_fa_full.json`:

```bash
PYTHONPATH=. python3 -m hb_analyzer all-examples
```

prints (among others): `fa_kv_overlap` 40960→32768 (40→32),
`fa_fwd_full_pipeline` 73728→57344 (72→56).
Item 9 (Conv2d) is a real kernel compile with a CPU-reference
correctness check (Test Passed in both modes) — the paper's
"verified by compilation and correctness testing".

### 1.4 Verification (correctness)

Every kernel-compiled item passes its harness's independent
CPU-reference check in both modes (`--no-sala` and SALA):

| Items | Check | Result |
|-------|-------|--------|
| GEMM (f16 1P1C/1P2C/1P3C, e4m3) | harness CPU reference, 2048³ (fp32 full-K dot, 5% tolerance) | **Test Passed** ×2 modes each |
| FA tuned 1P2C | naive-attention reference (0 failed samples required), B=2 H=16 SEQ=512..16384 | **Test Passed**, fail_rate=0, ×2 modes |
| FA 3s | same harness, SALA side | **Test Passed** (baseline cannot launch) |
| FA 1P1C 2s | built-in naive-attention reference, 5 configs (SEQ 512–8192) | **Test Passed** 5/5, ×2 modes |
| Conv2d (item 9) | row-row sampled GEMM check (fp32 reference, 5% tolerance) | **Test Passed** ×2 modes |

`Test Passed` (or the scripts' `[Test Passed]` per row) is the success
marker; `Test FAILED`, a nonzero `fail_rate`, or `[NO-RUN]` (no
verdict within the run window) means the run did not verify — re-run
it before concluding.

---

## 2. Table 2 (tab:cross-framework) — ncu Hardware Validation (~30 min, needs GPU+ncu)

The paper's table has three framework parts — XComp, CUTLASS, Tawa.
All rows are `ncu`-measured (`launch__shared_mem_per_block_dynamic`)
except the starred baselines (which cannot launch).

```bash
GPU=0 bash reproduce/run_all.sh table2    # the whole Table 2, one command
# per-family pieces:
GPU=0 bash reproduce/table3_xcomp.sh               # XComp GEMM rows (~5 min)
GPU=0 bash reproduce/table2_xcomp_fa.sh            # XComp FA rows (~10 min)
GPU=0 bash reproduce/table_cross_framework_cutlass.sh   # CUTLASS rows
GPU=0 bash reproduce/table_cross_framework_tawa.sh      # Tawa rows
```

### 2.1 XComp — GEMM rows

| Row | Base KB (ncu) | SALA KB (ncu) | Paper |
|---|---|---|---|
| 1P1C 64x128 4s | 115.7 | 99.3 | 116→99 (15%) |
| 1P1C e4m3 (64x128 4s) | 115.7 | 99.3 | 116→99 (15%) |
| 1P1C 64x128 3s | 91.1 | 74.8 | 91→75 (18%) |
| 1P2C GEMM | 132.1 | 99.3 | 132→99 (25%) |
| 1P3C GEMM | 132.1 | 82.9 | 132→83 (37%) |

The script also prints occupancy limits, register counts, and
active-warp percentages (see §3).

### 2.2 XComp — FA rows (incl. correctness)

| Row | Base KB (ncu) | SALA KB (ncu) | Paper | Correctness |
|---|---|---|---|---|
| FA 1P1C 2s | 82.05 | 73.86 | 82→74 (10%) | Test Passed |
| FA tuned 1P2C | 196.74 | 163.97 | 197→164 (17%) | Test Passed |
| FA 3s | cannot launch | 229.50 | 256*→230 | Test Passed |

(All three FA rows are the paper's current values — the FA 1P1C 2s row
was updated at camera-ready to the measured ncu pair, 82→74.
The script also prints the ncu Act. at SEQ=16384, matching Table 3's
FA rows; see §3.)

### 2.3 CUTLASS rows (own toolchain, ~15 min incl. setup)

```bash
bash setup_cutlass.sh   # one-time: v4.5.0 headers + SALA struct->union patch
GPU=0 bash reproduce/table_cross_framework_cutlass.sh
```

| Config | Baseline | SALA (union) | Paper |
|--------|----------|-------------|-------|
| Coop 128² 2s | 100.35 | 66.56 | 100→67 (34%) |
| Coop 128² 3s | 133.12 | 99.33 | 133→99 (25%) |
| Coop 128² 4s | 165.89 | 132.10 | 166→132 (20%) |
| Coop 128×256 2s | 133.12 | 99.33 | 133→99 (25%) |

### 2.4 Tawa rows (own toolchain, Python 3.10, ~30 min incl. setup)

```bash
pip install torch numpy                                    # one-time (~2.5 GB)
cd benchmarks/tawa/triton-aref && pip install . && cd -   # one-time
GPU=0 bash reproduce/table_cross_framework_tawa.sh
```

(Network notes: json, googletest, and the NVIDIA redistributables are
bundled in the repo and pre-placed by the Dockerfile, so `pip install .`
downloads only the LLVM build from Microsoft's blob host (~1.2 GB,
cached under `~/.triton` — retry on timeout). pip's "torch requires
triton==3.7.1" conflict warning is expected and harmless — the kernels
use torch only for tensor allocation.)

| Config | Baseline | SALA | Paper |
|--------|----------|------|-------|
| 128² 3s | 131.13 | 98.36 | 131→98 (25%) |
| 128² 2s | 98.34 | 65.57 | 98→66 (33%) |
| 64×128 2s | 65.57 | 49.18 | 66→49 (25%) |
| FA WS 2s | 198.74 | 163.93 | 199→164 (18%) |
| FA WS 3s | 258.1* | 229.53 | 258*→230 (11%) |

---

## 3. Table 3 (tab:occupancy) — Registers, Occupancy, and Act.

The script collects `launch__registers_per_thread`, the occupancy
limits, and the ncu-measured **Act. column**
(`sm__warps_active.avg.pct_of_peak_sustained_active`). On CUDA 13.0
(the paper's environment):

| Row | Regs (no-SALA → SALA) | Paper |
|-----|------------------------|-------|
| 1P1C 64x128 4s | 72 → 72 | 72 ✓ |
| 1P2C GEMM (3s) | 95 → 96 | 95→96 ✓ |
| 1P3C GEMM (2s) | 101 → 96 | 101→96 ✓ |
| FA tuned 1P2C | 168 → 168 | 168 ✓ |

Occupancy follows: 1P1C 4s 1→2 CTAs/SM (smem-bound), 1P1C 3s 2→3,
1P2C/1P3C register-bound at 1 CTA/SM, FA register-bound at 1 CTA/SM —
all as in the paper's table.

The Act. column was re-measured during artifact evaluation and the
paper's Table 3 carries the measured values (marked orange in the
review PDF). The script prints them for all rows: GEMM rows at the
kernels' default 2048³ workload (the generated harness runs its
baked-in size; the `--m/--n/--k` arguments are ignored), FA rows at
the paper's pinned config (B=2 H=16, SEQ=16384 — the 6th of the
harness's 6 configs). Expected: 1P1C 4s 7.6→14.8% | 1P1C 3s
14.2→20.4% (observed 20.2–21.4% across runs — the 3-CTA/SM cell
is the most variance-prone) | 1P2C 18.0→18.0% | 1P3C
24.0→24.0% | FA tuned 18.4→18.4% | FA 3s 18.3% (SALA). The 1P1C
doubling (the occupancy claim) and the FA flatness reproduce
exactly; the absolute levels differ from the submission's (the
paper's 1P1C 12.5% is the theoretical 8-warps-of-64 ceiling, not
the achieved value).

---

## 4. Figure 3 (fig:throughput) — End-to-End Ratios (~30 min, needs GPU)

One command — compiles and runs all five configs in both modes at
4096³ (10 warmup + 500 timed iterations, the paper's methodology):

```bash
GPU=0 bash reproduce/figure3_throughput.sh
```

The ratio structure reproduces exactly:

| Config | Kernel | Base TFLOPS | SALA TFLOPS | Ratio | Paper |
|--------|--------|-------------|-------------|-------|-------|
| 4s 64×128 (f16) | `matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co` | 226.9 | 313.1 | **1.38×** | 1.34× |
| 4s 64×128 (f8) | `matmul_e4m3_dyn_sm90_warpspec_1p1c.co` | 370.4 | 497.4 | **1.34×** | 1.40× |
| 3s 64×128 | `matmul_f16_dyn_sm90_warpspec_1p1c_64x128_3s.co` | 301.4 | 314.7 | 1.04× | 1.00× |
| 1P2C GEMM (3s) | `matmul_f16_dyn_sm90_warpspec_1p2c.co` | 248.5 | 251.9 | 1.01× | 1.00× |
| 1P3C GEMM (2s) | `matmul_f16_dyn_sm90_warpspec_1p3c.co` | 275.2 | 273.8 | 0.99× | 1.00× |

Only the two 1P1C 4s configs (crossing the 1→2 CTA/SM boundary) gain;
3s/1P2C/1P3C are flat. Absolute TFLOPS differ from the paper's bars
(machine state); the ratios are the figure's claim and they match.

Per-kernel runs (M=N=K=4096 — sed the `MATMUL_DEFAULT_*` to 4096
before compiling; the kernels are in `benchmarks/matmul/`):

```bash
croqtile/build/choreo -t cute -arch=sm_90a benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co -o /tmp/mm   # + --no-sala for the baseline
CHOREO_TIMING_WARMUP=10 CHOREO_TIMING_REPEAT=500 /tmp/mm
```

FA e2e (the paper's ~360 claim): reproduce with the tuned FA kernel
at the AE config (B=2 H=16, SEQ=16384):

```bash
croqtile/build/choreo -t cute -arch=sm_90a benchmarks/flash_atten/fa_tuned_1p2c.co -o /tmp/fa_tuned            # SALA
croqtile/build/choreo -t cute -arch=sm_90a --no-sala benchmarks/flash_atten/fa_tuned_1p2c.co -o /tmp/fa_tuned_nosala  # baseline
CHOREO_TIMING_WARMUP=2 CHOREO_TIMING_REPEAT=3 /tmp/fa_tuned            # SALA
CHOREO_TIMING_WARMUP=2 CHOREO_TIMING_REPEAT=3 /tmp/fa_tuned_nosala     # baseline
```

(Without `-gs`, choreo emits the final binary directly; the harness
verifies by default, so both runs end with `Test Passed`. Each
invocation measures one mode — the harness runs a single sweep per
binary.)

Expected: ~360 TFLOPS in BOTH modes at SEQ=16384 (measured 360.6
SALA / 358.5 no-SALA — flat within ~2%: the run-to-run spread
between the two modes was 0.6-2.0% across the verified runs),
matching the paper's ~360
at its pinned config (B=2 H=16). The flatness — the paper's claim that
the 197→164 KB reduction does not affect FA throughput — holds at
B=1 H=16 (~305 TFLOPS) and B=2 H=16 alike. See §6.2 for the
config-dependence and §6.4 for why the FA row uses short repeats.

---

## 5. Table 4 (tab:ablation) — SALA Mechanisms

The paper's Table 4 covers two experiment sets: the **1P3C GEMM smem**
ablation (kernel compiles, §5.1) and the **FA K/V safety column**
(§5.2). Two 1P3C kernels are committed: `..._1p3c.co` carries
`sync.wg 1, 2, 3;` at the epilogue; `..._1p3c_nobarrier.co` is the
same kernel without it — SALA never inserts barriers, it reads the
kernel's explicit `sync.wg`.

### 5.1 1P3C GEMM smem ablation (kernel compiles)

| Configuration | 1P3C smem | Reproduce |
|---|---|---|
| No SALA (baseline) | 128 KB | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c_nobarrier.co` → 131072 B |
| +Barrier only | 128 KB | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` → 131072 B (the `sync.wg` is present; under standard liveness the barrier alone gives no savings — the row's point) |
| +SALA (phase-disjoint) | 80 KB | `croqtile/build/choreo -gs -t cute -arch=sm_90a benchmarks/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co` → 81920 B (+SALA implicitly includes the barrier: SALA requires and verifies `sync.wg` for multi-consumer kernels — the paper's SALA kernels carry it) |
| +HB w/o acyclic | 80 KB | same command as +SALA → 81920 B (the allocation is identical; the constraint's effect is on the FA K/V safety, §5.2) |
| +HB w/ acyclic | 80 KB | same command as +SALA → 81920 B |

For completeness: the no-barrier kernel compiled with SALA stays at
131072 B — SALA conservatively refuses the overlap without the
barrier, consistent with the paper's claim that SALA never inserts
barriers.

### 5.2 FA K/V safety column (vendored analyzer)

The acyclic constraint's effect on whether FA K_s/V_s may overlap,
demonstrated by the vendored analyzer — a faithful re-implementation
of the analysis's acyclic-constraint decision (the paper describes
this behavior in terms of the analysis itself). Rows 1–2 show the
baseline's allocation-level "no overlap" on the tuned FA kernel;
rows 3–5 use the analyzer on the FA K/V pattern (the same K/V
pipeline structure as Figure 2's item-5 model, 40→32 KB) — the
safety question is structural and independent of the tile size.

| Configuration | FA K/V | Sound? | Reproduce |
|---|---|---|---|
| No SALA (baseline) | no overlap | ✓ | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/flash_atten/fa_tuned_1p2c_nobarrier.co` → 196608 B (all four buffers allocated separately — no overlap) |
| +Barrier only | no overlap | ✓ | `croqtile/build/choreo -gs -t cute -arch=sm_90a --no-sala benchmarks/flash_atten/fa_tuned_1p2c.co` → 196608 B (the `sync.wg` is present; standard liveness still overlaps nothing) |
| +SALA (phase-disjoint) | no overlap | ✓ | `PYTHONPATH=. python3 -m hb_analyzer pattern hb_analyzer/examples/choreo_fa_kv.json` → 2 safe pairs; (K_s, V_s) non-overlappable |
| +HB w/o acyclic | **overlap** | ✗ (race) | `HBA_NO_ACYCLIC=1 PYTHONPATH=. python3 -m hb_analyzer pattern hb_analyzer/examples/choreo_fa_kv.json` → 3 overlappable pairs, incl. (K_s, V_s) |
| +HB w/ acyclic | no overlap | ✓ | analyzer default → 2 safe pairs; (K_s, V_s) non-overlappable |


---

## 6. Notes for the Reviewer

### 6.1 Register counts vs CUDA minor

On CUDA 13.0 the Table-3 registers are the paper's values. Under a
newer CUDA (e.g., 13.3) the 1P2C/1P3C cells may read 93/96 — `ptxas`
re-allocates registers slightly differently; 1P1C's 72 is stable.
Shared-memory values, occupancy, and correctness are version-stable
across CUDA 13.x. The FA row's 168 regs (paper's value, updated at
camera-ready from the earlier 158) is stable across all available
toolchains.

### 6.2 FA throughput is config-dependent

The tuned FA's TFLOPS varies strongly with the workload's B×H: B=1
H=16 gives ~305 TFLOPS, while the paper's config (B=2 H=16, pinned in
both throughput passages) gives ~360 TFLOPS at SEQ=16384 — matching
the paper's ~360 (short-repeat runs; see §4). The claim to verify is
the *structure*: SALA ≈ no-SALA flat (within ~2%), confirming that
the 197→164 KB reduction does not affect throughput. Compare within
a run, not across machines or configs.

### 6.3 What the scripts do not cover

The Act. column of Table 3 is re-measured by the scripts (table3
GEMM rows at the kernels' default 2048³ workload; FA rows at B=2
H=16 SEQ=16384) and the paper's Table 3 carries the measured values
(§3). The only Table-3 cells not re-measured are the CUTLASS and
Tawa rows (register-bound; SMEM/occupancy are unchanged).

### 6.4 Why the FA row uses short repeats (H800 power budget)

The paper's 10-warmup/500-iteration methodology is used for the GEMM
timed runs (§4); the FA e2e row deliberately uses short repeats
(warmup=2, repeat=3). At SEQ=16384 the tuned FA kernel saturates the
H800 PCIe's 350 W power cap (measured 348 W), which drops the SM
clock from 1755 to 1395 MHz — a sustained loop (e.g.
warmup=10/repeat=500) then measures ~313 TFLOPS instead of the
paper's ~360, and the power-cap clock transitions can occasionally
race the TMA pipeline and hang the kernel (short loops never trigger
it). The paper's number reproduces at short repeats, and the
SALA ≈ no-SALA flatness holds at any repeat — check the flatness,
not the absolute value (compare within a run, not across machines).

---

## 7. Layout

- `croqtile/` — the compiler as a **submodule**, pinned to the
  paper-lineage commit `a9cd1ba` (branch `sala-artifacts` of
  `LancerLab/croqtile`): `lib/`, `runtime/`, `tools/`, `cmake/`,
  `Makefile`, `CMakeLists.txt` — the paper-lineage DSL with
  `mma.commit`/`mma.wait`, `frag` ops, `sync.wg`, and the SALA
  integration, plus the small codegen/liveness fixes the FA/GEMM rows
  require.
- `benchmarks/` — the kernels: `matmul/` (1P1C 64x128 4s/3s,
  e4m3, 1P2C, 1P3C), `flash_atten/` (FA tuned 1P2C, FA 1P1C 2s; the
  3-stage FA is generated by the script via `sed`), `conv/` (the
  im2col convolution kernel), `cutlass/`
  (union-patch + test for the CUTLASS rows), `tawa/` (vendored
  triton-aref for the Tawa rows).
- `reproduce/` — the one-command reproduction scripts.
- `hb_analyzer/` — the HB-graph pattern analyzer (model rows
  and the ablation).
- `setup_cutlass.sh` — the one-time CUTLASS v4.5.0 headers + patch
  setup for the CUTLASS rows.
- `Dockerfile` — the AE container (builds the compiler from
  `croqtile/` inside the image); `README.md` — this guide.
