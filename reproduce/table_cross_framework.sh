#!/usr/bin/env bash
# Paper tab:cross-framework (line 1288) — XComp rows.
# Both Base and SALA measured via ncu (paper methodology, line 1276-1278).
# Starred (*) baselines are compiler-reported (kernel OOB without SALA).
# Usage: GPU=0 bash reproduce/table_cross_framework.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CHOREO=""
if [[ -z "${CHOREO:-}" ]]; then
    for c in "$REPO/croqtile/build/choreo"; do
        [[ -x "$c" ]] && { CHOREO="$c"; break; }
    done
fi
[[ -x "${CHOREO:-}" ]] || {
    echo "ERROR: choreo not found. Set CHOREO=/path/to/choreo or build one:"
    echo "  cmake -S croqtile -B croqtile/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHOREO_DEFAULT_TARGET=cute && ninja -C croqtile/build choreo copp"
    exit 1
}
ARCH="${ARCH:-sm_90a}"
GPU="${GPU:-0}"

command -v ncu >/dev/null || { echo "ERROR: ncu not found"; exit 1; }
[[ -x "$CHOREO" ]] || { echo "ERROR: choreo not found at $CHOREO"; exit 1; }

TMPDIR="${TMPDIR:-/tmp/ae_cross_fw}"
mkdir -p "$TMPDIR"

NCU_SMEM="launch__shared_mem_per_block_dynamic"
NCU_METRICS="$NCU_SMEM,launch__occupancy_limit_shared_mem,launch__registers_per_thread"

echo "================================================================"
echo " Cross-framework SMEM — XComp rows (tab:cross-framework)"
echo " Both columns ncu-measured. * = OOB baseline (compiler-reported)."
echo " GPU: $GPU"
echo "================================================================"

# -------------------------------------------------------------------
# measure: compile --no-sala for Base, default for SALA; ncu both
# -------------------------------------------------------------------
measure() {
    local name="$1" co="$2" kernel_fn="${3:-__choreo_device_matmul}" size_args="${4:---m=4096 --n=4096 --k=4096}"
    local safename="${name// /_}"

    # ---- Base (--no-sala) ----
    $CHOREO -gs -t cute -arch=$ARCH --no-sala "$co" \
        -o "$TMPDIR/${safename}_base.result" 2>/dev/null
    bash "$TMPDIR/${safename}_base.result" --compile-link 2>/dev/null
    local base_exe=$(grep -oP '(?<=-o )\S+\.exe' "$TMPDIR/${safename}_base.result" | head -1)
    local base_raw=$(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_SMEM \
        -k "$kernel_fn" "$base_exe" $size_args 2>&1)
    local base_kb=$(echo "$base_raw" | grep "$NCU_SMEM" | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    local base_regs=$(echo "$base_raw" | grep "registers_per_thread" | head -1 | grep -oP '[0-9]+' | head -1)

    # ---- SALA (default) ----
    $CHOREO -gs -t cute -arch=$ARCH "$co" \
        -o "$TMPDIR/${safename}_sala.result" 2>/dev/null
    bash "$TMPDIR/${safename}_sala.result" --compile-link 2>/dev/null
    local sala_exe=$(grep -oP '(?<=-o )\S+\.exe' "$TMPDIR/${safename}_sala.result" | head -1)
    local sala_raw=$(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRICS \
        -k "$kernel_fn" "$sala_exe" $size_args 2>&1)
    local sala_kb=$(echo "$sala_raw" | grep "$NCU_SMEM" | head -1 | grep -oP '[0-9]+\.[0-9]+' | head -1)
    local sala_occ=$(echo "$sala_raw" | grep "occupancy_limit_shared" | head -1 | grep -oP '[0-9]+' | tail -1)

    local save_pct=""
    if [[ -n "$base_kb" && -n "$sala_kb" && "$base_kb" != "0.00" ]]; then
        save_pct=$(echo "scale=1; 100*($base_kb - $sala_kb)/$base_kb" | bc 2>/dev/null || echo "N/A")
    fi

    printf "%-22s  %7s KB  %7s KB  %5s%%  %3s regs  occ=%s\n" \
        "$name" "$base_kb" "$sala_kb" "$save_pct" "$base_regs" "$sala_occ"
}

# -------------------------------------------------------------------
# XComp GEMM
# -------------------------------------------------------------------
BENCH="$REPO/benchmarks/matmul"
FA_DIR="$REPO/benchmarks/flash_atten"

echo ""
echo "--- XComp GEMM ---"
measure "1P1C 64x128 4s"   "$BENCH/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co"
measure "1P1C 64x128 3s"   "$BENCH/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_3s.co"
measure "1P2C GEMM"        "$BENCH/matmul_f16_dyn_sm90_warpspec_1p2c.co"
measure "1P3C GEMM"        "$BENCH/matmul_f16_dyn_sm90_warpspec_1p3c.co"

# FA rows (different kernel function, no size args)
echo ""
echo "--- XComp Flash Attention ---"
measure "FA tuned 1P2C"    "$FA_DIR/fa_tuned_1p2c.co" \
    "__choreo_device_flash_atten" ""

# -------------------------------------------------------------------
# Paper expected
# -------------------------------------------------------------------
echo ""
echo "================================================================"
echo " Paper expected — tab:cross-framework (all values in KB)"
echo "================================================================"
echo ""
echo "  Framework   Kernel               Base     SALA     Save   Occ"
echo "  ---------   ------               ----     ----     ----   ---"
echo "  XComp       1P1C 64x128 4s        116       99      15%    1→2"
echo "  XComp       1P1C 128² 3s †         91       75      18%    1→2"
echo "  XComp       1P2C GEMM             132       99      25%    1→1"
echo "  XComp       1P3C GEMM             132       83      37%    1→1"
echo "  XComp       FA 1P1C 2s             80       72      10%    2→2"
echo "  XComp       FA tuned 1P2C         197      164      17%    1→1"
echo "  XComp       FA tuned 3s          256*      230      10%    —→1"
echo "  ---------   ------               ----     ----     ----   ---"
echo "  CUTLASS     Coop 128² 2s          100       67      34%    1→1"
echo "  CUTLASS     Coop 128² 3s          133       99      25%    1→1"
echo "  CUTLASS     Coop 128² 4s          166      132      20%    1→1"
echo "  CUTLASS     Coop 128×256 2s       133       99      25%    1→1"
echo "  ---------   ------               ----     ----     ----   ---"
echo "  Tawa        128² 3s               131       98      25%    1→1"
echo "  Tawa        128² 2s                98       66      33%    1→1"
echo "  Tawa        64×128 2s              66       49      25%    2→2"
echo "  Tawa        FA WS 2s              199      164      18%    1→1"
echo "  Tawa        FA WS 3s             258*      230      11%    —→1"
echo ""
echo "  *  Baseline is compiler-reported (kernel OOB without SALA)."
echo "  †  Paper labels this row '128² 3s' but the kernel is 64×128 3s, and the"
echo "     paper's numbers (116→99 KB) duplicate the 64×128 4s measurement (a"
echo "     labeling error). Shown above is the actual ncu measurement of the"
echo "     3-stage kernel (1p1c_64x128_3s.co): 91→75 KB (18%)."
echo ""
echo "  CUTLASS and Tawa rows require external dependencies."
echo "  See REPRODUCTION_LOG.md for details."
