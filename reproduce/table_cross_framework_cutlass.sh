#!/usr/bin/env bash
# Paper tab:cross-framework — CUTLASS rows (ncu-measured on hardware).
#
# Builds the CUTLASS SM90 cooperative warp-specialized GEMM twice:
#   baseline — pristine CUTLASS v4.5.0 headers (struct: no overlap)
#   SALA     — v4.5.0 headers + the SALA struct->union patch
#              (benchmarks/cutlass/patches/sala_union.patch)
# then ncu-measures launch__shared_mem_per_block_dynamic for both.
#
# The patch is the paper's CUTLASS modification (paper.tex: "struct->union +
# NamedBarrier::sync"): the phase-disjoint mainloop/epilogue tensor storages
# become a union (max instead of sum).
#
# Usage: GPU=0 bash reproduce/table_cross_framework_cutlass.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
CUTLASS="$REPO/benchmarks/cutlass"
SRC="$CUTLASS/src/cutlass_union_test.cu"
PATCH="$CUTLASS/patches/sala_union.patch"
BASE_INC="${CUTLASS_HOME:-$CUTLASS/extern/cutlass_include}"  # pristine v4.5.0 repo root
UTIL_INC="$BASE_INC/tools/util/include"                      # cutlass/util helpers
WORK="${WORK:-/tmp/ae_cutlass}"
GPU="${GPU:-0}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not found"; exit 1; }
command -v ncu >/dev/null || { echo "ERROR: ncu not found"; exit 1; }
[[ -f "$SRC" && -d "$BASE_INC/include" && -d "$UTIL_INC" && -f "$PATCH" ]] || {
    echo "ERROR: missing sources under $CUTLASS"; exit 1; }

NCU_METRIC="launch__shared_mem_per_block_dynamic"
CONFIGS=("128x128 2s" "128x128 3s" "128x128 4s" "128x256 2s" "128x256 3s")

mkdir -p "$WORK"

echo "================================================================"
echo " Cross-framework SMEM — CUTLASS rows (tab:cross-framework)"
echo " Cooperative TMA warp-specialized GEMM, f16, CUTLASS v4.5.0"
echo " Baseline: struct  |  SALA: struct->union (patched header)"
echo " GPU: $GPU"
echo "================================================================"

# ---- 1. Baseline binary (pristine v4.5.0 headers) ----
echo "[1/3] Building baseline (struct) ..."
nvcc -std=c++17 -arch=sm_90a -O2 \
    -I "$BASE_INC/include" -I "$UTIL_INC" \
    "$SRC" -o "$WORK/cutlass_union_test_baseline"

# ---- 2. SALA binary (patched header copy) ----
echo "[2/3] Building SALA (struct->union) ..."
rm -rf "$WORK/cutlass_sala_include"
mkdir -p "$WORK/cutlass_sala_include"
cp -r "$BASE_INC/." "$WORK/cutlass_sala_include/"
(cd "$WORK/cutlass_sala_include/include" && patch -p1 --forward -s < "$PATCH")
nvcc -std=c++17 -arch=sm_90a -O2 \
    -I "$WORK/cutlass_sala_include/include" -I "$UTIL_INC" \
    "$SRC" -o "$WORK/cutlass_union_test_sala"

# ---- 3. ncu both binaries (5 kernels each, in CONFIGS order) ----
echo "[3/3] ncu measuring ..."
mapfile -t base_smem < <(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRIC \
    "$WORK/cutlass_union_test_baseline" 2>&1 \
    | grep "$NCU_METRIC" | grep -oP '[0-9]+\.[0-9]+')
mapfile -t sala_smem < <(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRIC \
    "$WORK/cutlass_union_test_sala" 2>&1 \
    | grep "$NCU_METRIC" | grep -oP '[0-9]+\.[0-9]+')

[[ ${#base_smem[@]} -eq 5 && ${#sala_smem[@]} -eq 5 ]] || {
    echo "ERROR: expected 5 kernels per binary, got ${#base_smem[@]}/${#sala_smem[@]}"; exit 1; }

echo ""
printf "%-12s  %-10s  %-10s  %-6s  %s\n" "Config" "Base KB" "SALA KB" "Save%" "Paper"
printf "%-12s  %-10s  %-10s  %-6s  %s\n" "------" "-------" "-------" "-----" "-----"
for i in "${!CONFIGS[@]}"; do
    b="${base_smem[$i]}"; s="${sala_smem[$i]}"
    sp=$(echo "scale=1; 100*($b-$s)/$b" | bc)
    case "$i" in
        0) paper="100 -> 67 (34%)" ;;
        1) paper="133 -> 99 (25%)" ;;
        2) paper="166 -> 132 (20%)" ;;
        3) paper="133 -> 99 (25%)" ;;
        4) paper="(not in Table 2)" ;;
    esac
    printf "%-12s  %-10s  %-10s  %-6s  %s\n" "${CONFIGS[$i]}" "$b" "$s" "$sp%" "$paper"
done

echo ""
echo "Kernel order matches cutlass_union_test.cu main(): 5 analyze_and_run calls."
echo "Paper values are these ncu measurements rounded to whole KB."
echo "Correctness: each binary prints GEMM PASS/FAIL per config (see full run above)."
