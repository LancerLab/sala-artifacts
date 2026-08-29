#!/usr/bin/env bash
# Figure 3 (fig:throughput) — end-to-end GEMM ratios at 4096^3 (the paper's
# methodology: 10 warmup + 500 timed iterations, cudaEvent). Compiles each of
# the five 1P1C/1P2C/1P3C configs in both modes (--no-sala and SALA), runs
# both binaries, and reports the measured TFLOPS and the ratio against the
# paper's ratio. ~30 min on an H800 (needs GPU).
# Usage: GPU=0 bash reproduce/figure3_throughput.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export CUTE_HOME="${CUTE_HOME:-$REPO/croqtile/extern/cutlass}"
ARCH="${ARCH:-sm_90a}"
GPU="${GPU:-0}"

# Prefer CHOREO from the env, then the branch's own build.
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

TMPDIR="${TMPDIR:-/tmp/ae_figure3}"
mkdir -p "$TMPDIR"
BENCH="$REPO/benchmarks/matmul"
WARMUP="${CHOREO_TIMING_WARMUP:-10}"
REPEAT="${CHOREO_TIMING_REPEAT:-500}"

printf "%-14s  %10s  %10s  %6s  %8s\n" "Config" "Base TF" "SALA TF" "Ratio" "Paper"
printf "%-14s  %10s  %10s  %6s  %8s\n" "------" "-------" "-------" "-----" "-----"

measure() {  # $1=label $2=co $3=paper_ratio
    local label=$1 co=$2 paper=$3
    # choreo's -o path must not contain spaces/parens: use a safe stem.
    local stem=$(echo "$label" | tr -cd '[:alnum:]')
    # 4096^3: the kernels default to 2048^3; sed the MATMUL_DEFAULT_* to 4096
    # (on a /tmp copy — the committed kernels stay untouched).
    sed -E 's/(#define MATMUL_DEFAULT_[MNK]) [0-9]+/\1 4096/' "$co" > "$TMPDIR/${stem}.co"
    "$CHOREO" -t cute -arch="$ARCH" --no-sala "$TMPDIR/${stem}.co" -o "$TMPDIR/${stem}_base" 2>/dev/null
    "$CHOREO" -t cute -arch="$ARCH"           "$TMPDIR/${stem}.co" -o "$TMPDIR/${stem}_sala" 2>/dev/null
    local b s
    b=$(CUDA_VISIBLE_DEVICES=$GPU CHOREO_TIMING_WARMUP=$WARMUP CHOREO_TIMING_REPEAT=$REPEAT \
        "$TMPDIR/${stem}_base" 2>/dev/null | grep -oP 'TFLOPS: \K[0-9.]+' | tail -1)
    s=$(CUDA_VISIBLE_DEVICES=$GPU CHOREO_TIMING_WARMUP=$WARMUP CHOREO_TIMING_REPEAT=$REPEAT \
        "$TMPDIR/${stem}_sala" 2>/dev/null | grep -oP 'TFLOPS: \K[0-9.]+' | tail -1)
    local ratio
    [[ -n "$b" && -n "$s" ]] && ratio=$(echo "scale=2; $s/$b" | bc)
    printf "%-14s  %8.1f  %8.1f  %6s  %8s\n" "$label" "${b:-0}" "${s:-0}" "${ratio:-NA}" "$paper"
}

measure "4s 64x128 (f16)" "$BENCH/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co" "1.34x"
measure "4s 64x128 (f8)"  "$BENCH/matmul_e4m3_dyn_sm90_warpspec_1p1c.co"           "1.40x"
measure "3s 64x128"       "$BENCH/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_3s.co"  "1.00x"
measure "1P2C (3s)"       "$BENCH/matmul_f16_dyn_sm90_warpspec_1p2c.co"            "1.00x"
measure "1P3C (2s)"       "$BENCH/matmul_f16_dyn_sm90_warpspec_1p3c.co"            "1.00x"

echo ""
echo "Paper values: 1P1C rows gain 1.34-1.40x (crossing the 1->2 CTA/SM boundary);"
echo "3s/1P2C/1P3C are flat (1.00x). Absolute TFLOPS differ with machine state."
