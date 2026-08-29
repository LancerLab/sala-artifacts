#!/usr/bin/env bash
# Master reproduction script for SALA artifact evaluation (make-style targets).
# Usage:
#   bash reproduce/run_all.sh              # everything (Fig.2 + Table 2; needs GPU+ncu, ~40 min)
#   bash reproduce/run_all.sh figure2      # Fig. 2 (hb-results) only (fast, no GPU)
#   bash reproduce/run_all.sh table2       # paper Table 2: cross-framework ncu SMEM (needs GPU+ncu, ~30 min)
#   bash reproduce/run_all.sh cross        # Table 2's CUTLASS+Tawa rows only (needs GPU+ncu)
#   bash reproduce/run_all.sh table2 cross # multiple targets
#   GPU=1 bash reproduce/run_all.sh figure2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GPU="${GPU:-0}"
TARGETS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu=*) GPU="${1#*=}"; shift ;;
        -*) echo "Unknown: $1"; exit 1 ;;
        *) TARGETS+=("$1"); shift ;;
    esac
done
export GPU

run() {
    local flag="$1" script="$2" desc="$3"
    local fire=0
    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        # bare run_all.sh = everything; "cross" is a subset of "table2",
        # so under "all" each script runs once (as its "table2" entry)
        [[ "$flag" != "cross" ]] && fire=1
    else
        for t in "${TARGETS[@]}"; do [[ "$t" == "$flag" ]] && fire=1; done
    fi
    if [[ $fire == 1 ]]; then
        echo ""
        echo "==== $desc ===="
        bash "$SCRIPT_DIR/$script"
    fi
}

echo "================================================================"
echo " SALA Artifact Evaluation — Reproduction"
echo " GPU: $GPU"
echo "================================================================"

run "figure2" "figure2_compiler_smem.sh" "Fig. hb-results: compiler-reported SMEM (~10s, no GPU)"
run "table2" "table3_xcomp.sh" "Table 2 (tab:cross-framework): XComp GEMM ncu rows (~5min, needs GPU+ncu)"
run "table2" "table2_xcomp_fa.sh" "Table 2 (tab:cross-framework): XComp FA ncu rows (~10min, needs GPU+ncu)"
run "table2" "table_cross_framework_cutlass.sh" \
    "Table 2 (tab:cross-framework): CUTLASS ncu rows (~5min, needs GPU+ncu)"
run "table2" "table_cross_framework_tawa.sh" \
    "Table 2 (tab:cross-framework): Tawa ncu rows (~10min, needs GPU+ncu, triton-aref)"
run "cross" "table_cross_framework_cutlass.sh" \
    "Table 2 (tab:cross-framework): CUTLASS ncu rows (~5min, needs GPU+ncu)"
run "cross" "table_cross_framework_tawa.sh" \
    "Table 2 (tab:cross-framework): Tawa ncu rows (~10min, needs GPU+ncu, triton-aref)"

echo ""
echo "================================================================"
echo " Done. See REPRODUCTION_LOG.md for details."
echo "================================================================"
