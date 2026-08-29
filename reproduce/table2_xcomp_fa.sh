#!/usr/bin/env bash
# Table 2 (tab:cross-framework) XComp FA rows — ncu-measured SMEM + correctness
# (~10 min, needs GPU + ncu). The GEMM rows are table3_xcomp.sh; the CUTLASS
# and Tawa rows are the table_cross_framework_*.sh scripts.
# Usage: GPU=0 bash reproduce/table2_xcomp_fa.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
B="$REPO/benchmarks/flash_atten"
GPU="${GPU:-0}"
ARCH="${ARCH:-sm_90a}"
export CUTE_HOME="${CUTE_HOME:-$REPO/croqtile/extern/cutlass}"
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
M="launch__shared_mem_per_block_dynamic,sm__warps_active.avg.pct_of_peak_sustained_active"
OUT=/tmp/ae_table2_fa; mkdir -p "$OUT"
export CHOREO_TIMING_WARMUP=2 CHOREO_TIMING_REPEAT=3   # keep the 6-config runs fast

command -v ncu >/dev/null || { echo "ERROR: ncu not found"; exit 1; }

smem_of() {  # $1=label $2=kern $3=co $4=modeflag -> prints "pass|smem|act"
    local label=$1 kern=$2 co=$3 flag=$4
    local r=$OUT/${label}.result
    "$CHOREO" -gs -t cute -arch="$ARCH" $flag "$co" -o "$r" >/dev/null 2>&1 \
        || { echo "NO-COMPILE|"; return; }
    bash "$r" --compile-link >/dev/null 2>&1 || { echo "NO-BUILD|"; return; }
    local exe=$(grep -oP '(?<=-o )\S+\.exe' "$r" | head -1)
    # Short repeats for the correctness pass: the harness default (10 warmup +
    # 500 timed iterations) saturates the H800 350 W power budget and drops the
    # SM clock, which can deadlock under load (see the guide 6.4).
    local out=$(CUDA_VISIBLE_DEVICES=$GPU CHOREO_TIMING_WARMUP=2 CHOREO_TIMING_REPEAT=3
        timeout 900 stdbuf -o0 -e0 "$exe" 2>&1)
    local pass=$(echo "$out" | grep -o 'Test Passed\|Test FAILED' | head -1)
    # Act. is profiled at SEQ=16384 (6th of the harness's 6 configs ->
    # launch 6 with WARMUP=0 REPEAT=1 -> --launch-skip 5), the paper's
    # pinned config; matches Table 3 (tab:occupancy).
    local ncu_out=$(CUDA_VISIBLE_DEVICES=$GPU timeout 900 env CHOREO_TIMING_WARMUP=0 CHOREO_TIMING_REPEAT=1 \
        ncu --launch-skip 5 --launch-count 1 \
        --metrics $M -k "$kern" "$exe" 2>&1)
    local smem=$(echo "$ncu_out" | grep 'shared_mem_per_block_dynamic' | head -1 | awk '{print $NF}')
    local act=$(echo "$ncu_out" | grep .sm__warps_active.avg. | head -1 | awk '{print $NF}')
    echo "${pass:-NO-RUN}|${smem:-NA}|${act:-NA}"
}

printf "%-12s  %-10s  %-10s  %-6s  %s\n" "Row" "Base KB" "SALA KB" "Save%" "Paper"
printf "%-12s  %-10s  %-10s  %-6s  %s\n" "---" "-------" "-------" "-----" "-----"

FA="$B/fa_tuned_1p2c.co"    # committed fa_3s.co is the 3-stage variant (STAGES 3)

# FA 1P1C 2s (fa_1p1c_2s)
r1=$(smem_of fa_1p1c_nosala "__choreo_device_flash_atten_sala" "$B/fa_1p1c_2s.co" "--no-sala")
r2=$(smem_of fa_1p1c_sala   "__choreo_device_flash_atten_sala" "$B/fa_1p1c_2s.co" "")
b=$(echo "$r1" | cut -d'|' -f2); s=$(echo "$r2" | cut -d'|' -f2)
a1=$(echo "$r1" | cut -d'|' -f3); a2=$(echo "$r2" | cut -d'|' -f3)
printf "%-12s  %-10s  %-10s  %-6s  %s  [%s] Act %s->%s\n" "FA 1P1C 2s" "$b KB" "$s KB" \
    "$(echo "scale=1; 100*($b-$s)/$b" | bc)%" "82->74 (10%)" "${r1%%|*}" "${a1:-NA}" "${a2:-NA}"

# FA tuned 1P2C
r1=$(smem_of fa_tuned_nosala "__choreo_device_flash_atten" "$FA" "--no-sala")
r2=$(smem_of fa_tuned_sala   "__choreo_device_flash_atten" "$FA" "")
b=$(echo "$r1" | cut -d'|' -f2); s=$(echo "$r2" | cut -d'|' -f2)
a1=$(echo "$r1" | cut -d'|' -f3); a2=$(echo "$r2" | cut -d'|' -f3)
printf "%-12s  %-10s  %-10s  %-6s  %s  [%s] Act %s->%s\n" "FA tuned" "$b KB" "$s KB" \
    "$(echo "scale=1; 100*($b-$s)/$b" | bc)%" "197->164 (17%)" "${r1%%|*}" "${a1:-NA}" "${a2:-NA}"

# FA tuned 3s — baseline cannot launch (OOB), SALA only
r2=$(smem_of fa_3s_sala "__choreo_device_flash_atten" "$B/fa_3s.co" "")
s=$(echo "$r2" | cut -d'|' -f2); a2=$(echo "$r2" | cut -d'|' -f3)
if [[ -n "$s" && "$s" != "NA" ]]; then
    printf "%-12s  %-10s  %-10s  %-6s  %s  [%s] Act %s\n" "FA 3s" "256*" "$s KB" \
        "$(echo "scale=1; 100*(262144/1024-$s)/(262144/1024)" | bc)%" "256*->230 (10%)" "${r2%%|*}" "${a2:-NA}"
else
    printf "%-12s  %-10s  %-10s  %-6s  %s  [%s]\n" "FA 3s" "256*" "ERR" "-" "256*->230 (10%)" "${r2%%|*}"
fi

echo ""
echo "Paper expects: FA 1P1C 2s: 82->74 (10%) | FA tuned: 197->164 (17%) |"
echo "               FA 3s: 256*->230 (10%, starred baseline cannot launch)."
echo "Note: FA 1P1C 2s row updated at camera-ready to the measured ncu pair (82->74)."
