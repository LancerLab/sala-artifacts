#!/usr/bin/env bash
# Table 2 (tab:cross-framework) XComp GEMM rows: ncu-measured shared memory
# on hardware (~5 min, needs GPU+ncu). Also prints occupancy-limit and
# register counts plus the ncu-measured Act. column (sm__warps_active) for
# Table 3 (tab:occupancy): GEMM rows at the kernels' default 2048^3
# workload, FA rows at the paper's pinned B=2 H=16 SEQ=16384 config.
# Usage: GPU=0 bash reproduce/table3_xcomp.sh
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

command -v ncu >/dev/null || { echo "ERROR: ncu not found"; exit 1; }

TMPDIR="${TMPDIR:-/tmp/ae_table3}"
mkdir -p "$TMPDIR"

NCU_METRICS="launch__shared_mem_per_block_dynamic,launch__shared_mem_per_block_static,launch__occupancy_limit_shared_mem,launch__occupancy_limit_registers,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active"

printf "%-10s  %-10s  %-10s  %-6s  %-6s  %-7s  %s\n" "Kernel" "Base KB" "SALA KB" "Save%" "Occ" "Regs" "Act. no/+S"
printf "%-10s  %-10s  %-10s  %-6s  %-6s  %-7s  %s\n" "----------" "-------" "-------" "-----" "---" "------" "--------"

measure() {
    local name="$1" co="$2"
    "$CHOREO" -gs -t cute -arch="$ARCH" --no-sala "$co" -o "$TMPDIR/${name}_nosala.result" 2>/dev/null
    "$CHOREO" -gs -t cute -arch="$ARCH" "$co" -o "$TMPDIR/${name}_sala.result" 2>/dev/null
    bash "$TMPDIR/${name}_nosala.result" --compile-link 2>/dev/null
    bash "$TMPDIR/${name}_sala.result" --compile-link 2>/dev/null
    local nexe=$(grep -oP '(?<=-o )\S+\.exe' "$TMPDIR/${name}_nosala.result" | head -1)
    local sexe=$(grep -oP '(?<=-o )\S+\.exe' "$TMPDIR/${name}_sala.result" | head -1)
    local n_ncu=$(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRICS -k __choreo_device_matmul "$nexe" --m=4096 --n=4096 --k=4096 2>&1)
    local s_ncu=$(CUDA_VISIBLE_DEVICES=$GPU ncu --metrics $NCU_METRICS -k __choreo_device_matmul "$sexe" --m=4096 --n=4096 --k=4096 2>&1)
    local bsmem=$(echo "$n_ncu" | grep "shared_mem_per_block_dynamic" | head -1 | awk '{printf "%.1f", $NF}')
    local ssmem=$(echo "$s_ncu" | grep "shared_mem_per_block_dynamic" | head -1 | awk '{printf "%.1f", $NF}')
    local bocc=$(echo "$n_ncu" | grep "occupancy_limit_shared" | head -1 | awk '{print $NF}')
    local socc=$(echo "$s_ncu" | grep "occupancy_limit_shared" | head -1 | awk '{print $NF}')
    local regs=$(echo "$n_ncu" | grep "registers_per_thread" | head -1 | awk '{print $NF}')
    local sregs=$(echo "$s_ncu" | grep "registers_per_thread" | head -1 | awk '{print $NF}')
    local bact=$(echo "$n_ncu" | grep "sm__warps_active.avg" | head -1 | awk '{print $NF}')
    local sact=$(echo "$s_ncu" | grep "sm__warps_active.avg" | head -1 | awk '{print $NF}')
    local sp=""
    if [[ -n "$bsmem" && -n "$ssmem" && "$bsmem" != "0" ]]; then
        sp=$(echo "scale=1; 100*($bsmem-$ssmem)/$bsmem" | bc)
    fi
    printf "%-10s  %-10s  %-10s  %-6s  %-6s  %-7s  %s\n" "$name" "$bsmem KB" "$ssmem KB" "$sp%" "$bocc->$socc" "$regs->${sregs:-NA}" "${bact:-NA}->${sact:-NA}"
}

BENCH="$REPO/benchmarks/matmul"
measure "1P1C-4s" "$BENCH/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co"
measure "1P1C-e4m3" "$BENCH/matmul_e4m3_dyn_sm90_warpspec_1p1c.co"
measure "1P1C-3s" "$BENCH/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_3s.co"
measure "1P2C" "$BENCH/matmul_f16_dyn_sm90_warpspec_1p2c.co"
measure "1P3C" "$BENCH/matmul_f16_dyn_sm90_warpspec_1p3c.co"

# ---- Table 3 (tab:occupancy) FA rows: Act. (sm__warps_active) at the
# ---- paper's pinned config B=2 H=16 SEQ=16384 (6th of the harness's 6
# ---- configs -> launch 6 with WARMUP=0 REPEAT=1 -> --launch-skip 5).
FA="$REPO/benchmarks/flash_atten/fa_tuned_1p2c.co"
fa_measure() {  # $1=label $2=co $3=modeflag -> prints "smem|occ|regs|act"
    # Full stats for the FA rows (Table 3): the same ncu launches that
    # measure Act. also report SMEM/occupancy/registers. Compile from
    # the kernel's dir with a relative path: the -gs result script
    # resolves "fa_helper.hpp" / "build/bench_configs.inc" relative to
    # the .co's path.
    local label=$1 co=$2 flag=$3
    local d=$(dirname "$co") b=$(basename "$co")
    local r="$TMPDIR/${label}.result"
    ( cd "$d" \
      && "$CHOREO" -gs -t cute -arch="$ARCH" $flag "$b" -o "$r" 2>/dev/null ) \
        || { echo "NA|NA|NA|NA"; return; }
    bash "$r" --compile-link 2>/dev/null || { echo "NA|NA|NA|NA"; return; }
    local exe=$(grep -oP '(?<=-o )\S+\.exe' "$r" | head -1)
    local out=$(CUDA_VISIBLE_DEVICES=$GPU CHOREO_TIMING_WARMUP=0 CHOREO_TIMING_REPEAT=1 \
        ncu --metrics $NCU_METRICS --launch-skip 5 --launch-count 1 \
        -k __choreo_device_flash_atten "$exe" 2>&1)
    local smem=$(echo "$out" | grep 'shared_mem_per_block_dynamic' | head -1 | awk '{print $NF}')
    local occ=$(echo "$out" | grep 'occupancy_limit_shared' | head -1 | awk '{print $NF}')
    local regs=$(echo "$out" | grep 'registers_per_thread' | head -1 | awk '{print $NF}')
    local act=$(echo "$out" | grep 'sm__warps_active' | head -1 | awk '{print $NF}')
    echo "${smem:-NA}|${occ:-NA}|${regs:-NA}|${act:-NA}"
}
f1=$(fa_measure fa_tuned_nosala "$FA" "--no-sala")
f2=$(fa_measure fa_tuned_sala   "$FA" "")
f3=$(fa_measure fa_3s_sala       "$REPO/benchmarks/flash_atten/fa_3s.co" "")
IFS='|' read -r b1 o1 r1 a1 <<< "$f1"
IFS='|' read -r b2 o2 r2 a2 <<< "$f2"
IFS='|' read -r b3 o3 r3 a3 <<< "$f3"
printf "%-10s  %-10s  %-10s  %-6s  %-6s  %-7s  %s\n" "FA tuned" "$b1 KB" "$b2 KB" \
    "$(echo "scale=1; 100*($b1-$b2)/$b1" | bc)%" "$o1->$o2" "$r1->$r2" "$a1->$a2"
printf "%-10s  %-10s  %-10s  %-6s  %-6s  %-7s  %s\n" "FA 3s" "256*" "$b3 KB" \
    "$(echo "scale=1; 100*(262144/1024-$b3)/(262144/1024)" | bc)%" "NA->$o3" "NA->$r3" "NA->$a3 (SALA)"

echo ""
echo "Paper expects: 1P1C: 116->99 (15%) | e4m3: 116->99 (15%) | 3s: 91->75 (18%) | 1P2C: 132->99 (25%) | 1P3C: 132->83 (37%)"
echo "Act. column (sm__warps_active): GEMM rows at the kernels' default 2048^3 workload;"
echo "FA rows at B=2 H=16 SEQ=16384. Paper Table 3: 1P1C-4s 7.6->14.8% | 1P1C-3s 14.2->20.4% |"
echo "1P2C 18.0->18.0% | 1P3C 24.0->24.0% | FA tuned 18.4->18.4% | FA 3s 18.3% (SALA)."
