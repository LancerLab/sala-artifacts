#!/usr/bin/env bash
# Figure 2 (fig:hb-results): all nine items in one command (no GPU, ~10 s).
#   Items 1-4, 7, 8 — kernel compiles (compiler-internal allocation sizes)
#   Items 5, 6, 9  — HB-analyzer buffer-size models
# Usage: bash reproduce/figure2_compiler_smem.sh
#        CHOREO=/path/to/choreo bash reproduce/figure2_compiler_smem.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${ARCH:-sm_90a}"
BENCH="$REPO/benchmarks"

# The kernels need a choreo that accepts `mma.commit` (the original eval
# token). Prefer CHOREO from the env, then the original eval env, then the
# submodule build.
if [[ -z "${CHOREO:-}" ]]; then
    for c in "$REPO/croqtile/build/choreo"; do
        [[ -x "$c" ]] && { CHOREO="$c"; break; }
    done
fi
[[ -x "${CHOREO:-}" ]] || {
    echo "Error: choreo not found. Set CHOREO=/path/to/choreo or build one:"
    echo "  cmake -S croqtile -B croqtile/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCHOREO_DEFAULT_TARGET=cute && ninja -C croqtile/build choreo copp"
    exit 1
}

smem() { "$CHOREO" -gs -t cute -arch="$ARCH" $2 "$1" -o /dev/null 2>&1 \
           | grep -oP 'set to `\K[0-9]+' | head -1 || true; }
oob() { "$CHOREO" -gs -t cute -arch="$ARCH" $2 "$1" -o /dev/null 2>&1 \
           | grep -c 'OUT OF BOUND' || true; }

FA="$BENCH/flash_atten/fa_tuned_1p2c.co"
ANALYZER_OUT=$(cd "$REPO" && PYTHONPATH=. python3 -m hb_analyzer all-examples 2>&1 || true)
a_row() {  # $1=label $2=analyzer-kernel-name $3=paper
    local line=$(echo "$ANALYZER_OUT" | grep -m1 "$2 ")
    local base=$(echo "$line" | awk '{print $3}')
    local sala=$(echo "$line" | awk '{print $4}')
    local sp=$(echo "$line" | awk '{print $6}' | tr -d "%")
    printf "%-14s  %-8s  %-8s  %-6s  %s\n" "$1" \
        "$(echo "$base/1024" | bc) KB" "$(echo "$sala/1024" | bc) KB" "$sp%" "$3"
}
FA3S=/tmp/figure2_fa_3s.co
sed 's/#define STAGES 2/#define STAGES 3/' "$FA" > "$FA3S"

printf "%-14s  %-8s  %-8s  %-6s  %s\n" "Fig.2 item" "Base KB" "SALA KB" "Save%" "Paper"
printf "%-14s  %-8s  %-8s  %-6s  %s\n" "----------" "-------" "-------" "-----" "------------"

row() {  # $1=label $2=base_b $3=sala_b $4=paper
    local label=$1 base_b=$2 sala_b=$3 paper=$4
    if [[ -n "$base_b" && -n "$sala_b" && "$base_b" != "0" ]]; then
        local bk=$(echo "$base_b/1024" | bc)
        local sk=$(echo "$sala_b/1024" | bc)
        local sp=$(echo "scale=1; 100*($base_b-$sala_b)/$base_b" | bc)
        printf "%-14s  %-8s  %-8s  %-6s  %s\n" "$label" "$bk KB" "$sk KB" "$sp%" "$paper"
    else
        printf "%-14s  %-8s  %-8s  %-6s  %s\n" "$label" "ERR" "ERR" "-" "$paper"
    fi
}

# ---- kernel items 1-4 (GEMM) ----
row "1P1C f16"  "$(smem $BENCH/matmul/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co --no-sala)" \
                "$(smem $BENCH/matmul/matmul_f16_dyn_sm90_warpspec_1p1c_64x128_4s.co "")" "112->96 (-14%)"
row "1P1C e4m3" "$(smem $BENCH/matmul/matmul_e4m3_dyn_sm90_warpspec_1p1c.co --no-sala)" \
                "$(smem $BENCH/matmul/matmul_e4m3_dyn_sm90_warpspec_1p1c.co "")" "112->96 (-14%)"
row "1P2C f16"  "$(smem $BENCH/matmul/matmul_f16_dyn_sm90_warpspec_1p2c.co --no-sala)" \
                "$(smem $BENCH/matmul/matmul_f16_dyn_sm90_warpspec_1p2c.co "")" "128->96 (-25%)"
row "1P3C f16"  "$(smem $BENCH/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co --no-sala)" \
                "$(smem $BENCH/matmul/matmul_f16_dyn_sm90_warpspec_1p3c.co "")" "128->80 (-37%)"

# ---- model items 5, 6 ----
a_row "FA K/V"  "fa_kv_overlap"          "40->32 (-20%)"
a_row "FA full" "fa_fwd_full_pipeline"   "72->56 (-22%)"

# ---- kernel items 7, 8 (FA) ----
row "FA tuned" "$(smem "$FA" --no-sala)" "$(smem "$FA" "")" "192->160 (-17%)"

FA3S_NO=$(smem "$FA3S" --no-sala); FA3S_S=$(smem "$FA3S" "")
if [[ -n "$FA3S_S" ]]; then
    # baseline cannot compile: 256 KB = q 32 + k 3x32 + v 3x32 + o 32
    printf "%-14s  %-8s  %-8s  %-6s  %s\n" "FA 3s" "256 KB*" \
        "$(echo "$FA3S_S/1024" | bc) KB" \
        "$(echo "scale=1; 100*(262144-$FA3S_S)/262144" | bc)%" "256*->224 (-12%)"
else
    printf "%-14s  %-8s  %-8s  %-6s  %s\n" "FA 3s" "ERR" "ERR" "-" "256*->224 (-12%)"
fi
rm -f "$FA3S"

# ---- item 9: conv kernel compile (the paper's 'verified by compilation') ----
CONV="$REPO/benchmarks/conv/conv2d_fprop_ws_1p1c.co"
CONV_B=$($CHOREO -gs -t cute -arch="$ARCH" --no-sala "$CONV" -o /dev/null 2>&1 | grep -oP 'set to `\K[0-9]+' | head -1)
CONV_S=$($CHOREO -gs -t cute -arch="$ARCH"           "$CONV" -o /dev/null 2>&1 | grep -oP 'set to `\K[0-9]+' | head -1)
printf "%-14s  %-8s  %-8s  %-6s  %s\n" "Conv2d"     "$(echo "$CONV_B/1024" | bc) KB" "$(echo "$CONV_S/1024" | bc) KB"     "$(echo "scale=1; 100*($CONV_B-$CONV_S)/$CONV_B" | bc)%" "96->64 (-33%)"


echo ""
echo "* FA 3s baseline: exceeds the SMEM bound (cannot compile) — the paper's star."
echo "  Items 5/6 are HB-analyzer models (hb_analyzer/examples/); item 9"
echo "  is the conv kernel compile (benchmarks/conv/conv2d_fprop_ws_1p1c.co)."
