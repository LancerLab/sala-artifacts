#!/usr/bin/env bash
# One-time CUTLASS v4.5.0 setup for the CUTLASS rows of
# tab:cross-framework: clones the pristine v4.5.0 checkout into
# benchmarks/cutlass/extern/cutlass_include (the struct->union patch is
# applied by table_cross_framework_cutlass.sh to its own working copy).
# The compiler's own CUTLASS (v4.2.1) lives in croqtile/extern/cutlass
# and is fetched by the configure step when missing — not this script's
# job in the standalone layout.
# The clones use HTTP/1.1 (git -c): Ubuntu's git-over-HTTP/2 with GnuTLS
# can fail with 'GnuTLS recv error (-110)' under flaky networks.
set -euo pipefail
REPO="$(cd "$(dirname "$0")" && pwd)"   # this script sits at the repo root
GIT="git -c http.version=HTTP/1.1"

DEST="$REPO/benchmarks/cutlass/extern/cutlass_include"
if [[ ! -f "$DEST/include/cutlass/cutlass.h" ]]; then
    echo "[setup] cloning NVIDIA/cutlass v4.5.0 -> $DEST"
    mkdir -p "$DEST"
    $GIT clone --depth 1 --branch v4.5.0 \
        https://github.com/NVIDIA/cutlass.git "$DEST"
fi

# The struct->union patch is applied by table_cross_framework_cutlass.sh to
# its own working copy (the pristine clone stays untouched for the baseline).
echo "[setup] done."
