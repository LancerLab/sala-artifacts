#!/usr/bin/env python3
"""Run Tawa TMA FMHA kernel once for ncu profiling.

Usage:
  # 2-stage baseline:
  SALA_ENABLE=0 ncu --metrics l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,launch__shared_mem_per_block_driver,launch__shared_mem_per_block_static,launch__registers_per_thread,launch__occupancy_per_register,launch__occupancy_per_shared_mem_block,sm__warps_active.avg.pct_of_peak_sustained_active PYTHONPATH=/home/wsj/dev/triton-aref/python python3.10 tawa_fmha_ncu.py --stages 2

  # 3-stage SALA:
  SALA_ENABLE=1 ncu --metrics ... python3.10 tawa_fmha_ncu.py --stages 3

  # Simple run (for ncu --set full):
  SALA_ENABLE=1 ncu --set full -o tawa_fmha_3s_sala PYTHONPATH=/home/wsj/dev/triton-aref/python python3.10 tawa_fmha_ncu.py --stages 3
"""
import sys, os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--stages", type=int, default=3)
parser.add_argument("--seq", type=int, default=4096)
parser.add_argument("--gpu", type=int, default=1)
parser.add_argument("--check", action="store_true",
                    help="compare the output against a fp32 torch reference "
                         "(run WITHOUT ncu: the reference uses torch kernels)")
parser.add_argument("--membar", type=int, default=1, choices=(0, 1),
                    help="force the cross-tile mbarrier synchronization the SALA "
                         "overlap requires (default 1: without it the 3-stage "
                         "kernel races and produces NaNs; SMEM is identical)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

sys.modules['pytest'] = type(sys)('pytest')
sys.modules['pytest'].mark = type(sys)('mark')
sys.modules['pytest'].mark.parametrize = lambda *a, **kw: (lambda f: f)
sys.modules['pytest'].skip = lambda msg='': None

import torch, triton

sys.path.insert(0, os.environ.get("TRITON_AREF", "/home/wsj/dev/triton-aref")
                + "/python/test/unit/auto_ws")
from fmha_common import init_tensors, run_attention
from test_tma_fused_attention import _attn_fwd

BATCH, H, HEAD_DIM = 4, 32, 128
dtype = torch.float16
causal = True
sm_scale = 1.3
N_CTX = args.seq

q, k, v = init_tensors(BATCH, H, N_CTX, HEAD_DIM, dtype)

# Warmup
for _ in range(3):
    o = run_attention(
        _attn_fwd, q, k, v, causal, sm_scale,
        BLOCK_M=128, BLOCK_N=128, NUM_STAGES=args.stages,
        NUM_WARPS=8, USE_TTG_WS=False,
        WG_SPEC="mma_first", MATH_WG_PIPE=True,
        FORCE_MEMBAR=bool(args.membar),
    )
torch.cuda.synchronize()

# One more run for profiling
o = run_attention(
    _attn_fwd, q, k, v, causal, sm_scale,
    BLOCK_M=128, BLOCK_N=128, NUM_STAGES=args.stages,
    NUM_WARPS=8, USE_TTG_WS=False,
    WG_SPEC="mma_first", MATH_WG_PIPE=True,
    FORCE_MEMBAR=bool(args.membar),
)
torch.cuda.synchronize()

print(f"Done: {args.stages}-stage, SEQ={N_CTX}, output shape={o.shape}")

if args.check:
    # Numerical reference: fp32 softmax(q k^T * scale, causal) @ v, computed
    # one (batch, head) at a time to bound memory.
    ref = torch.empty((BATCH, H, N_CTX, HEAD_DIM), dtype=torch.float32, device=o.device)
    rows = torch.arange(N_CTX, device=o.device)
    mask = rows[None, :] > rows[:, None]          # causal: key > query masked
    with torch.no_grad():
        for b in range(BATCH):
            for h in range(H):
                qb = q[b, h].float()
                kb = k[b, h].float()
                vb = v[b, h].float()
                s = (qb @ kb.t()) * sm_scale
                s = s.masked_fill(mask, float("-inf"))
                ref[b, h] = torch.softmax(s, dim=-1) @ vb
    got = o.float()
    diff = (got - ref).abs()
    max_abs = diff.max().item()
    ref_scale = ref.abs().max().item()
    tol = 0.02 + 0.02 * ref_scale
    bad = int((diff > tol).sum().item())
    ok = (bad == 0) and torch.isfinite(got).all().item()
    print(f"FMHA reference (fp32 torch, causal): max|o-ref|={max_abs:.4f} "
          f"(ref scale {ref_scale:.3f}, tol {tol:.4f}), {bad} element(s) over tol "
          f"-> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
