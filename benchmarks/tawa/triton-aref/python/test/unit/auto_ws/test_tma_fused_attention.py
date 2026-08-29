"""
Fused Attention with TMA
========================


"""

try:  # for pytest
    from .fmha_common import *
except:  # for benchmark
    from fmha_common import *

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    k_desc_ptr,
    v_desc_ptr,  #
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    offs_hz: tl.constexpr,
    N_CTX: tl.constexpr,
    fp8_v: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offs_kv = offs_hz * N_CTX + lo

    # loop over k, v and update accumulator
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
        k = k_desc_ptr.load(
            [offs_kv, 0],
        )
        qk = tl.dot(q, k.T)
        if STAGE == 2:
            # tl.device_print("start_n:", start_n)
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]
        # update acc
        if fp8_v:
            v = v_desc_ptr.load(
                [offs_hz * HEAD_DIM, start_n],
            )
            p = p.to(tl.float8e5)
            acc = tl.dot(p, v.T, acc)
        else:
            v = v_desc_ptr.load(
                [offs_kv, 0],
            )
            p = p.to(tl.float16)
            acc = tl.dot(p, v, acc)
        # update m_i and l_i
        m_i = m_ij
        offs_kv += BLOCK_N
    return acc, l_i, m_i


# @triton.autotune(configs_tma, key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    q_desc_ptr,
    k_desc_ptr,
    v_desc_ptr,
    o_desc_ptr,
    m_desc_ptr,
    sm_scale,
    M,
    Out,  #
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,  #
    Z,
    H,
    N_CTX,  #
    HEAD_DIM: tl.constexpr,  #
    BLOCK_M: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    STAGE: tl.constexpr,  #
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [q_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [k_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [v_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # """
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [o_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # tl.inline_asm_elementwise(
    #     "fence.proxy.tensormap::generic.acquire.gpu [$1], 128; // $0 dummy reg",
    #     "=r, l",
    #     [m_desc_ptr],
    #     dtype=tl.int32,
    #     is_pure=False,
    #     pack=1,
    # )
    # """
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = q_desc_ptr.load(
        [off_hz * N_CTX + start_m * BLOCK_M, 0],
    )

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc_ptr,
            v_desc_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            4 - STAGE,
            offs_m,
            offs_n,
            off_hz,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(
            acc,
            l_i,
            m_i,
            q,
            k_desc_ptr,
            v_desc_ptr,  #
            start_m,
            qk_scale,  #
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,  #
            2,
            offs_m,
            offs_n,
            off_hz,
            N_CTX,
            V.dtype.element_ty == tl.float8e5,  #
        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    """
    tl._experimental_descriptor_store(
        m_desc_ptr,
        m_i,
        [off_hz * N_CTX + start_m * BLOCK_M],
    )
    tl._experimental_descriptor_store(
        o_desc_ptr,
        acc.to(Out.type.element_ty),
        [off_hz * N_CTX + start_m * BLOCK_M, 0],
    )
    """
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))


def attention(
    q,
    k,
    v,
    causal,
    sm_scale,
    NUM_WARPS,
    USE_TTG_WS,
    WG_SPEC,
    MATH_WG_PIPE,
    FORCE_MEMBAR,
):
    return run_attention(
        _attn_fwd,
        q,
        k,
        v,
        causal,
        sm_scale,
        BLOCK_M=128,
        BLOCK_N=128,
        NUM_STAGES=2,
        NUM_WARPS=NUM_WARPS,
        USE_TTG_WS=USE_TTG_WS,
        WG_SPEC=WG_SPEC,
        MATH_WG_PIPE=MATH_WG_PIPE,
        FORCE_MEMBAR=FORCE_MEMBAR,
    )


@pytest.mark.parametrize(
    "Z, H, N_CTX, HEAD_DIM, WG_SPEC",
    [
        (2, 2, 1024, 128, "tma_load_first"),
        (2, 2, 2048, 128, "mma_first"),
        (2, 2, 4096, 128, None),
        (2, 2, 8192, 128, "mma_first"),
        (2, 2, 16384, 128, None),
    ],
)
@pytest.mark.parametrize("math_wg_pipe", [False, True])
@pytest.mark.parametrize("causal", [False, True])
# Disable USE_TTG_WS until proper integration is done
# @pytest.mark.parametrize("USE_TTG_WS", [False, True])
@pytest.mark.parametrize("USE_TTG_WS", [False])
def test_op(
    Z,
    H,
    N_CTX,
    HEAD_DIM,
    WG_SPEC,
    math_wg_pipe,
    causal,
    USE_TTG_WS,
    dtype=torch.float16,
):
    if torch.cuda.get_device_capability()[0] >= 10:
        # math_wg_pipe=True isn't supported on Blackwell yet
        if math_wg_pipe == True:
            pytest.skip("math wg pipelining isn't supported on Blackwell")
        # on blackwell there is a race in 8 warps mode
        # we will first make 8 warps to work for matmul then look at FMHA
        NUM_WARPS = 4
        MATH_WG_PIPE = False
    elif torch.cuda.get_device_capability()[0] >= 9:
        MATH_WG_PIPE = math_wg_pipe
        NUM_WARPS = 8
    else:
        pytest.skip("causal attention isn't supported on sm <= 9.0")

    torch.manual_seed(20)
    sm_scale = 0.5
    q, k, v = init_tensors(Z, H, N_CTX, HEAD_DIM, dtype)
    # triton implementation
    tri_out = attention(
        q,
        k,
        v,
        causal,
        sm_scale,
        NUM_WARPS,
        USE_TTG_WS,
        WG_SPEC,
        MATH_WG_PIPE,
        FORCE_MEMBAR=False,
    )
    # official FA implementation (broken for fp8)
    if supports_hopper() and dtype == torch.float16:
        fa2_out = triton_reference(q, k, v, causal, sm_scale)
        assert torch.allclose(tri_out, fa2_out, atol=1e-2, rtol=1.0 - 3)

    ref_out = torch_reference(q, k, v, causal, sm_scale, N_CTX, dtype)

    # compare
    # assert torch.allclose(ref_out, tri_out, atol=1e-2, rtol=0)
    # print("ref_out :", ref_out)
    # print("tri_out :", tri_out)
    ERROR_TOLERANCE = 2e-3 if dtype == torch.float16 else 3e-2
    assert_close_verbose(
        tri_out.to(torch.float16),
        ref_out.to(torch.float16),
        rtol=ERROR_TOLERANCE,
        atol=ERROR_TOLERANCE,
    )


BATCH, N_HEADS, HEAD_DIM = 4, 32, 128
HAS_FLASH_BENCH = HAS_FLASH
HAS_FLASH_BENCH = False
# vary seq length for fixed head and batch=4
configs = []
for mode in ["fwd"]:
    for causal in [False, True]:
        for provider in ["triton-fp16", "triton-fp8"]:
            if mode == "bwd" and not causal:
                continue
            configs.append(
                triton.testing.Benchmark(
                    x_names=["N_CTX"],
                    x_vals=[2**i for i in range(10, 15)],
                    line_arg="provider",
                    line_vals=[f"{provider}"] + (["flash"] if HAS_FLASH_BENCH else []),
                    line_names=[f"{provider}"]
                    + (
                        ["Flash-3"]
                        if HAS_FLASH_BENCH and supports_hopper()
                        else ["Flash-2"] if HAS_FLASH_BENCH else []
                    ),
                    styles=[("red", "-"), ("blue", "-"), ("green", "-")],
                    ylabel="ms",
                    plot_name=f"fused-attention-batch{BATCH}-head{N_HEADS}-d{HEAD_DIM}-{provider}-{mode}-causal={causal}",
                    args={
                        "H": N_HEADS,
                        "BATCH": BATCH,
                        "HEAD_DIM": HEAD_DIM,
                        "mode": mode,
                        "causal": causal,
                    },
                )
            )


@triton.testing.perf_report(configs)
def bench_flash_attention(
    BATCH,
    H,
    N_CTX,
    HEAD_DIM,
    causal,
    mode,
    provider,
    USE_TTG_WS=False,
    MATH_WG_PIPE=True,
    NUM_WARPS=8,
    WG_SPEC="mma_first",
    FORCE_MEMBAR=False,
    device="cuda",
):
    def bench_fn(q, k, v, causal, sm_scale):
        return attention(
            q,
            k,
            v,
            causal,
            sm_scale,
            NUM_WARPS,
            USE_TTG_WS,
            WG_SPEC,
            MATH_WG_PIPE,
            FORCE_MEMBAR,
        )

    return bench_flash_attention_with_configs(
        bench_fn, BATCH, H, N_CTX, HEAD_DIM, causal, mode, provider, device
    )


if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM = (2, 2, 16384, 128)
    wg_spec = ()
    wg_spec = "mma_first"
    # wg_spec = "tma_load_first"
    use_ttg_ws = False
    force_membar = False
    math_wg_pipe = False
    # math_wg_pipe = False
    # use_ttg_ws = True
    # force_membar = True  # when using with ttng.wg perf is closer to ttg.ws=true
    if 1:
        test_op(
            Z,
            H,
            N_CTX,
            HEAD_DIM,
            USE_TTG_WS=use_ttg_ws,
            WG_SPEC=wg_spec,
            math_wg_pipe=math_wg_pipe,
            causal=True,
            dtype=torch.float16,
        )
    else:
        test_op(
            Z,
            H,
            N_CTX,
            HEAD_DIM,
            USE_TTG_WS=use_ttg_ws,
            WG_SPEC=wg_spec,
            math_wg_pipe=math_wg_pipe,
            causal=False,
            dtype=torch.float8_e5m2,
        )

    bench_flash_attention.run(
        USE_TTG_WS=use_ttg_ws,
        WG_SPEC=wg_spec,
        FORCE_MEMBAR=force_membar,
        save_path=".",
        print_data=True,
    )
