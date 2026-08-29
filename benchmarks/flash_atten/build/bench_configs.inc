// AE benchmark configs for the FA harness (causal prefill, D=128 bf16).
#pragma once

static const mha_helper::BenchConfig kBenchConfigs[] = {
    {2, 16, 512, 512, true, "B=2 H=16 SEQ=512 causal prefill"},
    {2, 16, 1024, 1024, true, "B=2 H=16 SEQ=1024 causal prefill"},
    {2, 16, 2048, 2048, true, "B=2 H=16 SEQ=2048 causal prefill"},
    {2, 16, 4096, 4096, true, "B=2 H=16 SEQ=4096 causal prefill"},
    {2, 16, 8192, 8192, true, "B=2 H=16 SEQ=8192 causal prefill"},
    {2, 16, 16384, 16384, true, "B=2 H=16 SEQ=16384 causal prefill"},
};
static constexpr int kBenchConfigCount = sizeof(kBenchConfigs) / sizeof(kBenchConfigs[0]);
