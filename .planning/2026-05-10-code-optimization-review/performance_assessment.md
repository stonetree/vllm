# Quantitative Assessment: First Two Optimization Rounds

## Scope
This assessment covers code changes after baseline `c610de3` in the current branch:

- GPU H2D metadata copy reordering.
- GPU NUMA memory policy.
- Ascend K/V swap parallelism.
- Ascend NUMA memory policy.
- FlashLB recomputation debounce.
- Low-hanging host optimizations: NumPy buffer reuse, OpenMP affinity defaults, jemalloc compatibility.

The values below are estimates from code-path analysis and Amdahl-style folding. They are not hardware measurements.

## Overall Expected Benefit

| Scenario | Typical Expected Gain | Best Plausible Case | Notes |
|---|---:|---:|---|
| Kunpeng + NV GPU | 1-4% lower step/TPOT or higher throughput | ~8% | Requires multi-NUMA and visible H2D/HOST gaps. |
| Kunpeng + Ascend | 2-6% typical | 10-15% | Higher only when KV offload, MoE/EPLB, or N2D copy overhead is material. |

## Itemized Estimate

| Optimization | Trigger | Component-Level Effect | End-to-End Effect |
|---|---|---:|---:|
| GPU H2D copy reordering | GPU worker metadata copies per step | Saves small transfer gaps, roughly 5-20us/step if copies were fragmented | 0.3-2%; previous 3-8% is optimistic without trace evidence |
| GPU NUMA memory policy | Multi-NUMA, wrong-node CPU/pinned allocations | H2D bandwidth/latency can improve 10-30% | 0-3% typical; 5-8% only if H2D is a major bottleneck |
| Ascend K/V dual-stream swap | KV offload enabled | swap phase may drop 20-45% if copy engines/streams overlap | 2-12% depending on swap share of TTFT |
| Ascend NUMA memory policy | Multi-NUMA and NPU-local CPU buffers matter | N2D/D2H improves 5-20% | 0-3% typical; 3-5% in remote NUMA cases |
| FlashLB debounce | MoE + EPLB recomputes frequently | FlashLB recomputation reduced 50-95% during stable load | 0-3% typical; 3-8% in EPLB-hot MoE workloads |
| NumPy token-count buffer reuse | Every GPU step | Saves small allocation/copy, roughly 1-2us/step | <0.2% |
| OpenMP affinity defaults | Ascend ARM, OpenMP/ACL CPU threads active | Less cross-core drift, better cache locality | 0-2% typical; P99 may improve more |
| jemalloc compatibility | CPU worker or allocation-heavy startup/runtime | Mostly compatibility; runtime benefit uncertain | 0-1% for online GPU/NPU inference |

## Interpretation

- First-round gains are credible but should not be expected to produce double-digit improvements in normal GPU/NPU online inference.
- The strongest likely gains are conditional: KV offload, MoE/EPLB, bad NUMA placement, or visibly fragmented transfer timelines.
- The second round mostly improves mergeability and low-risk host hygiene; it should be measured as tail-latency and stability work rather than headline throughput work.

## Required Evidence Before Claiming Success

Use target hardware to capture:

- TTFT mean/P90/P99.
- TPOT mean/P90/P99.
- Scheduler step wall time.
- Input preparation wall time.
- H2D/N2D memcpy count and duration.
- Device idle gaps caused by HOST.
- `numastat -p` local vs remote memory placement.

