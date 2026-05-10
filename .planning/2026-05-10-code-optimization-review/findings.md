# Findings: Code Optimization Review

## Initial Inventory
- Repository contains baseline commit `c610de3` plus seven post-baseline commits on `kunpeng-host-perf-phase2`.
- Optimization code changes are limited to GPU H2D batching/NUMA binding, Ascend K/V swap parallelism/NUMA binding/FlashLB debounce, CPU worker cache, and planning documents.
- Existing progress/findings files are under `.planning/2026-05-09-kunpeng-host-perf-analysis/` and `.planning/2026-05-10-kunpeng-perf-review/`.
- `.planning/2026-05-10-kunpeng-perf-review/findings.md` already records a peer-review pivot away from speculative Python-to-C++/SVE work because many hot paths are already implemented in C/C++/Rust or are not app-level bottlenecks.

## Confirmed Code Issues
- `vllm-ascend-0.18.0/vllm_ascend/eplb/core/policy/policy_flashlb.py` contains literal placeholder lines `import ...existing imports...` and `...`; `python3 -m py_compile` fails with `SyntaxError` at line 505.
- GPU NUMA binding discovers GPU NUMA node with `nvmlDeviceGetHandleByIndex(0)` inside a static method, ignoring the worker's `local_rank`; multi-GPU workers can bind memory to the wrong NUMA node.
- Both GPU and Ascend NUMA binding code call `libc.mbind(addr=NULL, len=0, ...)` and describe it as process/future allocation policy. Local verification shows `libc.so.6` does not expose an `mbind` symbol, so the current `ctypes.CDLL(...).mbind(...)` call can raise `AttributeError`; even with a syscall wrapper, `mbind` is range-based and not the right interface for future allocation policy.
- Other changed Python files compile when `policy_flashlb.py` is excluded.

## Planning Logic Issues
- The 2026-05-09 corrected report correctly separates CPU-only SIMD/SVE optimization from GPU/NPU HOST-control and data-movement optimization.
- The 2026-05-10 Phase 2 findings/task plan correctly pivot away from speculative Python-to-C++/SVE work, but `final_report.md` and `implementation_plan.md` in the same directory still recommend SVE hashing, metadata sinking, and SHM IPC based on assumptions that `phase2_review.md` already rejected.
- The first-round implementation plan contains a critical API assumption error around `mbind`: it claims `mbind(0, 0, ...)` sets process-wide/future allocation policy, which is not a safe Linux NUMA API assumption.

## Consolidated Direction
- For Kunpeng+NV GPU and Kunpeng+Ascend, the next useful HOST-side work should target Python/PyTorch object churn, metadata packing, small tensor H2D/N2D transfers, scheduler data structures, and CPU affinity/isolation.
- Avoid broad "rewrite Python in C++" work. Use profiling to identify narrow hot loops or packing functions, then consider C++/Rust only for stable, small-surface paths.
- TTFT is most sensitive to prefill metadata construction, small H2D/N2D transfers, prefix/KV-cache work, and offload paths. TPOT is most sensitive to scheduler per-step overhead, output processing, spec-decode metadata, and HOST gaps between accelerator kernels.
