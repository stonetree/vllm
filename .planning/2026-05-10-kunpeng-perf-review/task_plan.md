# Task Plan - Kunpeng Host Performance Analysis Phase 2 (Pivoted)

## Goal
Validate current bottlenecks on Kunpeng and implement high-ROI optimizations (oneDNN+ACL, isolcpus) while discarding low-ROI "Python-to-C++" tasks.

## Phases

### Phase 1: Review Current Optimizations (Complete)
- [x] Analyze batching H2D copies.
- [x] Analyze NUMA memory binding.
- [x] Analyze Parallel K/V swap.
- [x] Analyze FlashLB debounce.

### Phase 2: Empirical Profiling & Verification
- [ ] Run benchmark with `py-spy` or `nsys` to identify real host-side bottlenecks.
- [ ] Verify `oneDNN+ACL` loading on Kunpeng (check environment and library paths).
- [ ] Check `torch.ops._C` kernel dispatch for ARM (ensure SVE path is hit).

### Phase 3: High-ROI Optimizations
- [ ] Implement `isolcpus` detection in `vllm_ascend/cpu_binding.py`.
- [ ] Enable/Fix dispatch for existing C++ kernels (RoPE/Activation) if bypassed.

### Phase 4: Revised Final Recommendation
- [ ] Document profiling results.
- [ ] Summarize ROI of implemented/discarded tasks.
