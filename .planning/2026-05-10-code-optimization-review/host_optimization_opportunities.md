# Next-Round HOST-Side Optimization Opportunities

## Guiding Principle
For Kunpeng+NV GPU and Kunpeng+Ascend, the accelerator performs model compute. The Kunpeng CPU affects customer-visible TTFT/TPOT mainly through control-plane latency, object churn, metadata packing, small transfers, IPC, output processing, and NUMA/affinity behavior.

Reducing Python/PyTorch framework tax is a valid next direction, but it should be driven by profiling and focused on narrow hot paths. Avoid large rewrites of scheduler or model runner logic before evidence exists.

## P0 Candidates

### 1. Ascend Metadata Copy Batching

Target examples:

- `vllm-ascend-0.18.0/vllm_ascend/worker/model_runner_v1.py`
- `query_start_loc`, `seq_lens`, `positions`, `discard_request_indices`, draft/spec-decode metadata.

Hypothesis:

- Ascend still has many scattered `copy_to_gpu()` calls and small N2D metadata transfers.
- Batching and deferring these copies should reduce TTFT and device HOST gaps.

Expected impact:

- TTFT: 2-8%.
- TPOT: 0.5-3%.
- Best on prefill-heavy, small-batch, and high-concurrency mixed workloads.

### 2. Ascend Spec Decode / Rejection Sampler Buffer Reuse

Target examples:

- `vllm-ascend-0.18.0/vllm_ascend/sample/rejection_sampler.py`
- Frequent `torch.tensor(..., pin_memory=True)`, `torch.arange(..., pin_memory=True)`, `torch.ones/full(..., pin_memory=True)`.

Hypothesis:

- Per-step pinned CPU tensor construction and small N2D copies become visible on Kunpeng.
- Runner-level persistent pinned buffers can remove allocator and registration overhead.

Expected impact:

- Spec decode TPOT: 2-6%.
- Non-spec-decode workloads: little or no effect.

### 3. GPU Async Scheduling Scatter Index Buffer Reuse

Target example:

- `vllm-0.18.0/vllm/v1/worker/gpu_model_runner.py`, around async scheduling scatter index tensor construction.

Hypothesis:

- Creating Python lists, pinned tensors, and H2D index tensors per step creates avoidable HOST overhead.
- Persistent CPU/GPU index buffers can reduce TPOT in async/spec decode workloads.

Expected impact:

- TPOT: 0.5-3% generally.
- 2-5% in async scheduling or spec-decode-heavy workloads.

### 4. Scheduler Integer-Index Data Path

Target:

- vLLM v1 scheduler output and worker input metadata.

Hypothesis:

- Python dict/list/set and request-id string lookups are more costly on Kunpeng than high-frequency x86.
- A stable integer request index and array-oriented SchedulerOutput can reduce Python object churn.

Possible increments:

- Keep existing public request IDs, but use integer IDs internally in hot paths.
- Represent per-step scheduled tokens, request indices, and block IDs as arrays.
- Avoid repeated string-keyed dict lookups in worker metadata packing.

Expected impact:

- TPOT: 2-7%.
- High-concurrency decode upper bound: 5-10%.

### 5. Engine/Scheduler/Worker CPU Affinity Layering

Target:

- NV GPU worker processes.
- Ascend worker and existing `cpu_binding.py`.
- EngineCore/scheduler/output-processing threads or processes.

Hypothesis:

- Tail latency is sensitive to scheduler core migration and interference from output/IPC/device threads.
- Explicit core partitioning can improve P99 TTFT/TPOT more than average throughput.

Expected impact:

- Mean: 0-3%.
- P99 TTFT/TPOT: 5-20% in noisy or overloaded deployments.

## P1 Candidates

### Prefix Cache Hash Configuration

Do not implement SVE hashing first. Prefer measuring and recommending `xxhash` where correctness policy allows it.

Expected impact:

- Prefix-cache-heavy long-context TTFT: 1-4%.
- Ordinary workloads: near zero.

### ZMQ/msgpack Message Slimming

Do not jump directly to shared memory. First measure and reduce message size/object count in SchedulerOutput and worker responses.

Expected impact:

- TPOT: 0.5-2%.
- Higher only if profiling shows IPC >3% of step time.

### Output Processing Isolation

Keep Rust tokenizer. Focus on Python output assembly, response queueing, and core isolation.

Expected impact:

- High-concurrency streaming P99: 3-10%.

### Ascend KV Offload Fusion

After dual streams, consider fused K/V swap or layer-batched swap submission.

Expected impact:

- KV-offload TTFT: 5-15%.
- No effect when KV offload is disabled.

## Directions To Avoid Unless Profiling Contradicts This

- CPU SVE RoPE/Attention/Activation for GPU/NPU inference: not on the main compute path.
- Broad C++ rewrite of scheduler: high regression risk.
- Metadata prep C++ sinking without evidence: NumPy operations are already C-level.
- Custom SHM/futex IPC before measuring ZMQ/msgpack overhead.
- Treating jemalloc as a primary online inference optimization.

## Profiling Gate

Promote an item to implementation only if at least one is true:

- Python frame consumes >3% wall time in `py-spy`; >8% is P0.
- Native `perf` shows Python object, allocation, or locking overhead >5%.
- Device timeline shows HOST gaps before kernels or many small H2D/N2D copies.
- Benchmark shows P99 TTFT/TPOT outliers correlate with scheduler/input/output phases.

