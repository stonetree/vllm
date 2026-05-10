# Kunpeng Host Performance Optimization Plan (Phase 2)

## 1. Objective
Reduce host-side CPU overhead ("Language Tax") on Kunpeng ARM64 servers by sinking performance-critical Python/PyTorch logic into C++ extensions. Leverage ARM-specific architectural features (SVE, LSE) and zero-copy IPC to maximize online inference throughput for both NV GPU and Ascend NPU.

## 2. Architecture: `vllm_kunpeng_ops`
Create a unified C++ extension module `vllm_kunpeng_ops` using `pybind11`. This module will house all Kunpeng-specific host accelerations.

### 2.1 Compilation Flags
- `-march=armv8.2-a+sve` (Enable SVE and LSE instructions)
- `-O3 -ffast-math`
- `-fopenmp` (For multi-threaded host logic)

---

## 3. Detailed Implementation Tasks

### Task 1: SVE-Accelerated Prefix Caching Hashing
**Target:** `vllm-0.18.0/vllm/v1/core/kv_cache_utils.py`
**Problem:** `hash_block_tokens` uses Python-based hashing, which is slow for long sequences on Kunpeng's single-thread performance.
**Implementation:**
1.  **C++ sinking:** Implement `hash_block_sve` in `csrc/v1/kunpeng/hashing.cpp`.
2.  **SVE usage:** Use SVE intrinsics (`svld1_s32`, `svadd_s32`, etc.) to process token IDs and parent hashes in parallel.
3.  **Python integration:**
    ```python
    # vllm/v1/core/kv_cache_utils.py
    import vllm_kunpeng_ops
    
    def hash_block_tokens(...):
        # Fallback to C++ SVE implementation
        return vllm_kunpeng_ops.hash_block_sve(parent_hash, token_ids, extra_keys)
    ```

### Task 2: Input Metadata Preparation Sinking
**Target:** `vllm-0.18.0/vllm/v1/worker/gpu_model_runner.py`
**Problem:** The `_prepare_inputs` method performs multiple NumPy operations and loops (e.g., `query_start_loc`, `seq_lens` calculation) which are CPU-intensive.
**Implementation:**
1.  **C++ sinking:** Implement `prepare_input_metadata_kunpeng` in `csrc/v1/kunpeng/input_prep.cpp`.
2.  **Logic:** Move the entire logic of calculating `query_start_loc`, `seq_lens`, and `discard_request_mask` into a single C++ function.
3.  **Python integration:**
    ```python
    # vllm/v1/worker/gpu_model_runner.py
    def _prepare_inputs(...):
        # Replace multiple numpy/torch calls with one C++ call
        vllm_kunpeng_ops.prepare_input_metadata_kunpeng(
            self.input_batch, self.query_start_loc.np, self.seq_lens.np, ...
        )
    ```

### Task 3: Shared Memory Engine-Worker IPC
**Target:** `vllm-0.18.0/vllm/v1/engine/core_client.py` and `gpu_worker.py`
**Problem:** gRPC/Unix Sockets for every request metadata transfer introduce 100-200µs latency.
**Implementation:**
1.  **Shared Memory:** Use `multiprocessing.shared_memory` or a raw POSIX shm buffer.
2.  **Zero-copy:** The Engine writes `InputBatch` metadata directly to SHM. The Worker reads it without serialization.
3.  **Synchronization:** Use a lightweight Futex or a SHM-based spinlock (optimized with ARM LSE `yield` or `wfe`).

### Task 4: Parallel Detokenizer Sinking (Ascend Specific)
**Target:** `vllm-ascend-0.18.0/vllm_ascend/detokenizer.py` (if applicable) or common `detokenizer.py`
**Problem:** Detokenization (Token ID -> String) is a sequential bottleneck.
**Implementation:**
1.  **C++ backend:** Use `sentencepiece` or `tokenizers` C++ API directly.
2.  **Parallelization:** Implement a C++ thread pool that handles detokenization for all finished sequences in a batch concurrently.
3.  **Zero-copy access:** Access the GPU/NPU output buffer via Unified Memory or DMA-BUF if supported.

### Task 5: Hardened Affinity & IRQ Steering
**Target:** `vllm-ascend-0.18.0/vllm_ascend/cpu_binding.py`
**Problem:** Simple `mbind` is insufficient for high-load online inference.
**Implementation:**
1.  **Core Isolation:** Detect `isolcpus` from kernel cmdline and prioritize binding the `LLMEngine` to these cores.
2.  **IRQ Steering Script:** Provide a C++/Python utility to auto-detect the PCIe slot of the GPU/NPU and steer corresponding `virtio-net` and `hisi-zip` (or NPU) interrupts to the local NUMA node.

---

## 4. Implementation Schedule

| Phase | Description | Key Deliverable |
|-------|-------------|-----------------|
| Phase 2.1 | Build System & Hashing Sinking | `vllm_kunpeng_ops` with SVE Hashing |
| Phase 2.2 | Metadata Prep Sinking | Accelerated `_prepare_inputs` |
| Phase 2.3 | Shared Memory IPC | Zero-copy `InputBatch` transfer |
| Phase 2.4 | Detokenizer & IRQ Tuning | Multi-threaded C++ Detokenizer |

## 5. Success Criteria
- **Latency:** Reduce host-side pre-processing latency by >30% on sequences >2048 tokens.
- **Throughput:** Increase maximum TPS (Tokens Per Second) by >15% on Kunpeng 920.
- **CPU Utilization:** Significant reduction in `python` process CPU % while maintaining same throughput.
