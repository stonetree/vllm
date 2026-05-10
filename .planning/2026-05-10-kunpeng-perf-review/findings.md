# Findings - Kunpeng Host Performance Analysis Phase 2

## Current Optimization Review (First Round)
- (Verified in Phase 1)

## Peer Review Evaluation (Phase 2 Proposals)
A peer review (`phase2_review.md`) has identified significant flaws in the initial Phase 2 proposal:
1.  **Language Tax Overestimated**: vLLM already uses C/C++/Rust for its hot paths (Hashing uses `hashlib` (C), Metadata uses NumPy (C), Detokenizer uses `tokenizers` (Rust)).
2.  **Task 1 (Hashing)**: Total overhead is estimated at <30µs/req. Sinking to SVE would yield <1% end-to-end gain.
3.  **Task 2 (Metadata Prep)**: Operations are already NumPy C-level. Python overhead is <500ns.
4.  **Task 3 (IPC)**: vLLM uses ZMQ+msgpack, not gRPC. Latency is ~10-20µs, not 100-200µs.
5.  **Task 4 (Detokenizer)**: Already uses Rust backend.
6.  **Task 5 (IRQ)**: System-level, not application-level.

## Revised Strategy (Pivot)
Based on the review, the focus shifts from "Sinking Python" to "Optimizing existing C++ Kernels and System Config":
1.  **Empirical Profiling**: Use `py-spy` or `nsys` to verify the actual bottlenecks before any new implementation.
2.  **oneDNN+ACL Verification**: Check if ARM-optimized backends are correctly loaded.
3.  **`isolcpus` Integration**: Add kernel-level core isolation awareness to `cpu_binding.py`.
4.  **C++ Kernel Dispatch**: Verify if RoPE/Activation kernels are optimally dispatched on ARM.
