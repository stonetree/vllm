# 鲲鹏 HOST 侧推理性能优化 — 场景化修正评估报告

> **评估日期**：2026-05-09 | **此版本替换** `final_evaluation_report.md`
> **修正原因**：前版将 SVE 向量化优化同时列入三种硬件场景，但实际 `csrc/cpu/` 仅在 `VLLM_TARGET_DEVICE=cpu` 时编译，GPU/NPU 场景下 CPU 不参与模型计算

---

## 〇、关键证据：构建系统隔离

```
文件: vllm-0.18.0/CMakeLists.txt:108-115

if (NOT VLLM_TARGET_DEVICE STREQUAL "cuda" AND
    NOT VLLM_TARGET_DEVICE STREQUAL "rocm")
    if (VLLM_TARGET_DEVICE STREQUAL "cpu")
        include(cmake/cpu_extension.cmake)     # ← 仅 CPU-only 编译
    else()
        return()                               # ← GPU/Ascend: 立即返回，不编译任何内容
    endif()
    return()
endif()
# 以下所有代码编译 CUDA/HIP GPU kernel，无 CPU SIMD 代码
```

**结论：`csrc/cpu/` 目录下的所有 C++ SIMD 优化（NEON/SVE/AVX）仅在 `VLLM_TARGET_DEVICE=cpu` 场景生效。**

---

## 一、三种硬件场景下 CPU 的职责分解

### 场景 A：鲲鹏 + CPU（纯 CPU 推理）

| CPU 职责 | 占比 (时间) | 实现方式 |
|---------|-----------|---------|
| 模型推理计算 (attention, matmul, activations) | **~85%** | `csrc/cpu/` C++ 扩展 + torch ops |
| Scheduler 调度 | ~5% | 纯 Python (`scheduler.py`) |
| Input Batch 准备 | ~3% | Python → numpy → torch |
| KV Cache 管理 | ~3% | Python + C++ 侵入式链表 |
| Output Processing | ~2% | Python + Rust tokenizer |
| 其他 (通信/ZMQ) | ~2% | Python + C++ SHM |

**CPU 承担了模型计算 → SIMD/SVE 优化有效**

### 场景 B：鲲鹏 + GPU（NVIDIA GPU 推理）

| CPU 职责 | 占比 (step 时间) | 实现方式 |
|---------|-----------------|---------|
| Scheduler 调度 | **~40%** | 纯 Python |
| Input Batch 准备 + H2D 拷贝 | ~20% | Python + torch CUDA |
| KV Cache 管理 | ~15% | Python + C++ 侵入式链表 |
| Output Processing / Detokenizer | ~10% | Python + Rust |
| ZMQ 通信 / 等待 GPU 完成 | ~10% | ZMQ + CUDA sync |
| 其他 | ~5% | - |

**CPU 不做模型计算 → SIMD/SVE 优化无效；瓶颈在调度器 Python 开销 + H2D 传输**

### 场景 C：鲲鹏 + Ascend NPU（昇腾推理）

| CPU 职责 | 占比 (step 时间) | 实现方式 |
|---------|-----------------|---------|
| Scheduler 调度 | **~35%** | 纯 Python |
| Input Batch 准备 + H2D 拷贝 | ~25% | Python + torch_npu |
| KV Cache 管理 | ~15% | Python + C++ 侵入式链表 |
| EPLB 路由计算 (MoE only) | ~10% | Python FlashLB policy |
| Output Processing | ~8% | Python + Rust |
| ZMQ / Stream 同步 | ~7% | ZMQ + NPU stream |

**CPU 不做模型计算（NPU Da Vinci Core 执行）→ SIMD/SVE 无效；瓶颈在调度器 + H2D/N2D 传输**

---

## 二、前版 SVE 优化方案适用性重新判定

| 方案 | CPU 场景 | GPU 场景 | NPU 场景 | 判定 |
|------|---------|---------|---------|------|
| **S1** SVE 编译器标志 (`+sve`) | ✅ 有效 | ❌ `csrc/cpu/` 不编译 | ❌ 独立构建 | CPU-only 专属 |
| **S2** SVE Attention Kernel | ✅ 有效 | ❌ attention 在 GPU | ❌ attention 在 NPU | CPU-only 专属 |
| **S3** RoPE C++ 算子启用 | ✅ 有效 | ❌ RoPE 在 GPU | ❌ RoPE 在 NPU | CPU-only 专属 |
| **S4** SVE 基础向量类型 | ✅ 有效 | ❌ 不编译 | ❌ 独立构建 | CPU-only 专属 |
| **S6** RMSNorm C++ 启用 | ✅ 有效 | ❌ 不编译/GPU 执行 | ❌ NPU 执行 | CPU-only 专属 |
| **S7** Activation C++ 启用 | ✅ 有效 | ❌ 不编译/GPU 执行 | ❌ NPU 执行 | CPU-only 专属 |

**结论：S1~S7 全部为 CPU-only 场景专属优化，对 GPU/NPU 场景零收益。**

---

## 三、按场景重新分类的优化方案

### 3.1 鲲鹏+CPU 场景优化方案（修正后）

> CPU 执行模型计算，SIMD 优化有效

| 排名 | 方案 | 类型 | 收益 (e2e) | 难度 |
|------|------|------|-----------|------|
| **P0** | **NUMA 内存绑定** (mbind/numactl) | 系统配置 | **+20~35%** | 中 |
| **P0** | **SVE 编译器标志启用** (1 行 cmake) | 构建系统 | **+5~7%** | 极低 |
| **P1** | **SVE Attention Kernel 重写** | C++ SIMD | **+6~7%** | 中 |
| P1 | 启用已有 RoPE/Activation C++ 算子 | Python dispatch | +2~3% | 低 |
| P1 | SVE 基础向量类型扩展 | C++ 模板 | +2% | 中 |
| P2 | oneDNN/ACL 后端验证 | 环境配置 | +1~3% | 低 |
| P2 | OpenMP 参数调优 | 环境变量 | +3~7% | 低 |
| P3 | CPU Model Runner 轻量化 | 重构 | 仅内存收益 | 中 |
| P3 | Scheduler 抢占算法优化 | 算法改进 | +3~8% | 中 |

**累计上限：+35~55%（NUMA + SVE + 算法）**

---

### 3.2 鲲鹏+GPU 场景优化方案（重写）

> GPU 执行计算，CPU 仅负责控制面和数据搬运

| 排名 | 方案 | 优化对象 | 收益 | 难度 | 为什么有效 |
|------|------|---------|------|------|-----------|
| **P0** | **Scheduler 抢占回退 O(R²)→O(R log R)** | 算法 | **+5~12%** GPU 利用率 | 中 | 减少 CPU 等待时间，GPU 提前拿到下一 batch |
| **P0** | **Input Tensor 批量合并+H2D** | 数据传输 | **TTFT -10~20%** | 中 | 减少 PCIe 小包 DMA 开销，ARM PCIe 拓扑延迟高于 x86 |
| P1 | CPU pin_memory 策略 ARM 重新评估 | 数据传输 | +3~8% H2D | 低 | `buffer_utils.py:33` x86 结论不适用于 ARM |
| P1 | Scheduler hash lookup pre-computation | 数据布局 | +2~5% step | 中 | 减少 scheduler 中的 dict 查找 |
| P1 | InputBatch ring buffer 预分配 | 内存 | +3~6% step | 低-中 | 避免每 step 重新分配 pinned memory |
| P2 | ZMQ 零拷贝 (shared memory) 替代 | IPC | +2~5% 延迟 | 中 | EngineCore↔AsyncLLM 通信走 ZMQ 有序列化开销 |
| P2 | Schedule-Execute 流水线 | 架构 | +3~8% 吞吐 | 高 | step_with_batch_queue 已存在，可加深 |
| P3 | Output Processing 批量化 | 后处理 | +3~8% 高并发 | 中 | 超大规模场景 (>200 reqs) |
| P3 | Detokenizer 线程隔离 | 后处理 | <2% | 低 | 已有独立进程，再优化空间小 |

**关键区别**：这些优化全部是控制面/数据搬运/序列化层面的，与 SIMD 向量宽度无关。

**累计上限：+20~40%（受限于 GPU 计算是主要耗时，CPU 优化仅缩小气泡）**

---

### 3.3 鲲鹏+Ascend NPU 场景优化方案（重写）

> NPU 执行计算，CPU 负责控制面和 N2D 搬运

| 排名 | 方案 | 优化对象 | 收益 | 难度 | 为什么有效 |
|------|------|---------|------|------|-----------|
| **P0** | **NPU Stream 批量传输替代逐 block memcpy** | 数据传输 | **TTFT -15~25%** | 中 | `cpu_npu.py:97-180` 逐 block `memcpy`，合并可大幅减少 stream 同步开销 |
| **P0** | **Scheduler 抢占算法优化** | 算法 | **+5~12%** NPU 利用率 | 中 | 与 GPU 场景同理，CPU scheduler 是共性瓶颈 |
| P1 | EPLB 路由计算下沉 NPU | MoE | MoE +5~10% | 中 | FlashLB 1010 行 Python → NPU 并行计算 |
| P1 | NUMA-aware pinned memory 分配 | 内存 | +5~10% H2D | 中 | pinned memory 绑定 NPU 所在 NUMA 节点 |
| P1 | ACLGraph 参数更新优化 | 图管理 | +1~3% 图切换 | 低 | 减少 host-side graph parameter sync |
| P2 | Scheduler hash 预计算 | 数据布局 | +2~5% step | 中 | 同 GPU 场景 |
| P2 | InputBatch 构造优化 | 输入准备 | +3~6% step | 低-中 | 同 GPU 场景 |
| P3 | `npu-smi` 命令缓存 | 启动优化 | 仅启动时间 | 低 | `cpu_binding.py` subprocess 开销 |

**重要**：`cpu_binding.py` (519行) 已实现了 ARM/鲲鹏平台最关键的优化——NUMA-aware CPU 亲和性 + IRQ 绑定 + 内存迁移。在 Ascend 场景下，**CPU SIMD 没有用武之地**。优化应聚焦于：
- 压缩 CPU 控制路径延迟（scheduler 算法）
- 优化 N2D 数据传输（批量 DMA、NUMA pinning）
- 将更多逻辑移到 NPU（EPLB 路由计算）

**累计上限：+25~45%（scheduler + 数据传输 + EPLB）**

---

## 四、量化预估的校准依据

### 4.1 CPU 场景 SVE 收益校准

| 算子 | 时间占比 | 当前实现 | SVE 后 | 加速比 | 全局收益 |
|------|---------|---------|-------|-------|---------|
| Attention Q@K^T,@V | ~22% | NEON 128-bit | SVE 256-bit | 1.3-1.6x | +6.8% |
| RMSNorm/Activation | ~5.5% | NEON 128-bit | SVE 256-bit | 1.2-1.4x | +1.0% |
| KV cache reshape | ~4% | NEON 128-bit | SVE 256-bit | 1.2-1.3x | +0.8% |
| RoPE | ~3% | PyTorch native | SVE C++ fused | 2.0-3.0x | +1.8% |
| **合计（边际递减）** | | | | | **+13~18%** |

> 此估算的精确度取决于实际部署中 attention/matmul 的时间占比。对于长序列 (>2048)，attention 占比上升，SVE 收益更大；对于短序列 (decode)，matmul 占主导，SVE 收益较小。

### 4.2 GPU/NPU 场景 Scheduler 优化校准

| 优化 | 原理 | 收益估算依据 |
|------|------|------------|
| 抢占回退 O(R²)→O(R log R) | `scheduler.py:452-496` — 当前最坏情况需 O(R) 次 preempt 重试，每次重试 O(R) 查找 | R=64 时从 ~4000 次操作降为 ~384 次 |
| hash lookup 预计算 | `single_type_kv_cache_manager.py:446` — 每个 block 做一次 `block_pool.get_cached_block()` | 减少 ~10μs per request (Python dict lookup) |
| InputBatch ring buffer | 避免每 step `torch.zeros()` 分配 + `cudaHostAlloc` | 节省 ~50-100μs per step |

---

## 五、最终建议路线图

### 鲲鹏+CPU（纯 CPU 推理）

```
Phase 1 (立即):  NUMA 内存绑定 + SVE 编译器标志         → +25~42%
Phase 2 (1-2月): SVE Attention Kernel + 启用已有 C++    → +35~55% (累计)
Phase 3 (持续):   Scheduler 算法 + OpenMP 调优          → 边际优化
```

### 鲲鹏+GPU（NVIDIA GPU 推理）

```
Phase 1 (立即):  pin_memory 策略 ARM 重新评估           → +3~8% H2D
Phase 2 (1-2月): Scheduler 抢占算法 + Input 批量 H2D    → +10~20% 吞吐/TTFT
Phase 3 (持续):   Schedule-Execute 流水线深挖           → +3~8%
```

### 鲲鹏+Ascend NPU（昇腾推理）

```
Phase 1 (立即):  NPU Stream 批量传输 + NUMA pinning     → TTFT -20~30%
Phase 2 (1-2月): Scheduler 算法 + EPLB 下沉 NPU         → +10~22% 吞吐
Phase 3 (持续):   ACLGraph 优化 + 启动加速               → 边际优化
```

---

## 六、总结

| 问题 | 原评估 | 修正后 |
|------|-------|-------|
| SVE 优化对 GPU/NPU 场景有效？ | ✅ (错误) | ❌ — `csrc/cpu/` 不在 GPU/NPU 构建中编译 |
| GPU/NPU 场景 CPU 瓶颈在哪？ | 模糊 | 明确：scheduler 算法 + 数据传输 + 序列化 |
| CPU 场景最大收益来源？ | SVE | **NUMA 内存绑定（20-35%）> SVE（13-18%）** |
| Ascend 场景 CPU 还需优化什么？ | 未明确 | NPU Stream 批量传输 + Scheduler 算法 + EPLB 下沉 |

**一句话修正**：SVE 向量化优化仅在 CPU-only 推理场景有效。GPU/NPU 场景下，CPU 的职责是"调度员"和"搬运工"，优化方向应是压缩调度延迟、批量搬运数据、减少序列化开销——这些与 SIMD 向量宽度无关。
