# Findings: 鲲鹏 Host 侧推理性能瓶颈分析

> 最后更新: Phase 8 场景化修正

---

## 1. 代码库架构总览

### 1.1 vllm-0.18.0（主线版本）
- **引擎层**: `v1/engine/core.py` (2036行) — EngineCore 主循环，调度 → 执行 → 输出
- **调度层**: `v1/core/sched/scheduler.py` (2280行) — 纯 Python 调度器
- **KV Cache 管理层**: `v1/core/kv_cache_manager.py` (513行) + `kv_cache_utils.py` (1688行)
- **Worker 层**: `v1/worker/gpu_model_runner.py` (6720行) — 输入准备、模型执行
- **CPU 适配**: `v1/worker/cpu_worker.py` (237行), `cpu_model_runner.py` (130行)
- **平台层**: `platforms/cpu.py` (506行) — CPU 架构检测、NUMA、SIMD kernel 导入
- **C++ CPU 扩展**: `csrc/cpu/` (19个文件) — NEON/ASIMD attention、RMSNorm、RoPE、oneDNN+ACL

### 1.2 vllm-ascend-0.18.0（昇腾适配版本）
- **Worker 层**: `worker/model_runner_v1.py` (3441行) — NPU 模型执行
- **平台层**: `platform.py` (905行) — NPU 平台集成
- **CPU 绑定**: `cpu_binding.py` (519行) — ARM Kunpeng NUMA/IRQ 绑定（**已有深度优化**）
- **C++ NPU 扩展**: `csrc/` (AscendC kernel + ACLNN custom ops) — 运行于 Da Vinci NPU core
- **补丁层**: `patch/` 目录 (29+ 补丁) — 上游 vllm 兼容性修复

---

## 2. 🔑 关键发现：构建系统隔离

```
文件: vllm-0.18.0/CMakeLists.txt:108-115

csrc/cpu/ 目录仅在 VLLM_TARGET_DEVICE=cpu 时编译。
GPU (cuda/rocm) 和 Ascend (独立构建系统) 场景下，
此目录完全未参与编译 → CPU 不执行模型计算。
```

**影响**: SVE/NEON 等 CPU SIMD 优化仅在鲲鹏+CPU 场景有效。

---

## 3. 三种场景下 CPU 职责分解

| | 鲲鹏+CPU | 鲲鹏+GPU | 鲲鹏+Ascend NPU |
|---|---|---|---|
| 模型计算 | CPU (85%时间) | GPU | NPU Da Vinci Core |
| 主要瓶颈 | NUMA带宽 + SIMD宽度 | Scheduler延迟 + PCIe H2D | Scheduler延迟 + N2D传输 |
| `csrc/cpu/` 编译？ | ✅ | ❌ | ❌ |
| SVE 优化有效？ | ✅ (13-18%) | ❌ | ❌ |
| NUMA 内存绑定？ | P0 (20-35%) | P1 (pinned memory) | P1 (pinned memory) |
| Scheduler 算法？ | P3 (3-8%) | P0 (5-12%) | P0 (5-12%) |

---

## 4. 已确认瓶颈与优化（逐场景）

### 鲲鹏+CPU

| ID | 瓶颈 | 优先级 | 优化方向 | 预期收益 |
|----|------|--------|---------|---------|
| C1 | NUMA 内存未绑定 | P0 | mbind/numactl --membind | +20~35% |
| C2 | SVE 编译标志未启用 | P0 | cmake: +sve (1行) | +5~7% |
| C3 | Attention 用 NEON 128-bit | P1 | SVE Attention kernel 重写 | +6~7% |
| C4 | RoPE C++ 算子存在但绕过 | P1 | 启用已有 binding | +1.8% |
| C5 | OpenMP 无 ARM 调优 | P2 | OMP_WAIT_POLICY=active | +3~7% |
| C6 | Activation C++ 显式跳过 | P2 | 移除 forward_native 硬编码 | +0.4% |
| C7 | CPU Model Runner 继承 GPU | P3 | 重构解耦 | 内存节省 |
| C8 | Inductor ARM 代码生成 | P3 | 参数调优 | +2~5% |

### 鲲鹏+GPU (NVIDIA)

| ID | 瓶颈 | 优先级 | 优化方向 | 预期收益 |
|----|------|--------|---------|---------|
| G1 | Scheduler 抢占 O(R²) 算法 | P0 | 算法改进 O(R log R) | GPU 利用率 +5~12% |
| G2 | Input tensor 逐条 H2D | P0 | 批量合并 DMA | TTFT -10~20% |
| G3 | ARM pin_memory 策略未验证 | P1 | 重新评估 ARM 行为 | H2D +3~8% |
| G4 | InputBatch 每 step 分配 | P1 | Ring buffer 预分配 | step -3~6% |
| G5 | ZMQ 序列化开销 | P2 | SHM 零拷贝替代 | 延迟 -2~5% |
| G6 | Schedule-Execute 气泡 | P2 | 加深流水线 | 吞吐 +3~8% |
| G7 | Output Processing 非批量化 | P3 | 批量 detokenize | 高并发 +3~8% |

### 鲲鹏+Ascend NPU

| ID | 瓶颈 | 优先级 | 优化方向 | 预期收益 |
|----|------|--------|---------|---------|
| A1 | NPU Stream 逐 block memcpy | P0 | 批量传输 + stream 复用 | TTFT -15~25% |
| A2 | Scheduler 抢占 O(R²) 算法 | P0 | 同 GPU 场景 | NPU 利用率 +5~12% |
| A3 | EPLB Python 路由计算 | P1 | 下沉 NPU 并行计算 | MoE +5~10% |
| A4 | NUMA-aware pinned memory | P1 | mbind NPU 所在 NUMA 节点 | H2D +5~10% |
| A5 | cpu_binding.py npu-smi subprocess | P3 | 缓存/sysfs 替代 | 启动加速 |
| A6 | ACLGraph 参数同步 | P3 | 减少 host-side sync | 图切换延迟 |

---

## 5. C++ 扩展深入发现（已确认）

### 已有 CPU C++ 算子（csrc/cpu/）

| 算子 | ARM 实现 | 当前使用状态 |
|------|---------|------------|
| `cpu_attention_with_kv_cache` | ✅ NEON (401行) | ✅ 活跃 |
| `cpu_attn_reshape_and_cache` | ✅ NEON | ✅ 活跃 |
| `rms_norm / fused_add_rms_norm` | ✅ 向量化 | ⚠️ 仅 custom_ops 启用 |
| `rotary_embedding` | ✅ 向量化 | ❌ **绕过** (TODO 注释: line 283) |
| `silu_and_mul / gelu_and_mul` | ✅ 向量化 | ❌ **显式跳过** (forward_native) |
| `onednn_mm` | ✅ ACL 后端 | ✅ 活跃 |
| `weight_packed_linear` | ❌ AVX512-only | N/A |
| `cpu_fused_moe` | ❌ AVX512-only | N/A |

### 编译器标志

```
当前: -march=armv8.2-a+bf16+dotprod+fp16     (NEON 128-bit)
可改: -march=armv8.2-a+sve+bf16+dotprod+fp16  (SVE 256-bit, 1行改动)
```

### Ascend C++ 代码

Ascend 项目的 C++ 全部是 AscendC NPU kernel（Da Vinci Core）+ 通用 C++17 宿主代码。**无 ARM NEON/SVE CPU 侧优化。** CPU binding 的 ARM 优化完全在 Python 层 (`cpu_binding.py`)。

---

## 6. 遗漏瓶颈补充

### L1: PyTorch ARM 后端成熟度不足
- oneDNN+ACL 已集成但需验证是否正确加载
- 使用 `torch.backends.mkldnn.is_available()` 确认

### L2: 内存带宽瓶颈（CPU 场景根因）
- CPU 推理是 memory-bound，鲲鹏 920 理论 150-200 GB/s
- 跨 NUMA 访问有效带宽降为 30-50%

### L3: ARM 弱内存排序模型
- 多线程同步需要更多屏障指令
- 影响 TP>1 场景的 CpuCommunicator SHM 通信
