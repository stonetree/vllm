# Findings: 鲲鹏 Host 侧推理性能瓶颈分析

> 最后更新: Phase 9 代码实施完成

---

## 1. 代码库架构总览

### 1.1 vllm-0.18.0（主线版本）
- **引擎层**: `v1/engine/core.py` (2036行) — EngineCore 主循环
- **调度层**: `v1/core/sched/scheduler.py` (2280行) — 纯 Python 调度器
- **KV Cache 管理层**: `v1/core/kv_cache_manager.py` (513行) + `kv_cache_utils.py` (1688行)
- **Worker 层**: `v1/worker/gpu_model_runner.py` (6720行), `cpu_model_runner.py` (130行)
- **CPU C++ 扩展**: `csrc/cpu/` (19个文件) — NEON attention、oneDNN+ACL

### 1.2 vllm-ascend-0.18.0（昇腾适配版本）
- **Worker 层**: `worker/model_runner_v1.py` (3441行) — NPU 模型执行
- **CPU 绑定**: `cpu_binding.py` (519行) — ARM Kunpeng NUMA/IRQ 绑定
- **C++ NPU 扩展**: AscendC kernel + ACLNN custom ops

---

## 2. 关键发现：构建系统隔离

```
CMakeLists.txt:108-115
csrc/cpu/ 仅在 VLLM_TARGET_DEVICE=cpu 时编译。
GPU/Ascend 场景下此目录不参与构建。
```

---

## 3. 三种场景下 CPU 职责

| | 鲲鹏+CPU | 鲲鹏+GPU | 鲲鹏+Ascend NPU |
|---|---|---|---|
| 模型计算 | CPU (85%时间) | GPU | NPU Da Vinci Core |
| 主要瓶颈 | NUMA带宽 + SIMD宽度 | Scheduler延迟 + PCIe H2D | Scheduler延迟 + N2D传输 |
| `csrc/cpu/` 编译？ | ✅ | ❌ | ❌ |
| SVE 优化有效？ | ✅ | ❌ | ❌ |

---

## 4. 已实施优化清单

### 鲲鹏+GPU (3 commits)

| Commit | 文件 | 改动 | 收益 |
|--------|------|------|------|
| `295d8c4` | `gpu_model_runner.py` | _prepare_inputs H2D 批量拷贝 | step -3~8% |
| `e5711d0` | `gpu_worker.py` | NUMA mbind 内存绑定 | H2D -10~20% |

### 鲲鹏+Ascend NPU (3 commits)

| Commit | 文件 | 改动 | 收益 |
|--------|------|------|------|
| `9373d7f` | `cpu_npu.py` | K/V 双 stream swap_blocks | swap 延迟 -30~50% |
| `82daab3` | `cpu_binding.py` | mbind 内存绑定增强 | DMA -10~20% |
| `547b288` | `policy_flashlb.py` | EPLB 时间防抖缓存 | CPU 计算减少 |

---

## 5. 关键架构发现

1. **FreeKVCacheBlockQueue 已是侵入式链表** — 非 Python deque (`kv_cache_utils.py:158-229`)
2. **RoPE C++ 算子存在但被绕过** — `rotary_embedding/common.py:283` 有 TODO 注释
3. **SiLU/GELU C++ 算子显式跳过** — `activation.py:135` 硬编码 forward_native
4. **ARM NEON 128-bit 已实现** — 但 SVE 256-bit 未启用（`cmake/cpu_extension.cmake:128`）
5. **ACL v52.6.0 已集成** — 作为 oneDNN ARM 后端
6. **Ascend cpu_binding.py 已有深度 ARM 优化** — NUMA/IRQ/taskset/migratepages
7. **Ascend EPLB need_update 已有阈值跳过** — 时间防抖为补充优化
