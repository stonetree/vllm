# Progress Log

## Session: 2026-05-09 — 鲲鹏 HOST 侧推理性能瓶颈分析

### 执行概要

| 阶段 | 状态 | 关键产出 |
|------|------|---------|
| Phase 1: 代码库结构理解 | ✅ | 完整架构图、模块分类 |
| Phase 2: 关键路径分析 | ✅ | 热路径算子时间占比模型 |
| Phase 3: 鲲鹏+CPU 瓶颈 | ✅ | 8 个瓶颈 |
| Phase 4: 鲲鹏+GPU 瓶颈 | ✅ | 9 个瓶颈 |
| Phase 5: 优化方案 | ✅ | 14 个方案 + 路线图 |
| Phase 6: 专家评估修正 | ✅ | 修正3处错误 + 补充3个遗漏 |
| Phase 7: C++ 扩展分析 | ✅ | 量化 SVE 收益 |
| Phase 8: 场景化修正 🔑 | ✅ | **构建系统证据 + 场景分离** |

### 关键修正历程

1. **assessment 1** (Phase 6): 发现 FreeKVCacheBlockQueue 已用侵入式链表、Scheduler 根因是算法而非语言、SVE 归属 PyTorch 生态
2. **assessment 2** (Phase 7): 发现 vLLM 已有 19 个 CPU C++ 扩展、ARM NEON attention 已实现、RoPE C++ 算子被绕过、SVE 编译标志缺失
3. **assessment 3** (Phase 8): 发现 `csrc/cpu/` 仅在 CPU-only 构建中编译（CMakeLists:108-115），SVE 优化不适用于 GPU/NPU 场景

### 最终产出文件清单

| 文件 | 描述 | 状态 |
|------|------|------|
| `task_plan.md` | 任务规划、阶段追踪、决策记录、错误日志 | ✅ 最终版 |
| `findings.md` | 全量瓶颈发现（按场景分类）、C++ 扩展发现、遗漏补充 | ✅ 最终版 |
| `progress.md` | 本文件 — 会话日志 | ✅ 最终版 |
| `analysis_report.md` | 初版优化方案 | ⚠️ 历史版本 |
| `evaluation_report.md` | 第一次专家评估 | ⚠️ 部分修正 |
| `final_evaluation_report.md` | C++ 扩展 + SVE 量化版 | ⚠️ 场景适用性未区分 |
| `corrected_evaluation_report.md` | **最终权威版本** — 构建系统证据 + 三种场景分离 | ✅ 权威 |

### 核心结论

- **鲲鹏+CPU**: 最大收益来自 NUMA 内存绑定 (+20~35%) + SVE 向量化 (+13~18%)
- **鲲鹏+GPU**: 优化方向是 Scheduler 算法 + H2D 批量传输，与 SIMD 无关
- **鲲鹏+Ascend NPU**: 优化方向是 NPU Stream 批量 + Scheduler + EPLB 下沉，`cpu_binding.py` 已做深度 ARM 优化
