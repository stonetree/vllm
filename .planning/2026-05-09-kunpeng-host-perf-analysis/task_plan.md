# Task Plan: 鲲鹏 Host 侧推理性能瓶颈分析

## Goal
分析 vllm 代码（vllm-0.18.0 和 vllm-ascend-0.18.0），识别在 **鲲鹏+CPU**、**鲲鹏+GPU**、**鲲鹏+Ascend NPU** 三种硬件组合下，在线推理场景的 **HOST侧性能瓶颈**，并给出优化方法。

## Scope
- 聚焦 HOST 侧（CPU 端）的性能瓶颈，不深入 device 侧
- 涉及模型加载、KV cache 管理、调度、内存管理、数据传输、tokenizer 等
- 覆盖三种硬件组合：鲲鹏+CPU、鲲鹏+GPU (NVIDIA)、鲲鹏+Ascend NPU

## Phases

### Phase 1: 代码库结构理解 ✅
- 理解 vllm-0.18.0 整体架构（V1 engine、scheduler、worker、platform）
- 理解 vllm-ascend-0.18.0 的改动（ascend 适配层、CPU binding、NPU model runner）
- 识别与 HOST 性能相关的核心模块
- **产出**: 完整目录树 + 模块职责分类

### Phase 2: 关键路径分析 ✅
- 在线推理请求处理全链路分析（API Server → AsyncLLM → EngineCore → Worker）
- 识别 CPU 热点路径（调度、内存管理、KV cache、tokenizer）
- 对比 vanilla vllm vs ascend 版本差异
- **产出**: 热路径算子时间占比模型

### Phase 3: 鲲鹏+CPU 场景瓶颈识别 ✅
- 纯 CPU 推理的 HOST 侧瓶颈
- 内存带宽、NUMA、向量化等
- **产出**: 8 个瓶颈 + 优化方案

### Phase 4: 鲲鹏+GPU 场景瓶颈识别 ✅
- GPU 推理的 HOST 侧瓶颈
- PCIe 传输、host-device 同步、调度延迟等
- **产出**: 9 个瓶颈 + 优化方案

### Phase 5: 优化方案输出 ✅
- 针对每种场景给出具体优化方法
- 优先级排序和预期收益评估
- **产出**: 14 个优化方案 + 优先级矩阵 + 路线图

### Phase 6: 专家评估与修正 ✅
- 逐瓶颈评估正确性、难度、收益、优先级
- 发现并修正 3 处事实错误（FreeKVCacheBlockQueue 实现、Scheduler 根因、SVE 归属）
- 补充 3 个遗漏瓶颈（PyTorch ARM 后端、内存带宽、弱内存排序）
- **产出**: evaluation_report.md

### Phase 7: C++ 扩展深入分析 ✅
- 发现 vLLM 已有 19 个 CPU C++ 扩展 + 6 种 ISA 向量抽象
- ARM NEON attention kernel 已实现（401行）
- ACL (ARM Compute Library) 已集成作为 oneDNN 后端
- RoPE/Activation C++ 算子存在但被绕过
- **关键发现**: ARM 编译使用 NEON (128-bit)，未启用 SVE (256-bit)
- **产出**: final_evaluation_report.md（已被 Phase 8 替换）

### Phase 8: 场景化修正 ✅ 🔑
- **关键修正**: 验证 CMakeLists.txt 构建系统 — `csrc/cpu/` 仅在 `VLLM_TARGET_DEVICE=cpu` 时编译
- GPU/Ascend 场景下 CPU 不执行模型计算 → SVE 向量化优化不适用
- 区分三种场景下 CPU 的实际职责：
  - 鲲鹏+CPU: CPU 执行全部模型计算 → SVE 有效
  - 鲲鹏+GPU: CPU 是"调度员+搬运工" → 算法/数据传输优化
  - 鲲鹏+Ascend NPU: CPU 是"调度员+搬运工" → 算法/N2D 传输优化
- **产出**: corrected_evaluation_report.md（最终权威版本）

## Decisions Made

| 决策 | 时间 | 理由 |
|------|------|------|
| 将 GPU 进一步区分为 NVIDIA GPU 和 Ascend NPU | Phase 8 | Ascend 有独立的构建系统和 CPU binding 优化 |
| SVE 优化标记为 CPU-only 场景专属 | Phase 8 | CMakeLists.txt:108-115 证伪了 GPU 场景下 `csrc/cpu/` 的编译 |
| Scheduler C 扩展化降级为 P3 | Phase 6 | 算法优化 ROI 远高于语言迁移；FreeKVCacheBlockQueue 已做侵入式链表优化 |
| NUMA 内存绑定为所有场景最高优先级 | Phase 6/8 | CPU 推理是 memory-bound；GPU/NPU 场景下 pinned memory 的 NUMA 位置影响 DMA 效率 |
| corrected_evaluation_report.md 作为最终权威版本 | Phase 8 | 修正了前版 SVE 适用范围错误，增加了构建系统证据 |

## Errors Encountered

| 错误 | 发现阶段 | 修正 |
|------|---------|------|
| 称 FreeKVCacheBlockQueue 使用 Python deque | Phase 6 | 实际是侵入式链表 (kv_cache_utils.py:158-229)，O(1) |
| 将 Scheduler 瓶颈归因于 "Python 慢" | Phase 6 | 真正根因是抢占回退 O(R²) 算法 (scheduler.py:452-496) |
| 将 ARM SVE 向量化列为 vLLM 优化任务 | Phase 6 | SVE 属于 PyTorch/oneDNN 生态，非 vLLM 可控 |
| 将 SVE 优化同时列入 GPU/NPU 场景 | Phase 8 | `csrc/cpu/` 不在 GPU/NPU 构建中编译 (CMakeLists:108-115) |
| 未区分 NVIDIA GPU 和 Ascend NPU 的 CPU 侧职责差异 | Phase 8 | Ascend 有独立构建系统、CPU binding 模块、NPU stream 管理 |

## Final Document Index

| 文件 | 描述 | 状态 |
|------|------|------|
| `task_plan.md` | 本文件 — 整体任务规划与执行记录 | ✅ 最终版 |
| `findings.md` | 详细瓶颈发现（17个）+ 全链路分析 + 代码位置 | ✅ 最终版 |
| `progress.md` | 完整会话日志 + 产出物清单 | ✅ 最终版 |
| `analysis_report.md` | 初版优化方案（14个方案 + 路线图） | ⚠️ 已被后续版本修正 |
| `evaluation_report.md` | 专家评估（修正3处错误 + 补充3个遗漏） | ⚠️ 部分结论已被 Phase 8 修正 |
| `final_evaluation_report.md` | C++ 扩展分析版（SVE 量化 + 启用手法） | ⚠️ SVE 适用范围未区分场景 |
| **`corrected_evaluation_report.md`** | **最终权威版本 — 场景化修正 + 构建系统证据** | ✅ **权威** |
