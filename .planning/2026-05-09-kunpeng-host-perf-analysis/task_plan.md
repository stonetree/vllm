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

### Phase 2: 关键路径分析 ✅
- 在线推理请求处理全链路分析（API Server → AsyncLLM → EngineCore → Worker）
- 识别 CPU 热点路径（调度、内存管理、KV cache、tokenizer）
- 对比 vanilla vllm vs ascend 版本差异

### Phase 3: 鲲鹏+CPU 场景瓶颈识别 ✅
- 纯 CPU 推理的 HOST 侧瓶颈
- 内存带宽、NUMA、向量化等

### Phase 4: 鲲鹏+GPU 场景瓶颈识别 ✅
- GPU 推理的 HOST 侧瓶颈
- PCIe 传输、host-device 同步、调度延迟等

### Phase 5: 优化方案输出 ✅
- 针对每种场景给出具体优化方法
- 优先级排序和预期收益评估

### Phase 6: 专家评估与修正 ✅
- 逐瓶颈评估正确性、难度、收益、优先级
- 发现并修正 3 处事实错误（FreeKVCacheBlockQueue 实现、Scheduler 根因、SVE 归属）
- 补充 3 个遗漏瓶颈（PyTorch ARM 后端、内存带宽、弱内存排序）

### Phase 7: C++ 扩展深入分析 ✅
- 发现 vLLM 已有 19 个 CPU C++ 扩展 + 6 种 ISA 向量抽象
- ARM NEON attention kernel 已实现（401行）
- ACL (ARM Compute Library) 已集成作为 oneDNN 后端
- **关键发现**: ARM 编译使用 NEON (128-bit)，未启用 SVE (256-bit)

### Phase 8: 场景化修正 ✅ 🔑
- **关键修正**: 验证 CMakeLists.txt 构建系统 — `csrc/cpu/` 仅在 `VLLM_TARGET_DEVICE=cpu` 时编译
- SVE 优化仅在 CPU-only 场景有效；GPU/NPU 场景 CPU 不执行模型计算
- 区分三种场景下 CPU 的实际职责

### Phase 9: 代码实施 ✅ 🔑
- 初始化 git 仓库 + 创建 `kunpeng-host-optimization` 开发分支
- 实施 5 项代码修改（3 GPU + 2 NPU），共 6 个 commits
- 详见 implementation_plan.md

## Decisions Made

| 决策 | Phase | 理由 |
|------|-------|------|
| 将 GPU 进一步区分为 NVIDIA GPU 和 Ascend NPU | 8 | Ascend 有独立的构建系统和 CPU binding 优化 |
| SVE 优化标记为 CPU-only 场景专属 | 8 | CMakeLists.txt:108-115 证伪了 GPU 场景下 `csrc/cpu/` 的编译 |
| Scheduler C 扩展化降级为 P3 | 6 | 算法优化 ROI 远高于语言迁移 |
| 剔除 Scheduler 抢占优化 | 9 | 抢占仅 KV cache 满时触发（罕见），ROI 低 |
| 剔除 ARM pin_memory 重评估 | 9 | 是实验验证而非代码修改 |
| corrected_evaluation_report.md 作为最终权威版本 | 8 | 修正了前版 SVE 适用范围错误 |

## Errors Encountered

| 错误 | Phase | 修正 |
|------|-------|------|
| 称 FreeKVCacheBlockQueue 使用 Python deque | 6 | 实际是侵入式链表 (kv_cache_utils.py:158-229)，O(1) |
| 将 Scheduler 瓶颈归因于 "Python 慢" | 6 | 真正根因是抢占回退 O(R²) 算法 (scheduler.py:452-496) |
| 将 ARM SVE 向量化列为 vLLM 优化任务 | 6 | SVE 属于 PyTorch/oneDNN 生态，非 vLLM 可控 |
| 将 SVE 优化同时列入 GPU/NPU 场景 | 8 | `csrc/cpu/` 不在 GPU/NPU 构建中编译 (CMakeLists:108-115) |
| EPLB 缓存方案原有 need_update 已存在 | 9 | 实施时调整为时间防抖补充 |

## Final Document Index

| 文件 | 描述 | 状态 |
|------|------|------|
| `task_plan.md` | 本文件 — 整体任务规划与执行记录 | ✅ |
| `findings.md` | 详细瓶颈发现 + 全链路分析 + 代码位置 | ✅ |
| `progress.md` | 完整会话日志 + 产出物清单 | ✅ |
| `analysis_report.md` | 初版优化方案（14个方案 + 路线图） | ⚠️ 历史版本 |
| `evaluation_report.md` | 专家评估（修正3处错误 + 补充3个遗漏） | ⚠️ 部分修正 |
| `final_evaluation_report.md` | C++ 扩展分析版（SVE 量化 + 启用手法） | ⚠️ 场景未区分 |
| `corrected_evaluation_report.md` | 最终权威版本 — 场景化修正 + 构建系统证据 | ✅ 权威 |
| `implementation_plan.md` | 详细代码实施方案 + 自审记录 | ✅ |
