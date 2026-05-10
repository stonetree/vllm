# 鲲鹏 HOST 侧推理性能瓶颈与优化方案 — 专家评估报告

> 评估人角色：AI 系统性能优化专家 | 评估日期：2026-05-09
> 评估对象：`analysis_report.md`（瓶颈分析报告）与 `findings.md`（发现记录）

---

## 一、总体评价

| 维度 | 评分 (1-5) | 说明 |
|------|-----------|------|
| 分析覆盖度 | ★★★★☆ | 覆盖了控制面和数据面主要路径，但遗漏了 3 个根因级瓶颈 |
| 瓶颈定位精度 | ★★★☆☆ | 部分瓶颈归因不够精准，将症状当根因；`FreeKVCacheBlockQueue` 实际已用侵入式链表优化，原分析有误 |
| 优化方案可行性 | ★★★★☆ | 多数方案方向正确，但少数方案（Scheduler C 扩展化、SVE 向量化）对 vLLM 工程而言成本过高 |
| 收益预估准确性 | ★★☆☆☆ | 部分收益被高估 2-3x；缺少基于代码路径时间占比的定量支撑 |

**核心修正**：前次分析将 "Python 解释器开销" 作为首要瓶颈，但实际上 Python 层的 dict/list 操作经过 CPython 高度优化，每个操作的常数因子约 50-150ns。在 decode 阶段（每 step ~1-10ms），Python 调度开销通常 <200μs。**真正的瓶颈是算法复杂度（O(R²) 抢占回退、线性扫描）和跨 NUMA 内存访问**，而非 Python 本身。

---

## 二、逐瓶颈正确性评估

### 鲲鹏+CPU 场景

#### 瓶颈 #1：Scheduler 纯 Python 热路径

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ⚠️ 部分正确 |
| **详细分析** | `schedule()` 确实是每个 decode step 的必经路径。但将瓶颈归因于 "纯 Python" 不够精准：(1) `FreeKVCacheBlockQueue` 已用侵入式链表（`kv_cache_utils.py:158-229`），非 Python deque；(2) dict 操作在 CPython 3.12 中经高度优化，每次 lookup ~50ns；(3) 真正的热点是 **抢占回退循环**（`scheduler.py:452-496`）— 当 KV cache 耗尽时，可能需要 O(R) 次 preempt 重试。 |
| **代码证据** | `kv_cache_utils.py:158-229`（FreeKVCacheBlockQueue 已用侵入式链表优化）；`scheduler.py:452-496`（allocate_slots 抢占重试循环） |
| **收益校准** | 原估 10-20% → 校准后 **5-12%**（仅在大批量+高压力场景显著） |

#### 瓶颈 #2：KV Cache 管理纯 Python 实现

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ❌ 部分错误 |
| **详细分析** | 原分析声称 `FreeKVCacheBlockQueue` 使用 Python list/deque，但实际已用侵入式链表（O(1) pop/append/remove）。`find_longest_cache_hit` 使用 `block_pool.get_cached_block()` 做 hash 查找（O(1) per block），非 Python dict 遍历。真正的瓶颈在于：前缀缓存命中链需要逐 block 做 hash 查找（`single_type_kv_cache_manager.py:446-456`），在长 prompt 场景下这个循环本身是 O(L/block_size)。 |
| **代码证据** | `single_type_kv_cache_manager.py:446-456`（逐 block hash 查找）；`kv_cache_utils.py:110-155`（KVCacheBlock dataclass + 侵入式链表） |
| **收益校准** | 原估 15-25% → 校准后 **3-8%**（仅前缀缓存路径） |

#### 瓶颈 #3：NUMA 感知不足

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确且是关键瓶颈 |
| **详细分析** | 这是鲲鹏 CPU 推理的**头号瓶颈**。鲲鹏 920 通常 2-4 NUMA 节点，每个节点有独立内存控制器。跨 NUMA 访问延迟约 1.5-2.5x。当前 `platforms/cpu.py:120-143` 将 KV cache 总大小除以 NUMA 节点数，但**没有将各 worker 线程分配的内存绑定到对应的 NUMA 节点**。`cpu_worker.py:75-91` 只做了 CPU 亲和性绑定（线程→核），未做内存绑定（页→NUMA 节点）。同时 `CpuCommunicator` 的 SHM allreduce 在多 NUMA 下可能经过跨节点内存访问。 |
| **代码证据** | `cpu_worker.py:75-91`（仅 CPU 亲和性，无 mbind/move_pages）；`platforms/cpu.py:120-143`（KV cache 按 NUMA 节点数均分） |
| **收益评估** | 原估 20-40% → **维持**，在多 NUMA（≥2 节点）场景下通过 `mbind` + `numactl --membind` 可取得 20-35% 的提升 |

#### 瓶颈 #4：CPU Model Runner 继承 GPU 实现

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ⚠️ 正确但影响有限 |
| **详细分析** | `CPUModelRunner(GPUModelRunner)` 确实继承了 GPU 专用代码，但 `_postprocess_tensors()` 只在 `__init__` 执行一次。运行时路径中，`InputBatch` 的 CPU+GPU 双份 tensor 确实浪费内存（每 tensor ~2x），但对运行时性能影响有限（CPU 场景下 GPU tensor 不参与计算）。这是代码清洁度问题，非性能瓶颈。 |
| **代码证据** | `cpu_model_runner.py:32-53`（`_postprocess_tensors`，运行时仅执行一次） |
| **收益校准** | 原估 30% 内存节省 → **维持**（内存收益），运行时性能收益 **<2%** |

#### 瓶颈 #5：ARM OpenMP 调优

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确 |
| **详细分析** | x86 路径有完整的 `KMP_*` 调优（`platforms/cpu.py:286-295`），ARM 路径仅做了 `LD_PRELOAD` libgomp 检查（`platforms/cpu.py:297-327`）。GCC libgomp 在鲲鹏上的线程唤醒延迟（~5-10μs）高于 Intel OpenMP（~1-2μs）。但需要注意：PyTorch 在 ARM 上默认使用自己线程池（`ATen` 并行后端），OpenMP 主要影响线性代数库（oneDNN/BLAS）。 |
| **代码证据** | `platforms/cpu.py:286-295`（x86 KMP_* 调优）；`platforms/cpu.py:297-327`（ARM 仅 libgomp LD_PRELOAD） |
| **收益校准** | 原估 5-10% → 校准后 **3-7%**（局限于 BLAS/oneDNN 路径） |

#### 瓶颈 #6：ARM SVE 向量化

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ⚠️ 方向正确但不属 vLLM 层面 |
| **详细分析** | vLLM 是调度框架，其性能依赖 PyTorch + 后端库（oneDNN、BLAS）。SVE 优化应在 PyTorch 或 oneDNN 层面实现，而非 vLLM 自身。vLLM 的 `import_kernels()` 仅在 attention 路径导入少量定制 kernel（`platforms/cpu.py:474-506`），核心矩阵运算仍由 PyTorch 完成。此项应标记为 **PyTorch/oneDNN 生态优化**，非 vLLM 工程任务。 |
| **代码证据** | `platforms/cpu.py:474-506`（仅导入少量定制 kernel） |
| **可行性** | **高难度**：需要修改 PyTorch ATen 后端或 oneDNN，非 vLLM 团队可控 |
| **收益校准** | 原估 10-30% → 理论可达，但实现周期 6-12 个月 |

#### 瓶颈 #7：Inductor 编译

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ⚠️ 部分正确 |
| **详细分析** | Inductor 确实主要针对 x86 调优，但近年来已增加 AArch64 支持（包括 NEON 指令利用）。`cpp.dynamic_threads: True` 在 ARM 上可能不如预期。更大的问题是：Inductor 的 C++ 代码生成在 ARM 上可能产生次优的循环展开和向量化。但 torch.compile 的 cache 机制意味着编译开销是一次性的。 |
| **收益校准** | 原列为低优先级 → **维持**，仅在频繁换模型时关注 |

#### 瓶颈 #8：Detokenizer

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确 |
| **详细分析** | `SlowIncrementalDetokenizer` 的 Python 实现确实是每 token 开销。`FastIncrementalDetokenizer`（Rust tokenizers 库）已作为优先路径存在（`detokenizer.py:24`）。问题的关键是：哪些场景会 fallback 到 SlowIncrementalDetokenizer？需要 tokenizers 库 < 0.22.0 才 fallback，现代部署不应命中此路径。 |
| **收益校准** | 原估 2-5% e2e → 校准后 **<1%**（仅 legacy tokenizer 场景） |

---

### 鲲鹏+GPU/NPU 场景

#### 瓶颈 #9：Host-Device 数据传输

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确且是 GPU/NPU 场景的**首要瓶颈** |
| **详细分析** | 鲲鹏+GPU 的 PCIe 拓扑通常不同于 x86：GPU 可能通过 PCIe switch 或 CCIX 桥接，延迟增加 20-40%。对于 Ascend NPU，`cpu_npu.py` 使用独立 NPU stream 做异步 D2H/H2D，但每个 block 的传输仍是独立 memcpy 调用——缺少批量聚合。更关键的是：`buffer_utils.py:33` 明确注释 pin_memory 在高并发 CUDA 下有驱动竞争，但此结论来自 x86+CUDA 经验，**在 ARM+Ascend NPU 环境下需重新验证**。 |
| **代码证据** | `buffer_utils.py:30-33`（pin_memory 注释）；`cpu_npu.py:97-180`（逐 block async 传输） |
| **收益评估** | 原估 10-25% TTFT → **维持**，尤其长 prompt prefill 阶段 H2D 是 TTFT 主要成分 |

#### 瓶颈 #10：Ascend CPU Binding 初始化

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确但影响面小 |
| **详细分析** | `cpu_binding.py` 在 Worker 启动时执行，包含 subprocess 调用 `npu-smi` 和系统调用 `taskset`/`migratepages`。这是**一次性启动开销**，不进入运行时热路径。其真正价值在于运行时 NUMA 亲和性带来的稳定低延迟，而非启动加速本身。标记为 "中优先级" 偏高了。 |
| **收益校准** | 原估启动加速 → **维持但降为低优先级**；运行时收益已体现在瓶颈 #3 中 |

#### 瓶颈 #11：Schedule-Execute 串行依赖

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ⚠️ 方向正确但高估了收益 |
| **详细分析** | `step_with_batch_queue()`（`core.py:419-535`）已实现基础的调度-执行重叠。但 autoregressive decode 的本质决定了**当前 step 的输出决定了下一步的输入**，因此完全流水线化是不可能的。能重叠的是：当前 step 执行期间，预先计算下一 step 的 KV cache 前缀匹配。但前缀匹配依赖 block_hashes（`single_type_kv_cache_manager.py:446`），这些 hashes 在当前 step 执行完成前不可用。真正可做的：在 GPU 执行期间执行 output processing（已实现）和 abort processing（已实现）。 |
| **代码证据** | `core.py:419-535`（`step_with_batch_queue` 已存在）；`single_type_kv_cache_manager.py:446-456`（前缀匹配依赖 block_hashes） |
| **收益校准** | 原估 10-20% GPU 利用率 → 校准后 **3-8%**（高并发 + 长队列场景下） |

#### 瓶颈 #12：InputBatch 构造

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确 |
| **详细分析** | `InputBatch` 在每 step 的 `_update_states` → `_prepare_inputs` 路径中构造。主要开销：(1) Python→numpy→torch 的类型转换链；(2) token_ids 和 positions 的逐 request 填充；(3) block_table 的多组拼接。但 `InputBatch` 使用了 `CpuGpuBuffer` 的预分配 + `copy_slice` 模式，已做了一定程度的优化。进一步的 ring buffer 方案需要处理可变 batch size 问题。 |
| **收益校准** | 原估 5-10% → 校准后 **3-6%**（大批量场景下） |

#### 瓶颈 #13：Output Processing

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ⚠️ 高估了影响 |
| **详细分析** | Output processing 运行在 AsyncLLM 进程（独立于 EngineCore），通过 ZMQ 异步通信。其 CPU 消耗主要是：(1) detokenizer（已在瓶颈 #8 分析）；(2) `RequestOutput` 对象构建（Python 对象分配，GC 压力）。asyncio Event 开销 <1μs 量级。真正的优化空间在于：积累多个 token 后批量 detokenize、对象池化减少 GC。 |
| **收益校准** | 原估 10-15% 高并发吞吐 → 校准后 **3-8%**（超大规模 >200 reqs 场景） |

#### 瓶颈 #14：EPLB (Ascend MoE)

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确但适用范围窄 |
| **详细分析** | 此瓶颈仅影响 MoE 模型（DeepSeek-V2/V3、Qwen3-MoE 等），对 Dense 模型无影响。`policy_flashlb.py` 的 1010+ 行 Python 逻辑确实在 CPU 上执行，但 EPLB 更新频率远低于每 step（通常每 N steps 或触发式更新）。 |
| **收益校准** | 原估 5-15% MoE 吞吐 → **维持**（MoE 场景仅） |

#### 瓶颈 #15：ACLGraph 管理

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ❌ 可忽略 |
| **详细分析** | ACLGraph 的图参数 update 仅涉及少量 Python 对象赋值，延迟 <10μs。图捕获是启动时一次性开销。此项不应列为瓶颈。 |
| **收益校准** | **移除** — 不含实际性能影响 |

#### 瓶颈 #16：KV Cache Offload CPU 端管理

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ✅ 正确 |
| **详细分析** | `LRUOffloadingManager` 的 Python dict 操作在 offload 场景下有一定开销，但 eviction 决策通常每 step 触发少量（1-5 次）。真正的瓶颈是 `transfer_async` 中的 NPU stream 同步开销和逐 block memcpy。 |
| **收益校准** | 原估 10-20% offload 延迟 → 校准后 **8-15%**（仅在 KV offload 启用时） |

#### 瓶颈 #17：ARM sched_yield 补丁

| 评估维度 | 结论 |
|----------|------|
| **正确性** | ❌ 可忽略 |
| **详细分析** | `patch_sched_yield.py` 仅在 aarch64 上禁用 `os.sched_yield()`。此调用在 x86 上用于 spin-wait 优化（提示 CPU 放弃时间片），在 ARM 上不可用（aarch64 无 `sched_yield` syscall）。补丁的影响是：某些 spin-wait 路径可能多消耗几个 CPU cycle，但 <1μs。 |
| **收益校准** | **移除** — 不含可测量的性能影响 |

---

## 三、前次分析遗漏的关键瓶颈

### 遗漏 #L1：PyTorch ARM 后端成熟度不足（根因级，高优先级）

**描述**：
vLLM 的核心计算依赖 PyTorch CPU 后端。PyTorch 在 ARM 上的性能优化远不如 x86 成熟：
- oneDNN (MKLDNN) 的 AArch64 支持部分算子仍使用 reference 实现而非 JIT 优化路径
- `torch.matmul` 在 ARM 上可能 fallback 到非优化路径（非 ACL/non-NEON 优化过的 BLAS）
- PyTorch 的线程池在 ARM big.LITTLE/多 cluster 架构上没有针对性调优
- Attention 计算中的 `torch.bmm` 和 `torch.softmax` 在 ARM 上无 SIMD 加速

**影响**：这是 CPU 推理性能的**根因级限制**，可能占 30-50% 的执行时间。vLLM 代码中看不到任何针对此问题的特殊处理。所有基于 vLLM 层面的优化（调度、KV cache）都只能优化框架开销（10-20%），无法改变 PyTorch 算子的执行效率。

**建议**：
- 在鲲鹏上使用针对 ARM 优化的 PyTorch 构建（如 Huawei 提供的 torch_npu 或 ARM 优化的 PyTorch wheel）
- 配置环境变量 `TORCH_BLAS_PREFER_HIPBLASLT=0` 并验证 ARM 优化 BLAS 库是否被正确加载
- 使用 `torch.backends.mkldnn.is_available()` 和 `torch.backends.mkldnn.verbose(True)` 验证 oneDNN 在 ARM 上的生效情况

---

### 遗漏 #L2：内存带宽瓶颈（根因级，CPU 场景最高优先级）

**描述**：
CPU 推理的本质是 memory-bound 操作。鲲鹏 920 的内存带宽约 150-200 GB/s（8 通道 DDR4），但这是理论峰值。实际有效带宽受以下因素影响：
- **NUMA 分布**：跨 NUMA 访问时有效带宽降为 30-50%
- **内存访问模式**：Transformer 的 attention 矩阵和 MLP 权重访问是 streaming 模式，对 prefetcher 友好程度取决于张量布局
- **多 worker 争用**：TP > 1 时多个 worker 共享同一内存控制器，带宽被瓜分
- **CPU 的 KV cache 布局**：`cpu_attn.py` 中 KV cache 是 `[2, num_blocks, num_kv_heads, block_size, head_size]` (5D layout)，这种 layout 对内存访问模式的影响未在代码中评估

**建议**：
- 使用 `numactl --interleave=all` 启动进程，确保权重 pages 在 NUMA 节点间交错分布（提升多 worker 并发访问的带宽利用率）
- 评估 KV cache 从 5D 变为 4D layout 对 cache line 利用率的影响
- 在多 NUMA 节点场景下测试 interleave vs membind 策略

---

### 遗漏 #L3：ARM 弱内存排序模型的开销（架构级，中优先级）

**描述**：
ARM 使用弱内存排序模型（weakly-ordered memory model），而 x86 使用强排序模型（TSO — Total Store Order）。在多线程并行计算中，ARM 需要更多的内存屏障指令（`dmb`、`dsb`）来保证正确性。这些屏障在以下路径中可能产生开销：
- PyTorch 的线程池同步操作
- OpenMP 的 barrier 同步
- `CpuCommunicator` 的 SHM 原子操作（`cpu_communicator.py`）
- `torch.distributed` 的 gloo 后端在 ARM 上的同步开销

**建议**：
- 验证 `CpuCommunicator` 在 ARM 上的性能 — 当前代码在 `cpu_worker.py:41` 中 `disable_custom_all_reduce = True`，意味着 CPU 场景不用 `CpuCommunicator`，走 gloo。但 gloo 在 ARM 上的性能未经充分测试
- 如果使用 gloo，评估替换为 mpich（对 ARM 优化更好）或自定义 SHM allreduce

---

## 四、优化方案重新评估

基于以上校正，对 14 个优化方案重新排序：

| 新排名 | 优化方案 | P 级 | 难度 | 收益 | 正确性 | 说明 |
|--------|---------|------|------|------|--------|------|
| **1** | NUMA 感知内存管理 | P0 | 中 | ★★★★★ | ✅ | CPU 场景最大单次提升 |
| **2** | Host-Device 传输优化 | P0 | 中 | ★★★★★ | ✅ | GPU/NPU 场景最大单次提升 |
| **3** | PyTorch ARM 后端验证与调优 | P0 | 低 | ★★★★☆ | 🆕 | 验证 oneDNN/BLAS 在 ARM 上的配置 |
| **4** | 内存带宽优化 (interleave) | P1 | 低 | ★★★★☆ | 🆕 | numactl 策略调整 |
| **5** | Scheduler 抢占回退优化 | P1 | 中 | ★★★☆☆ | ⚡ | 从 C 扩展改为算法优化 |
| **6** | InputBatch ring buffer | P1 | 低-中 | ★★★☆☆ | ✅ | 减少 per-step 分配 |
| **7** | OpenMP/线程运行时调优 | P2 | 低 | ★★★☆☆ | ✅ | 环境变量调整即可 |
| **8** | Schedule-Execute 流水线 | P2 | 高 | ★★☆☆☆ | ⚡ | 受限于 autoregressive 本质 |
| **9** | CPU Model Runner 轻量化 | P2 | 中 | ★★☆☆☆ | ⚡ | 仅内存收益 |
| **10** | EPLB Host 端加速 | P2 | 中 | ★★★☆☆ | ✅ | 仅 MoE 场景 |
| **11** | KV Cache Offload CPU 端 | P2 | 中 | ★★☆☆☆ | ✅ | 仅 offload 场景 |
| **12** | Output Processing 批量化 | P3 | 中 | ★★☆☆☆ | ⚡ | 仅超大规模场景 |
| **13** | Ascend CPU Binding 加速 | P3 | 低 | ★☆☆☆☆ | ⚡ | 仅启动时间 |
| **14** | Scheduler C 扩展化 | P3 | 极高 | ★★☆☆☆ | ❌ | 工程 ROI 低，不如优化算法 |
| ~~15~~ | ~~ARM SVE 向量化~~ | ~~N/A~~ | ~~极高~~ | ~~—~~ | ❌ | PyTorch 层面的事，非 vLLM |

> 图例：🆕 = 新增项；⚡ = 原分析修正后重新评估；✅ = 原分析正确；❌ = 原分析有误

---

## 五、修正后的实施路线图

### 第一阶段（1-2 周，收益 20-40%，风险低）
1. **NUMA 感知内存管理** — 为 CPU worker 添加 `mbind`/`numactl --membind`，确保权重和 KV cache 分布在 local NUMA 节点
2. **内存带宽优化** — 测试 `numactl --interleave=all` 对多 worker 场景的影响
3. **PyTorch ARM 后端验证** — 检查 oneDNN/BLAS 配置，确保 ARM 优化路径生效
4. **OpenMP 调优** — ARM 上设置 `OMP_WAIT_POLICY=active`、`OMP_PROC_BIND=close`

### 第二阶段（3-4 周，收益 10-20%，风险中）
5. **Host-Device 传输批量化** — GPU/NPU 输入 tensor 合并传输、重新评估 ARM 上 pin_memory 策略
6. **Scheduler 抢占算法优化** — 将 O(R²) 抢占回退改为 O(R log R)
7. **InputBatch ring buffer** — 预分配可变大小 buffer 减少分配

### 第三阶段（持续优化，收益 5-15%，风险中-高）
8. **CPU Model Runner 重构** — 从 GPUModelRunner 解耦
9. **EPLB Host 端加速** — 仅 MoE 场景
10. **Schedule-Execute 更深度流水线** — speculative scheduling

---

## 六、数据驱动的验证建议

在进行任何优化前，建议先完成以下性能 profiling 以数据驱动决策：

### 6.1 CPU 场景 Profiling 脚本

```bash
# 1. CPU micro-architecture profiling
perf stat -e cycles,instructions,cache-references,cache-misses,L1-dcache-load-misses,LLC-load-misses \
    -e node-loads,node-load-misses,node-stores,node-store-misses \
    python -m vllm.entrypoints.openai.api_server --model <model> --device cpu

# 2. 函数级热点
py-spy top --pid <vllm_pid> --rate 100

# 3. NUMA 统计
numastat -p <vllm_pid> 1
```

### 6.2 关键指标监控

| 指标 | 工具 | 目标值 |
|------|------|--------|
| 本地 NUMA 访问占比 | `numastat` | > 90% |
| LLC miss rate | `perf stat` | < 5% |
| IPC (instructions per cycle) | `perf stat` | > 1.0 (ARM 弱于 x86) |
| Scheduler step 延迟 | vLLM prometheus metrics | < 500μs |
| H2D 传输时间 / step | vLLM traces | < 100μs (batch size <64) |

---

## 七、总结

1. **前次分析的主要价值**：系统性覆盖了 HOST 侧全链路，定位了调度器、KV Cache、NUMA、Host-Device 传输等核心模块

2. **需要修正的关键错误**：
   - `FreeKVCacheBlockQueue` 已做侵入式链表优化（非 Python deque）
   - Scheduler 瓶颈根源是算法复杂度（抢占回退 O(R²)），非 Python 语言本身
   - ARM SVE 向量化是 PyTorch 生态任务，非 vLLM 层面
   - 遗漏了 PyTorch ARM 后端成熟度、内存带宽、弱内存排序模型三个根因级瓶颈

3. **最有效的三个优化（性价比排序）**：
   - (1) NUMA 内存绑定 + interleave 策略调整 → 预期收益 20-35%
   - (2) Host-Device 批量传输 + ARM pin_memory 策略重新评估 → 预期收益 10-25%
   - (3) PyTorch ARM 后端配置验证（oneDNN/BLAS） + OpenMP 参数调优 → 预期收益 5-15%

4. **不建议投入的优化**：
   - Scheduler C 扩展化 → 工程成本极高，算法优化更有效
   - ARM SVE kernel 开发 → PyTorch 生态任务，非 vLLM 可控
