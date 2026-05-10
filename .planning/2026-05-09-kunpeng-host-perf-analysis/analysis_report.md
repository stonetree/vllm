# 鲲鹏 HOST 侧推理性能瓶颈分析与优化方案

> 基于 vllm-0.18.0 和 vllm-ascend-0.18.0 代码分析

---

## 一、整体结论

vLLM 在鲲鹏平台上的 HOST 侧性能瓶颈集中在两个层面：
1. **控制面 (Control Plane)**：调度器、KV Cache 管理、状态更新等纯 Python 逻辑，在 ARM 单核性能较弱的鲲鹏上占比更高
2. **数据面 (Data Plane)**：Host-Device 数据传输、NUMA 拓扑利用不充分、ARM SIMD/向量化不足

Ascend 版本（vllm-ascend）已针对鲲鹏做了大量 CPU 亲和性绑定优化（`cpu_binding.py`），但调度器和 KV Cache 管理仍是共享瓶颈。

---

## 二、鲲鹏+CPU 场景优化方案

### 优化 1：Scheduler 关键路径 C 扩展化

**现状**：`scheduler.py:338-749` 的 `schedule()` 方法每 decode step 执行一次，全部为纯 Python 操作（列表遍历、dict 查找、条件判断）。

**优化方法**：
- 将 `allocate_slots` 中的 `get_num_blocks_to_allocate` 逻辑下沉到 C++ 扩展 (`csrc/`)
- 使用 `numpy` 数组替代 `dict[str, int]` 存储 `num_scheduled_tokens`
- 预分配固定大小数组避免 Python list append 触发 GC
- 利用 `ctypes`/`cffi` 调用 C 实现的快速块查找

**预期收益**：step 延迟降低 10-20%

**代码位置**：
- `v1/core/sched/scheduler.py:375-508` (running request scheduling loop)
- `v1/core/sched/scheduler.py:557-749` (waiting request scheduling loop)
- `v1/core/kv_cache_manager.py:218-388` (`allocate_slots`)

---

### 优化 2：KV Cache 块管理数据结构替换

**现状**：`KVCacheBlock` 是 Python 对象，`FreeKVCacheBlockQueue` 使用 Python list。

**优化方法**：
- 将 `FreeKVCacheBlockQueue` 替换为 C 实现的环形缓冲区或 intrusive linked list
- `block_pool` 的 `req_to_blocks` 使用 `numpy` 或 `array` 模块存储固定大小 block_id 数组
- 前缀缓存 hash 查找使用 `numpy` 的 `searchsorted` 或 C++ `std::unordered_map`
- block reference counting 使用 `numpy.uint16` 数组而非 Python int

**预期收益**：大批量（>64 reqs）场景下 KV cache 管理开销降低 15-25%

**代码位置**：
- `v1/core/kv_cache_utils.py` (KVCacheBlock, FreeKVCacheBlockQueue 定义)
- `v1/core/kv_cache_coordinator.py` (find_longest_cache_hit, allocate_new_blocks)

---

### 优化 3：NUMA 感知内存管理

**现状**：`platforms/cpu.py:120-143` 将 KV cache 内存平均分配到各 NUMA 节点，但线程和数据可能跨 NUMA 节点。

**优化方法**：
- 每个 Worker 进程绑定到单个 NUMA 节点（已在 `cpu_worker.py:75-91` 实现）
- **新增**：KV cache tensor 也绑定到对应 NUMA 节点（使用 `numactl --membind` 或 `mbind()` 系统调用）
- **新增**：使用 `move_pages()` 或 `libnuma` 确保模型权重分布在 local NUMA 节点
- 在 `CpuPlatform.check_and_update_config()` 中根据 NUMA 拓扑自动设置 `VLLM_CPU_KVCACHE_SPACE` 的 per-node 值
- ARM 平台启动时通过 `lscpu` 检测鲲鹏 TaiShan 核心拓扑，禁用跨 cluster 调度

**预期收益**：多 NUMA 场景下吞吐提升 20-40%

**代码位置**：
- `platforms/cpu.py:120-143` (`get_device_total_memory`)
- `platforms/cpu.py:351-390` (`get_allowed_cpu_core_node_list`)
- `platforms/cpu.py:393-443` (`discover_numa_topology`)
- `v1/worker/cpu_worker.py:75-91` (`_get_autobind_cpu_ids`)

---

### 优化 4：CPU Model Runner 轻量化

**现状**：`CPUModelRunner(GPUModelRunner)` 继承大量 GPU 专用逻辑。`InputBatch` 创建 CPU+GPU 双份 tensor。

**优化方法**：
- **方案 A**：让 `CPUModelRunner` 直接继承 `WorkerBase`，只包含 CPU 必需逻辑
- **方案 B（低成本）**：在 `__init__` 阶段将 `CpuGpuBuffer` 替换为纯 CPU buffer，避免运行时双重内存
- 移除 GPU 专用的 `_prepare_inputs` 中不必要的数据拷贝（positon 偏移计算等可利用 CPU 直接引用）
- `InputBatch` 增加 `is_cpu_only` 参数，跳过 GPU tensor 分配

**预期收益**：内存占用降低 30%，初始化加速

**代码位置**：
- `v1/worker/cpu_model_runner.py` (全部 130 行)
- `v1/worker/gpu_input_batch.py:81-98` (`InputBatch.__init__`)

---

### 优化 5：ARM 平台 OpenMP 调优

**现状**：ARM 平台使用 `libgomp`/`libomp`，无 x86 特有的 `KMP_*` 调优。

**优化方法**：
- 为 ARM 设置 `OMP_WAIT_POLICY=active`（避免线程频繁睡眠/唤醒）
- 设置 `OMP_DYNAMIC=false` 和 `OMP_PROC_BIND=close`
- 针对鲲鹏 GCC 工具链：使用 `-mcpu=tsv110` 或 `-mtune=neoverse-n1` 编译 PyTorch
- 测试 `libjemalloc` 替代 `libtcmalloc` — jemalloc 在 ARM 上有更好的 NUMA 感知
- 在 `platforms/cpu.py:300-327` 的 ARM 分支增加 `libomp` 优先于 `libgomp` 的逻辑（LLVM OpenMP 通常比 GCC 性能好）

**预期收益**：吞吐提升 5-10%

**代码位置**：
- `platforms/cpu.py:263-328` (环境变量配置)
- `v1/worker/cpu_worker.py:57-70` (`check_preloaded_libs`)

---

### 优化 6：ARM SVE 向量化

**现状**：`import_kernels()` 在 ARM 上直接 `import vllm._C`，无 SIMD 特化。

**优化方法**：
- 为鲲鹏编写 SVE 优化的 attention kernel（类似 x86 的 `_C_AVX512` 路径）
- 关键操作：scaled dot-product attention 的 SVE 向量化（鲲鹏 920 支持 256-bit SVE）
- 使用 PyTorch 的 `torch.backends.mkldnn`（oneDNN 已支持 AArch64）
- 利用 ARM Compute Library (ACL) 加速矩阵乘法
- 通过 `VLLM_CPU_SGL_KERNEL` 环境变量添加 ARM 路径

**预期收益**：矩阵运算性能提升 10-30%，attention 计算提升 15-25%

**代码位置**：
- `platforms/cpu.py:474-506` (`import_kernels`)
- `model_executor/kernels/linear/scaled_mm/cpu.py`

---

### 优化 7：Detokenizer 加速

**现状**：`SlowIncrementalDetokenizer` 纯 Python 实现。

**优化方法**：
- 优先使用 `FastIncrementalDetokenizer`（依赖 `tokenizers>=0.22.0`）
- 对于不支持 fast tokenizer 的模型，使用 `tokenizers-cpp` 或 Rust 绑定
- 将 detokenizer 移到独立线程，避免阻塞 output processing 主循环

**预期收益**：高并发 streaming 场景下 CPU 利用率降低 5-10%

**代码位置**：
- `v1/engine/detokenizer.py:30-65`
- `v1/engine/output_processor.py:269-299` (`make_request_output`)

---

## 三、鲲鹏+GPU (含 Ascend NPU) 场景优化方案

### 优化 8：Host-Device 数据传输优化

**现状**：`async_copy_to_gpu()` 逐个 tensor 做 `copy_(non_blocking=True)`。

**优化方法**：
- **批量传输**：将多个小 tensor 合并为一个大 tensor 做单次 DMA
- **Pinned Memory 策略**：ARM 上重新评估 pin_memory 行为。当前代码 (`buffer_utils.py:30`) 注释说 pin_memory 在高并发下有问题，但这是针对 CUDA 的结论，对 Ascend NPU 可能不同。在 ARM+NVIDIA GPU 环境下也应重新测试
- **NUMA-aware pinning**：使用 `mbind` 确保 pinned memory 分配在 GPU/NPU 所在的 NUMA 节点
- **Ascend 特有**：利用 HCCL 的 RDMA 能力，在 `cpu_npu.py` 中预分配 d2h/h2d buffer pool 减少 stream 创建开销
- **输入 prefetch**：在 scheduler 执行期间同时做上一批输出的 D2H 传输

**预期收益**：TTFT 降低 10-25%

**代码位置**：
- `v1/worker/gpu/buffer_utils.py:17-33` (`async_copy_to_gpu`)
- `ascend/kv_offload/cpu_npu.py:43-95` (`CpuNpuOffloadingHandler.__init__`)

---

### 优化 9：Schedule-Execute 流水线化

**现状**：`EngineCore.step()` 顺序执行 schedule → execute → update。

**优化方法**：
- 启用 `batch_queue_size > 1` 实现调度与执行重叠（`step_with_batch_queue`）
- **新增**：实现 speculative scheduling — 在当前 batch 执行期间预计算下一 batch 的 KV cache 分配
- **新增**：将 `update_from_output` 中非关键路径（如 stats 计算）延迟到下一个 step
- CPU 侧的 `async_scheduling=False` 在 `platforms/cpu.py:176` 被强制禁用 — 对于 GPU+NVIDIA 或 NPU 场景应该启用

**预期收益**：GPU/NPU 利用率提升 10-20%

**代码位置**：
- `v1/engine/core.py:378-407` (`step`)
- `v1/engine/core.py:419-535` (`step_with_batch_queue`)
- `platforms/cpu.py:175-176` (`async_scheduling = False`)

---

### 优化 10：Ascend CPU Binding 初始化加速

**现状**：`cpu_binding.py` 在启动时执行多个 subprocess 调用。

**优化方法**：
- 缓存 `npu-smi` 输出，仅在 NPU 拓扑变化时重新解析
- 使用 `sysfs` 接口 (`/sys/class/npu/`) 替代 `npu-smi` 命令获取 NPU 信息，避免 subprocess 开销
- IRQ 绑定改为使用 `irqbalance` 的配置文件 (`/etc/sysconfig/irqbalance`)，减少手动 `/proc/irq` 操作
- `migratepages` 调用改为后台异步执行

**预期收益**：多 NPU 启动时间减少 30-50%

**代码位置**：
- `ascend/cpu_binding.py:39-51` (`execute_command`)
- `ascend/cpu_binding.py:88-131` (npu-smi parsing)
- `ascend/cpu_binding.py:142-150` (`parse_topo_affinity`)

---

### 优化 11：InputBatch 构造优化

**现状**：每 step 在 Python 层重建 input tensor。

**优化方法**：
- 使用预分配 ring buffer 替代每 step 分配新 tensor
- `MultiGroupBlockTable` 使用 `numpy` 预计算，减少 Python 循环
- token_ids 和 positions 的填充使用 `torch.index_copy_` 或 scatter 操作替代逐个赋值
- Ascend 场景：新增的 `NPUInputBatch` 继承 `InputBatch`，复用优化

**预期收益**：step 延迟降低 5-10%

**代码位置**：
- `v1/worker/gpu_input_batch.py:81-1036` (InputBatch)
- `ascend/worker/npu_input_batch.py` (NPUInputBatch)

---

### 优化 12：Output Processing 批量化

**现状**：`OutputProcessor.process_outputs()` 逐 request 处理。

**优化方法**：
- 批量 detokenize：积累多个 token 后批量调用 tokenizer.decode()
- 使用 `asyncio.gather()` 并行处理独立请求的 output 构建
- `RequestOutput` 对象池化，减少 GC 压力
- 将 `LogprobsProcessor` 的 CPU 计算移到 worker 进程中

**预期收益**：高并发 (>100 reqs) streaming 场景吞吐提升 10-15%

**代码位置**：
- `v1/engine/output_processor.py:154-199` (`process_outputs`)
- `v1/engine/detokenizer.py:41-43` (`update`)

---

### 优化 13：EPLB Host 端计算加速 (Ascend MoE)

**现状**：`policy_flashlb.py` 1010+ 行 Python 逻辑在 CPU 上执行。

**优化方法**：
- 将 EPLB 的路由计算移到 worker 进程的 device 侧（利用 NPU 并行计算 token-to-expert 映射）
- 使用 `numpy` 向量化替代 Python 循环进行 expert capacity 计算
- EPLB 更新频率自适应：负载稳定时降低更新频率
- 预计算 expert-to-device 映射表缓存在 C++ 层

**预期收益**：MoE 模型吞吐提升 5-15%

**代码位置**：
- `ascend/eplb/core/policy/policy_flashlb.py`
- `ascend/eplb/core/eplb_worker.py`

---

### 优化 14：KV Cache Offload CPU 端优化

**现状**：LRU 维护使用 Python dict 操作。

**优化方法**：
- 使用 `collections.OrderedDict` 替代自定义 LRU 链表（Python 3.7+ dict 保持插入顺序）
- CPU 端 block 分配增加 NUMA 感知：offload 到 local NUMA 节点
- `CpuNpuOffloadingHandler.transfer_async()` 使用批量传输替代逐 block 传输
- 预取（prefetch）策略：在 block 被驱逐前预加载常用 block 到 NPU

**预期收益**：KV offload 场景延迟降低 10-20%

**代码位置**：
- `v1/kv_offload/lru_manager.py`
- `v1/kv_offload/worker/cpu_gpu.py`
- `ascend/kv_offload/cpu_npu.py:97-180` (`transfer_async`)

---

## 四、优化优先级矩阵

| 优先级 | 优化项 | 适用场景 | 实现复杂度 | 预期收益 |
|--------|--------|---------|-----------|---------|
| P0 | #3 NUMA 感知 | CPU | 中 | 20-40% 吞吐 |
| P0 | #1 Scheduler C 扩展 | CPU+GPU | 高 | 10-20% 延迟 |
| P0 | #9 Schedule-Execute 流水线 | GPU/NPU | 中 | 10-20% GPU 利用率 |
| P1 | #8 Host-Device 传输 | GPU/NPU | 中 | 10-25% TTFT |
| P1 | #2 KV Cache 数据结构 | CPU+GPU | 中 | 15-25% KV 操作 |
| P1 | #4 CPU Model Runner | CPU | 低 | 30% 内存 |
| P1 | #11 InputBatch | GPU/NPU | 低 | 5-10% 延迟 |
| P2 | #6 SVE 向量化 | CPU | 高 | 10-30% 矩阵运算 |
| P2 | #10 Ascend CPU Binding | NPU | 低 | 启动加速 |
| P2 | #12 Output Processing | CPU+GPU | 中 | 10-15% 吞吐 |
| P2 | #14 KV Offload | GPU/NPU | 中 | 10-20% 延迟 |
| P3 | #5 OpenMP 调优 | CPU | 低 | 5-10% 吞吐 |
| P3 | #7 Detokenizer | CPU+GPU | 低 | 2-5% e2e |
| P3 | #13 EPLB | NPU/MoE | 中 | 5-15% MoE |

---

## 五、建议实施路线

### 第一阶段（立即可做）
1. **优化 #4** — CPU Model Runner 轻量化（低复杂度，高收益）
2. **优化 #10** — Ascend CPU Binding 加速（低复杂度）
3. **优化 #5** — ARM OpenMP 调优（低复杂度）
4. **优化 #11** — InputBatch ring buffer（低复杂度）

### 第二阶段（重点攻关）
5. **优化 #3** — NUMA 感知内存管理（核心优化）
6. **优化 #9** — Schedule-Execute 流水线
7. **优化 #8** — Host-Device 传输批量化

### 第三阶段（深度优化）
8. **优化 #1** — Scheduler C 扩展
9. **优化 #2** — KV Cache 数据结构
10. **优化 #6** — ARM SVE 向量化
