# Profiling Guide: Kunpeng HOST-Side vLLM Online Inference

## Goal
Quantify Python/PyTorch/HOST overhead on Kunpeng+NV GPU and Kunpeng+Ascend, then decide which HOST-side optimizations should be implemented next.

Primary customer-visible metrics:

- TTFT mean/P90/P99.
- TPOT mean/P90/P99.
- Throughput.
- Tail latency under realistic request-rate pressure.

Internal metrics to correlate:

- Scheduler step time.
- Input preparation time.
- H2D/N2D memcpy count and duration.
- Device idle gaps caused by HOST.
- Output processing/detokenization time.
- IPC serialization/transport time.
- CPU allocation, lock, and object churn.

## Test Matrix

Run each workload for at least 3 minutes. Discard the first 30 seconds as warmup.

| Scenario | Purpose |
|---|---|
| Short prompt, high-concurrency decode | TPOT and scheduler pressure |
| Long prompt, low-concurrency prefill | TTFT and input transfer pressure |
| Mixed realistic traffic | Customer-like end-to-end behavior |
| Prefix cache on/off | Prefix hashing and KV-cache lookup impact |
| Spec decode on/off | Small tensor and rejection sampler overhead |
| Ascend KV offload or MoE/EPLB on/off | swap and FlashLB overhead |

## Start Server: NV GPU

```bash
export CUDA_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=INFO

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192
```

## Start Server: Ascend

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_LOGGING_LEVEL=INFO

python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192
```

## Benchmark Commands

Decode-heavy:

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --num-prompts 1000 \
  --request-rate 32 \
  --save-result \
  --result-dir ./prof_results/decode_heavy
```

Prefill-heavy:

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 64 \
  --num-prompts 300 \
  --request-rate 8 \
  --save-result \
  --result-dir ./prof_results/prefill_heavy
```

Mixed traffic:

```bash
python benchmarks/benchmark_serving.py \
  --backend vllm \
  --base-url http://127.0.0.1:8000 \
  --model /path/to/model \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 256 \
  --num-prompts 1000 \
  --request-rate 16 \
  --save-result \
  --result-dir ./prof_results/mixed
```

## Process Discovery

```bash
ps -ef | grep -E "api_server|vllm" | grep -v grep
```

Profile the API process, EngineCore process, worker process, and output/detokenizer process if separate.

## Python Profiling: py-spy

```bash
mkdir -p ./prof_results/pyspy

py-spy record \
  -p <PID> \
  --rate 999 \
  --duration 180 \
  --native \
  -o ./prof_results/pyspy/vllm_<PID>.svg
```

Real-time view:

```bash
py-spy top -p <PID> --rate 999 --native
```

Interpretation:

| Observation | Next Step |
|---|---|
| `scheduler.py` >8% wall time | Scheduler data structure/index optimization |
| `gpu_model_runner.py` or Ascend `model_runner_v1.py` >5% | Input preparation and metadata copy optimization |
| `msgpack` or `zmq` >3% | Message slimming before SHM |
| output/detokenizer >5% | Output core isolation or batching |
| frequent tensor/list/dict creation frames | Buffer/object reuse |

## Native Profiling: perf

```bash
mkdir -p ./prof_results/perf

perf record -F 997 -g -p <PID> -- sleep 180
perf report --stdio > ./prof_results/perf/perf_<PID>.txt
```

Search key symbols:

```bash
grep -Ei "PyObject|dict|list|unicode|malloc|free|futex|pthread|poll|epoll|msgpack|zmq|torch" \
  ./prof_results/perf/perf_<PID>.txt
```

Interpretation:

| Observation | Next Step |
|---|---|
| `malloc/free` >5% CPU | Object pool and persistent buffers |
| `futex/pthread` >5% CPU | Queue/thread/IPC contention analysis |
| Python object/hash/list symbols high | request_id-to-int and arrayization |
| msgpack/ZMQ high | reduce serialized fields and object count |

## NV GPU Timeline: nsys

```bash
mkdir -p ./prof_results/nsys

nsys profile \
  -t cuda,nvtx,osrt \
  --sample=cpu \
  --cpuctxsw=true \
  -o ./prof_results/nsys/vllm_trace \
  python -m vllm.entrypoints.openai.api_server \
    --model /path/to/model \
    --host 0.0.0.0 \
    --port 8000
```

Look for:

- HOST gaps between CUDA kernels.
- Many small `cudaMemcpyAsync` operations.
- GPU utilization drops aligned with scheduler/input preparation.
- CPU context switch spikes during request scheduling.

## Ascend Timeline: msprof / CANN Profiler

```bash
mkdir -p ./prof_results/msprof

msprof --application="python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model --host 0.0.0.0 --port 8000" \
  --output=./prof_results/msprof \
  --aicpu=on \
  --sys-hardware-mem=on \
  --sys-cpu-profiling=on
```

Look for:

- `aclrtMemcpy` or N2D/D2H fragmentation.
- HOST waits before NPU stream work.
- KV swap share in TTFT.
- FlashLB or MoE CPU spikes.

## NUMA and CPU Affinity

```bash
mkdir -p ./prof_results/numa

numactl --hardware > ./prof_results/numa/numactl_hardware.txt
numastat -p <PID> 1 | tee ./prof_results/numa/numastat_<PID>.txt
taskset -cp <PID> | tee ./prof_results/numa/taskset_<PID>.txt
ps -eLo pid,tid,psr,comm | grep -E "python|vllm|acl|npu" \
  > ./prof_results/numa/thread_cpu_map.txt
```

Interpretation:

| Observation | Next Step |
|---|---|
| Remote NUMA memory grows | Fix memory policy and allocation order |
| Scheduler migrates across cores | Engine/scheduler core isolation |
| Device IRQs on remote NUMA | IRQ affinity tuning outside or alongside vLLM |
| Output/IPC shares scheduler cores | Separate CPU sets |

## Environment Capture

```bash
mkdir -p ./prof_results

{
  date
  uname -a
  lscpu
  numactl --hardware
  python -V
  pip freeze | grep -E "vllm|torch|torch-npu|numpy|tokenizers|msgpack|pyzmq"
} > ./prof_results/env.txt
```

## Optional Lightweight Internal Instrumentation

If py-spy/perf indicate HOST gaps but attribution is unclear, add temporary per-step JSONL timers around:

- `scheduler.schedule`.
- KV cache allocate/free.
- input batch update.
- input preparation.
- H2D/N2D copy submission.
- model execute wall time.
- output process.
- detokenize.
- ZMQ send/receive.

Recommended output:

```json
{"step": 123, "schedule_us": 180, "prepare_us": 95, "copy_us": 42, "execute_us": 3100, "output_us": 70}
```

Use this only for controlled profiling runs. Keep overhead below 1% by aggregating in memory and flushing periodically.

## Decision Rules

| Evidence | Promote Optimization |
|---|---|
| Scheduler Python >8% wall time | Integer request index and SchedulerOutput arrayization |
| Input prepare >5% wall time | Metadata buffer reuse and copy batching |
| `malloc/free` >5% CPU | Per-step object pools and tensor caches |
| H2D/N2D has many small copies | Persistent pinned buffers and batched metadata transfer |
| ZMQ/msgpack >3% wall time | Message structure slimming before SHM |
| output/detokenize >5% wall time | Output core isolation and batch response handling |
| Device timeline shows HOST gaps | Prioritize scheduler/input prepare over device kernels |

## Deliverables After Each Profiling Run

Collect these artifacts before making optimization decisions:

- Benchmark JSON results for all scenarios.
- `py-spy` SVGs for each relevant process.
- `perf_<PID>.txt` for each relevant process.
- NV `nsys` trace or Ascend `msprof` output.
- NUMA/affinity logs.
- Environment snapshot.
- Short finding summary with top 5 HOST bottlenecks and proposed next changes.

