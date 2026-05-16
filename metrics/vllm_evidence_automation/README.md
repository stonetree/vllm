# vLLM evidence automation

This package automates repeated `vllm serve` restarts, affinity binding, single-request benchmark rounds, and evidence collection for the Kunpeng + RTX 4090 eager-mode investigation.

## Quick start

```bash
cd /home/liuliu/projects/vllm/metrics/vllm_evidence_automation
cp config.env.example config.env
vim config.env
./run_experiment.sh ./config.env
```

All evidence is saved under:

```text
/tmp/vllm_evidence_YYYYmmdd_HHMMSS/
  summary.tsv
  by_class/
    good/
    bad/
    middle/
    unknown/
  run_001/
    logs/
    snapshots/
    rounds/
      round_1/
        bench_command.txt
        bench.log
        bench.exit_code
        bench.metrics.json
        evidence/
          collectors.tsv
          collectors.final.tsv
          pidstat/
          gpu_watch/
          nvidia_q/
          snapshot_during/
      round_2/
```

Each run is one `vllm serve` restart. Each round is one benchmark execution against the same server.
The round directory is the root for that benchmark's outputs. The main process
writes the benchmark command, raw benchmark log, exit code, and parsed metrics
there. Each evidence collector writes only under its own
`round_N/evidence/<collector>/` directory.

## Evidence timing

The main loop captures:

- `T0_after_engine_birth`
- `T1_after_light_warmup`
- `T2_before_representative_warmup`
- `T3_after_representative_warmup`
- `T4_after_final_bind`
- `T5_during_round1`
- `T6_after_round1`
- `T7_during_round2`
- `T8_after_round2`

Each snapshot includes thread placement, allowed CPU list, current PSR, per-thread CPU, context switch counters, NUMA maps, numastat, smaps rollup, interrupts, and GPU query output.

During-benchmark evidence collection is isolated in scripts under
`collectors/`. The main process does not run `pidstat`, `perf`, `nvidia-smi`,
or during-snapshot commands directly during a benchmark round. It creates the
round evidence directory, starts enabled collectors in the background, records
their PID and output directory in `collectors.tsv`, and waits for them after the
benchmark exits. Final collector status is written to `collectors.final.tsv`.

Collector interface:

```text
collector.sh OUT_DIR API_PID ENGINE_PID DURATION_SEC ROUND TAG
```

Each collector writes `stdout.log`, `stderr.log`, `exit_code`, `meta.env`, and
collector-specific files with descriptive names, such as `pidstat_threads.txt`,
`gpu_watch.csv`, `perf_sched.data`, `perf_engine.report.txt`, or
`T5_during_round1.ENGINE.threads.tsv`.

## Important config knobs

- `RUNS`: number of serve restarts.
- `BENCH_ROUNDS_PER_SERVE`: number of benchmark rounds under one server.
- `GOOD_TPOT_MS` and `BAD_TPOT_MS`: first-round TPOT thresholds used for classification.
- `REPRESENTATIVE_WARMUP`: set to `1` to test whether representative warmup removes first-round state changes.
- `COLLECT_PIDSTAT`: set to `1` to collect per-thread pidstat output during each benchmark round.
- `COLLECT_GPU_WATCH`: set to `1` to collect per-second GPU state during each benchmark round.
- `COLLECT_PERF_SCHED`: set to `1` to collect `perf sched` evidence during each benchmark round.
- `COLLECT_PERF_ENGINE`: set to `1` to collect EngineCore callgraph evidence during each benchmark round.
- `COLLECT_SNAPSHOT_DURING`: set to `1` to collect a delayed during-benchmark snapshot.
- `COLLECT_NVIDIA_Q`: set to `1` to collect one detailed GPU query per benchmark round.
- `BENCH_CMD_TEMPLATE`: benchmark command. Keep `{MODEL}`, `{HOST}`, `{PORT}`, and `{NUM_PROMPTS}` placeholders if useful.

## Post-run analysis

```bash
./analyze_evidence.sh /tmp/vllm_evidence_YYYYmmdd_HHMMSS
```

The analysis helper prints the summary, first-round TPOT classification, top EngineCore CPU threads at T4/T6/T8, and NUMA summaries for one good and one bad sample when available.
