#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
run_script="$script_dir/run_experiment.sh"
config_example="$script_dir/config.env.example"

fail() {
    printf 'FAIL: %s\n' "$*" >&2
    exit 1
}

for collector in pidstat gpu_watch perf_sched perf_engine snapshot_after_delay nvidia_q; do
    path="$script_dir/collectors/$collector.sh"
    [[ -x "$path" ]] || fail "missing executable collector: collectors/$collector.sh"
    grep -q 'OUT_DIR.*API_PID.*ENGINE_PID.*DURATION_SEC.*ROUND.*TAG' "$path" \
        || fail "collector $collector.sh must document the unified interface"
    grep -q 'meta.env' "$path" \
        || fail "collector $collector.sh must write meta.env"
    grep -q 'exit_code' "$path" \
        || fail "collector $collector.sh must write exit_code"
done

grep -q '^start_bench_evidence_collectors() {' "$run_script" \
    || fail "run_experiment.sh must define start_bench_evidence_collectors"
grep -q '^wait_bench_evidence_collectors() {' "$run_script" \
    || fail "run_experiment.sh must define wait_bench_evidence_collectors"

for direct in 'pidstat -t' 'perf sched record' 'perf record' 'nvidia-smi.*-l 1'; do
    if grep -Eq "$direct" "$run_script"; then
        fail "run_experiment.sh must not contain direct collector command: $direct"
    fi
done

for knob in \
    COLLECT_PIDSTAT \
    COLLECT_GPU_WATCH \
    COLLECT_PERF_SCHED \
    COLLECT_PERF_ENGINE \
    COLLECT_SNAPSHOT_DURING \
    COLLECT_NVIDIA_Q \
    PIDSTAT_DURATION_SEC \
    GPU_WATCH_DURATION_SEC \
    PERF_SCHED_DURATION_SEC \
    PERF_ENGINE_DURATION_SEC \
    SNAPSHOT_DURING_DELAY_SEC; do
    grep -q "^$knob=" "$config_example" \
        || fail "config.env.example missing $knob"
done

grep -q 'collectors.tsv' "$run_script" \
    || fail "run_experiment.sh must create evidence/collectors.tsv"
grep -q 'collectors.final.tsv' "$run_script" \
    || fail "run_experiment.sh must create evidence/collectors.final.tsv"

perf_engine_script="$script_dir/collectors/perf_engine.sh"
grep -q 'perf record -F "$perf_freq" -g -p "$engine_pid" -o "$out_dir/perf_engine.data" -- sleep "$duration_sec"' "$perf_engine_script" \
    || fail "perf_engine collector must keep the perf record template and specify a per-collector perf.data path with -o"
grep -q 'perf report -i "$out_dir/perf_engine.data" --stdio --sort comm,dso,symbol > "$out_dir/perf_engine.report.txt"' "$perf_engine_script" \
    || fail "perf_engine collector must write perf report under its own output directory"
grep -q '^PERF_ENGINE_DURATION_SEC=60$' "$config_example" \
    || fail "config.env.example must default PERF_ENGINE_DURATION_SEC to 60"
grep -q ': "${PERF_ENGINE_DURATION_SEC:=60}"' "$run_script" \
    || fail "run_experiment.sh must default PERF_ENGINE_DURATION_SEC to 60"
grep -q 'KEEP_PERF_ENGINE_DATA=${KEEP_PERF_ENGINE_DATA:-0}' "$perf_engine_script" \
    || fail "perf_engine collector must read KEEP_PERF_ENGINE_DATA from the environment"
grep -q 'if \[\[ "$KEEP_PERF_ENGINE_DATA" != "1" \]\]' "$perf_engine_script" \
    || fail "perf_engine collector must delete perf.data unless KEEP_PERF_ENGINE_DATA=1"
grep -q 'rm -f "$out_dir/perf_engine.data"' "$perf_engine_script" \
    || fail "perf_engine collector must remove perf_engine.data by default"
if grep -q '^KEEP_PERF_ENGINE_DATA=' "$config_example"; then
    fail "KEEP_PERF_ENGINE_DATA must not be configured in config.env.example"
fi

printf 'PASS: bench evidence collectors have isolated scripts, config knobs, and manifests\n'
