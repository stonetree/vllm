#!/usr/bin/env bash
set -Eeuo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_file=${1:-"$script_dir/config.env"}
if [[ ! -f "$config_file" ]]; then
    printf 'config not found: %s\ncopy config.env.example to config.env first\n' "$config_file" >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$config_file"
export CONFIG_FILE="$config_file"

: "${MODEL:?MODEL is required}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"
: "${RUNS:=20}"
: "${BENCH_ROUNDS_PER_SERVE:=2}"
: "${NUM_PROMPTS:=10}"
: "${CORES:=4-31}"
: "${NUMA_NODE:=0}"
: "${SERVER_READY_TIMEOUT_SEC:=600}"
: "${PID_WAIT_TIMEOUT_SEC:=600}"
: "${KV_CACHE_WAIT_SEC:=120}"
: "${POST_LIGHT_WARMUP_SLEEP_SEC:=3}"
: "${POST_REP_WARMUP_SLEEP_SEC:=3}"
: "${POST_FINAL_BIND_SLEEP_SEC:=2}"
: "${BENCH_SNAPSHOT_DELAY_SEC:=10}"
: "${BETWEEN_RUN_SLEEP_SEC:=10}"
: "${SHUTDOWN_GRACE_SEC:=15}"
: "${REPRESENTATIVE_WARMUP:=1}"
: "${COLLECT_PIDSTAT:=1}"
: "${COLLECT_GPU_WATCH:=1}"
: "${COLLECT_PERF_SCHED:=0}"
: "${COLLECT_PERF_ENGINE:=0}"
: "${COLLECT_SNAPSHOT_DURING:=1}"
: "${COLLECT_NVIDIA_Q:=1}"
: "${PIDSTAT_DURATION_SEC:=120}"
: "${GPU_WATCH_DURATION_SEC:=120}"
: "${PERF_SCHED_DURATION_SEC:=20}"
: "${PERF_ENGINE_DURATION_SEC:=60}"
: "${SNAPSHOT_DURING_DELAY_SEC:=${BENCH_SNAPSHOT_DELAY_SEC:-10}}"
: "${NVIDIA_Q_DURATION_SEC:=0}"
: "${COLLECTOR_WAIT_TIMEOUT_SEC:=180}"
: "${PERF_FREQ:=99}"
: "${GOOD_TPOT_MS:=13.7}"
: "${BAD_TPOT_MS:=14.1}"
: "${VLLM_TMP_LOG:=/opt/data/.vllm.serv.tmp}"
: "${BENCH_CMD_TEMPLATE:?BENCH_CMD_TEMPLATE is required}"
: "${SERVE_EXTRA_ARGS:=}"

exp_root=${EXP_ROOT:-"/tmp/vllm_evidence_$(date +%Y%m%d_%H%M%S)"}
mkdir -p "$exp_root"
summary_file="$exp_root/summary.tsv"
printf 'run\tclass\tfirst_round_mean_tpot_ms\tapi_pid\tengine_pid\trun_dir\n' > "$summary_file"

log_msg() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

run_pre_serve_actions() {
    # User-defined actions that must run before each vllm serve launch belong here.
    # Keep this block before the serve subprocess starts so every run has the same
    # pre-serve environment reset point.
    sync || true

    # Clear PageCache, dentries, and inodes.
    if [[ -w /proc/sys/vm/drop_caches ]]; then
        echo 3 > /proc/sys/vm/drop_caches || true
    fi

    # Ask the kernel to compact physical memory for contiguous huge pages.
    if [[ -w /proc/sys/vm/compact_memory ]]; then
        echo 1 > /proc/sys/vm/compact_memory || true
    fi
}

substitute_bench_cmd() {
    local cmd=$BENCH_CMD_TEMPLATE
    cmd=${cmd//\{MODEL\}/$MODEL}
    cmd=${cmd//\{HOST\}/$HOST}
    cmd=${cmd//\{PORT\}/$PORT}
    cmd=${cmd//\{NUM_PROMPTS\}/$NUM_PROMPTS}
    printf '%s\n' "$cmd"
}

wait_http_ready() {
    local deadline=$((SECONDS + SERVER_READY_TIMEOUT_SEC))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://$HOST:$PORT/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    return 1
}

extract_pid_from_logs() {
    local pattern=$1
    local serve_log=$2
    {
        grep -h -oP "$pattern" "$serve_log" 2>/dev/null || true
        grep -h -oP "$pattern" "$VLLM_TMP_LOG" 2>/dev/null || true
    } | tail -n 1
}

wait_for_pids() {
    local serve_pid=$1
    local serve_log=$2
    local pid_file=$3
    local deadline=$((SECONDS + PID_WAIT_TIMEOUT_SEC))
    local api_pid engine_pid

    while (( SECONDS < deadline )); do
        api_pid=$(extract_pid_from_logs 'APIServer pid=\K[0-9]+' "$serve_log")
        engine_pid=$(extract_pid_from_logs 'EngineCore pid=\K[0-9]+' "$serve_log")
        [[ -z "$api_pid" ]] && api_pid=$serve_pid
        if [[ -n "$api_pid" && -n "$engine_pid" && -d "/proc/$api_pid" && -d "/proc/$engine_pid" ]]; then
            printf 'API_PID=%s\nENGINE_PID=%s\n' "$api_pid" "$engine_pid" > "$pid_file"
            return 0
        fi
        sleep 1
    done
    return 1
}

collect_snapshot() {
    local tag=$1
    local api_pid=$2
    local engine_pid=$3
    local out_dir=$4
    "$script_dir/collect_snapshot.sh" "$tag" "$api_pid" "$engine_pid" "$out_dir" >> "$out_dir/snapshot.log" 2>&1 || true
}

start_collector() {
    local name=$1
    local script=$2
    local out_dir=$3
    local duration_sec=$4
    local round=$5
    local tag=$6
    local api_pid=$7
    local engine_pid=$8
    local manifest=$9
    local started_at

    started_at=$(date -Iseconds)
    mkdir -p "$out_dir"
    PERF_FREQ="$PERF_FREQ" "$script" "$out_dir" "$api_pid" "$engine_pid" "$duration_sec" "$round" "$tag" \
        > "$out_dir/stdout.log" 2> "$out_dir/stderr.log" &
    local collector_pid=$!
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$collector_pid" "$out_dir" "$started_at" "$duration_sec" "running" >> "$manifest"
}

start_bench_evidence_collectors() {
    local round_dir=$1
    local round=$2
    local api_pid=$3
    local engine_pid=$4
    local tag=$5
    local evidence_dir="$round_dir/evidence"
    local manifest="$evidence_dir/collectors.tsv"

    mkdir -p "$evidence_dir"
    printf 'name\tpid\tout_dir\tstarted_at\tduration_sec\tstatus\n' > "$manifest"

    if [[ "$COLLECT_PIDSTAT" == "1" ]]; then
        start_collector pidstat "$script_dir/collectors/pidstat.sh" "$evidence_dir/pidstat" \
            "$PIDSTAT_DURATION_SEC" "$round" "$tag" "$api_pid" "$engine_pid" "$manifest"
    fi
    if [[ "$COLLECT_GPU_WATCH" == "1" ]]; then
        start_collector gpu_watch "$script_dir/collectors/gpu_watch.sh" "$evidence_dir/gpu_watch" \
            "$GPU_WATCH_DURATION_SEC" "$round" "$tag" "$api_pid" "$engine_pid" "$manifest"
    fi
    if [[ "$COLLECT_PERF_SCHED" == "1" ]]; then
        start_collector perf_sched "$script_dir/collectors/perf_sched.sh" "$evidence_dir/perf_sched" \
            "$PERF_SCHED_DURATION_SEC" "$round" "$tag" "$api_pid" "$engine_pid" "$manifest"
    fi
    if [[ "$COLLECT_PERF_ENGINE" == "1" ]]; then
        start_collector perf_engine "$script_dir/collectors/perf_engine.sh" "$evidence_dir/perf_engine" \
            "$PERF_ENGINE_DURATION_SEC" "$round" "$tag" "$api_pid" "$engine_pid" "$manifest"
    fi
    if [[ "$COLLECT_SNAPSHOT_DURING" == "1" ]]; then
        start_collector snapshot_during "$script_dir/collectors/snapshot_after_delay.sh" "$evidence_dir/snapshot_during" \
            "$SNAPSHOT_DURING_DELAY_SEC" "$round" "$tag" "$api_pid" "$engine_pid" "$manifest"
    fi
    if [[ "$COLLECT_NVIDIA_Q" == "1" ]]; then
        start_collector nvidia_q "$script_dir/collectors/nvidia_q.sh" "$evidence_dir/nvidia_q" \
            "$NVIDIA_Q_DURATION_SEC" "$round" "$tag" "$api_pid" "$engine_pid" "$manifest"
    fi
}

wait_bench_evidence_collectors() {
    local evidence_dir=$1
    local manifest="$evidence_dir/collectors.tsv"
    local final_manifest="$evidence_dir/collectors.final.tsv"
    local name pid out_dir started_at duration_sec status ended_at rc final_status

    [[ -f "$manifest" ]] || return 0
    printf 'name\tpid\tout_dir\tstarted_at\tended_at\tduration_sec\tstatus\texit_code\n' > "$final_manifest"

    while IFS=$'\t' read -r name pid out_dir started_at duration_sec status; do
        [[ "$name" == "name" || -z "$name" ]] && continue

        final_status=finished
        local timeout_marker="$out_dir/.collector_timeout"
        rm -f "$timeout_marker"
        (
            sleep "$COLLECTOR_WAIT_TIMEOUT_SEC"
            if kill -0 "$pid" >/dev/null 2>&1; then
                touch "$timeout_marker"
                kill -TERM "$pid" >/dev/null 2>&1 || true
                sleep 2
                kill -KILL "$pid" >/dev/null 2>&1 || true
            fi
        ) &
        local killer_pid=$!

        set +e
        wait "$pid" >/dev/null 2>&1
        rc=$?
        kill "$killer_pid" >/dev/null 2>&1 || true
        wait "$killer_pid" >/dev/null 2>&1 || true
        set -e

        if [[ -f "$timeout_marker" ]]; then
            final_status=timeout
        fi

        if [[ ! -f "$out_dir/exit_code" ]]; then
            printf '%s\n' "$rc" > "$out_dir/exit_code"
        fi
        rm -f "$timeout_marker"
        ended_at=$(date -Iseconds)
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$pid" "$out_dir" "$started_at" "$ended_at" "$duration_sec" "$final_status" "$rc" >> "$final_manifest"
    done < "$manifest"
}

light_warmup() {
    curl -fsS "http://$HOST:$PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d '{"model":"'"$MODEL"'","prompt":"Hello","max_tokens":1,"temperature":0.0}' >/dev/null
}

run_bench_round() {
    local round=$1
    local api_pid=$2
    local engine_pid=$3
    local run_dir=$4
    local round_dir="$run_dir/rounds/round_${round}"
    local snapshot_dir="$run_dir/snapshots"
    local during_tag after_tag
    mkdir -p "$round_dir"

    if [[ "$round" == "1" ]]; then
        during_tag=T5_during_round1
        after_tag=T6_after_round1
    elif [[ "$round" == "2" ]]; then
        during_tag=T7_during_round2
        after_tag=T8_after_round2
    else
        during_tag="T_round_${round}_during"
        after_tag="T_round_${round}_after"
    fi

    start_bench_evidence_collectors "$round_dir" "$round" "$api_pid" "$engine_pid" "$during_tag"

    local bench_cmd
    bench_cmd=$(substitute_bench_cmd)
    printf '%s\n' "$bench_cmd" > "$round_dir/bench_command.txt"
    set +e
    bash -lc "$bench_cmd" > "$round_dir/bench.log" 2>&1
    local bench_rc=$?
    set -e

    wait_bench_evidence_collectors "$round_dir/evidence"
    printf '%s\n' "$bench_rc" > "$round_dir/bench.exit_code"

    sleep 3
    collect_snapshot "$after_tag" "$api_pid" "$engine_pid" "$snapshot_dir"
    "$script_dir/parse_bench.py" "$round_dir/bench.log" > "$round_dir/bench.metrics.json" 2> "$round_dir/bench.metrics.err" || true

    return "$bench_rc"
}

classify_run() {
    local metrics_json=$1
    python3 - "$metrics_json" "$GOOD_TPOT_MS" "$BAD_TPOT_MS" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
good = float(sys.argv[2])
bad = float(sys.argv[3])
try:
    data = json.loads(path.read_text())
except Exception:
    print("unknown\t")
    raise SystemExit(0)
tpot = data.get("mean_tpot_ms")
if tpot is None:
    print("unknown\t")
elif tpot <= good:
    print(f"good\t{tpot}")
elif tpot >= bad:
    print(f"bad\t{tpot}")
else:
    print(f"middle\t{tpot}")
PY
}

stop_process_tree() {
    local pid=$1
    if [[ -z "$pid" || ! -d "/proc/$pid" ]]; then
        return 0
    fi
    pkill -TERM -P "$pid" >/dev/null 2>&1 || true
    kill -TERM "$pid" >/dev/null 2>&1 || true
    sleep "$SHUTDOWN_GRACE_SEC"
    pkill -KILL -P "$pid" >/dev/null 2>&1 || true
    kill -KILL "$pid" >/dev/null 2>&1 || true
}

chmod +x "$script_dir"/*.sh "$script_dir"/*.py "$script_dir"/collectors/*.sh
log_msg "EXP_ROOT=$exp_root"

for run in $(seq 1 "$RUNS"); do
    run_id=$(printf 'run_%03d' "$run")
    run_dir="$exp_root/$run_id"
    mkdir -p "$run_dir"/{logs,snapshots,rounds}
    serve_log="$run_dir/logs/serve.log"
    affinity_log="$run_dir/logs/affinity.log"
    pid_file="$run_dir/pids.env"

    log_msg "starting $run_id"
    run_pre_serve_actions
    : > "$serve_log"
    : > "$affinity_log"
    : > "$VLLM_TMP_LOG" 2>/dev/null || true

    if [[ -n "${EXTRA_ENV_FILE:-}" && -f "$EXTRA_ENV_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$EXTRA_ENV_FILE"
    fi

    export MODEL HOST PORT MALLOC_CONF OMP_NUM_THREADS OMP_WAIT_POLICY OMP_PROC_BIND OMP_PLACES
    export MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS PYTHONGC gc_disable
    if [[ -n "${LD_PRELOAD_EXTRA:-}" ]]; then
        export LD_PRELOAD="$LD_PRELOAD_EXTRA${LD_PRELOAD:+:$LD_PRELOAD}"
    fi

    # shellcheck disable=SC2086
    numactl -C "$CORES" -m "$NUMA_NODE" vllm serve "$MODEL" $SERVE_EXTRA_ARGS --port "$PORT" > "$serve_log" 2>&1 &
    serve_pid=$!
    printf '%s\n' "$serve_pid" > "$run_dir/serve.pid"

    if ! wait_for_pids "$serve_pid" "$serve_log" "$pid_file"; then
        log_msg "$run_id failed to detect API/Engine PIDs"
        stop_process_tree "$serve_pid"
        printf '%s\tunknown\t\t\t\t%s\n' "$run_id" "$run_dir" >> "$summary_file"
        continue
    fi
    # shellcheck disable=SC1090
    source "$pid_file"

    collect_snapshot T0_after_engine_birth "$API_PID" "$ENGINE_PID" "$run_dir/snapshots"
    "$script_dir/affinity.sh" initial "$API_PID" "$ENGINE_PID" "$affinity_log" || true

    if ! wait_http_ready; then
        log_msg "$run_id server did not become HTTP-ready"
        stop_process_tree "$serve_pid"
        printf '%s\tunknown\t\t%s\t%s\t%s\n' "$run_id" "$API_PID" "$ENGINE_PID" "$run_dir" >> "$summary_file"
        continue
    fi

    sleep "$KV_CACHE_WAIT_SEC"
    light_warmup > "$run_dir/logs/light_warmup.log" 2>&1 || true
    sleep "$POST_LIGHT_WARMUP_SLEEP_SEC"
    collect_snapshot T1_after_light_warmup "$API_PID" "$ENGINE_PID" "$run_dir/snapshots"

    if [[ "$REPRESENTATIVE_WARMUP" == "1" ]]; then
        collect_snapshot T2_before_representative_warmup "$API_PID" "$ENGINE_PID" "$run_dir/snapshots"
        "$script_dir/representative_warmup.py" > "$run_dir/logs/representative_warmup.log" 2>&1 || true
        sleep "$POST_REP_WARMUP_SLEEP_SEC"
        collect_snapshot T3_after_representative_warmup "$API_PID" "$ENGINE_PID" "$run_dir/snapshots"
    fi

    "$script_dir/affinity.sh" final "$API_PID" "$ENGINE_PID" "$affinity_log" || true
    sleep "$POST_FINAL_BIND_SLEEP_SEC"
    collect_snapshot T4_after_final_bind "$API_PID" "$ENGINE_PID" "$run_dir/snapshots"

    for round in $(seq 1 "$BENCH_ROUNDS_PER_SERVE"); do
        log_msg "$run_id round_$round"
        run_bench_round "$round" "$API_PID" "$ENGINE_PID" "$run_dir" || true
    done

    class_and_tpot=$(classify_run "$run_dir/rounds/round_1/bench.metrics.json")
    run_class=$(printf '%s' "$class_and_tpot" | awk -F '\t' '{print $1}')
    first_tpot=$(printf '%s' "$class_and_tpot" | awk -F '\t' '{print $2}')
    mkdir -p "$exp_root/by_class/$run_class"
    ln -s "../../$run_id" "$exp_root/by_class/$run_class/$run_id" 2>/dev/null || true
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$run_id" "$run_class" "$first_tpot" "$API_PID" "$ENGINE_PID" "$run_dir" >> "$summary_file"

    stop_process_tree "$serve_pid"
    sleep "$BETWEEN_RUN_SLEEP_SEC"
done

log_msg "done: $exp_root"
printf 'summary: %s\n' "$summary_file"
