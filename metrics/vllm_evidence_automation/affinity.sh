#!/usr/bin/env bash
set -Eeuo pipefail

mode=${1:?mode required: initial|final}
api_pid=${2:?api pid required}
engine_pid=${3:?engine pid required}
log_file=${4:?log file required}

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_file=${CONFIG_FILE:-"$script_dir/config.env"}
if [[ -f "$config_file" ]]; then
    # shellcheck disable=SC1090
    source "$config_file"
fi

: "${API_IO_MASK:=4-7}"
: "${API_GARBAGE_MASK:=16-31}"
: "${ENGINE_INITIAL_MASK:=9-15}"
: "${ENGINE_WORKER_MASK:=16-20}"
: "${ENGINE_MAIN_CORE:=8}"
: "${API_CUDA_CORE:=16}"
: "${ENGINE_CUDA_CORE:=9}"
: "${ENGINE_ZMQ_REAPER_CORE:=10}"
: "${ENGINE_ZMQ_IO0_CORE:=11}"
: "${ENGINE_ZMQ_IO1_CORE:=12}"
: "${ENGINE_TCPSTORE_CORE:=13}"
: "${ENGINE_CUDA_EVT_CORE:=14}"

mkdir -p "$(dirname "$log_file")"
exec >> "$log_file" 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

walk_and_bind() {
    local pid=$1
    local mask=$2
    local tid child

    for tid in $(ls "/proc/$pid/task" 2>/dev/null || true); do
        taskset -pc "$mask" "$tid" >/dev/null 2>&1 || true
    done

    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        walk_and_bind "$child" "$mask"
    done
}

bind_api_tree_except_engine() {
    local api=$1
    local engine=$2
    local io_mask=$3
    local garbage_mask=$4
    local tid child io_tid

    for tid in $(ls "/proc/$api/task" 2>/dev/null || true); do
        taskset -pc "$garbage_mask" "$tid" >/dev/null 2>&1 || true
    done

    for child in $(pgrep -P "$api" 2>/dev/null || true); do
        if [[ "$child" != "$engine" ]]; then
            walk_and_bind "$child" "$garbage_mask"
        fi
    done

    for io_tid in $(ps -T -p "$api" 2>/dev/null | grep -E 'iou-sqp|ZMQbg' | awk '{print $2}'); do
        taskset -pc "$io_mask" "$io_tid" >/dev/null 2>&1 || true
    done
}

bind_exact_by_name() {
    local pid=$1
    local keyword=$2
    local core=$3
    local tid

    printf '[%s] bind_exact_by_name pid=%s keyword=%s core=%s\n' "$(timestamp)" "$pid" "$keyword" "$core"
    for tid in $(ps -T -p "$pid" 2>/dev/null | grep "$keyword" | awk '{print $2}'); do
        printf '[%s]   tid=%s\n' "$(timestamp)" "$tid"
        taskset -pc "$core" "$tid" >/dev/null 2>&1 || true
    done
}

dump_binding() {
    local label=$1
    local pid=$2
    printf '\n[%s] %s pid=%s thread placement\n' "$(timestamp)" "$label" "$pid"
    ps -T -p "$pid" -o spid,psr,pcpu,comm 2>&1 || true
}

printf '\n[%s] affinity mode=%s api_pid=%s engine_pid=%s\n' "$(timestamp)" "$mode" "$api_pid" "$engine_pid"

if [[ "$mode" == "initial" ]]; then
    walk_and_bind "$engine_pid" "$ENGINE_INITIAL_MASK"
    bind_api_tree_except_engine "$api_pid" "$engine_pid" "$API_IO_MASK" "$API_GARBAGE_MASK"
    dump_binding API "$api_pid"
    dump_binding ENGINE "$engine_pid"
    exit 0
fi

if [[ "$mode" != "final" ]]; then
    printf 'unknown affinity mode: %s\n' "$mode" >&2
    exit 2
fi

bind_api_tree_except_engine "$api_pid" "$engine_pid" "$API_IO_MASK" "$API_GARBAGE_MASK"
bind_exact_by_name "$api_pid" 'iou-sqp' 4
bind_exact_by_name "$api_pid" 'ZMQbg/Reaper' 5
bind_exact_by_name "$api_pid" 'ZMQbg/IO/0' 6
bind_exact_by_name "$api_pid" 'ZMQbg/IO/1' 7
bind_exact_by_name "$api_pid" 'cuda0' "$API_CUDA_CORE"

walk_and_bind "$engine_pid" "$ENGINE_WORKER_MASK"
taskset -pc "$ENGINE_MAIN_CORE" "$engine_pid" >/dev/null 2>&1 || true
bind_exact_by_name "$engine_pid" 'cuda0' "$ENGINE_CUDA_CORE"
bind_exact_by_name "$engine_pid" 'ZMQbg/Reaper' "$ENGINE_ZMQ_REAPER_CORE"
bind_exact_by_name "$engine_pid" 'ZMQbg/IO/0' "$ENGINE_ZMQ_IO0_CORE"
bind_exact_by_name "$engine_pid" 'ZMQbg/IO/1' "$ENGINE_ZMQ_IO1_CORE"
bind_exact_by_name "$engine_pid" 'pt_tcpstore' "$ENGINE_TCPSTORE_CORE"
bind_exact_by_name "$engine_pid" 'cuda-EvtHandlr' "$ENGINE_CUDA_EVT_CORE"
taskset -pc "$ENGINE_MAIN_CORE" "$engine_pid" >/dev/null 2>&1 || true

dump_binding API "$api_pid"
dump_binding ENGINE "$engine_pid"
printf '[%s] final main-core tids on core %s: ' "$(timestamp)" "$ENGINE_MAIN_CORE"
ps -T -p "$engine_pid" -o spid,psr 2>/dev/null | awk -v c="$ENGINE_MAIN_CORE" '$2==c {printf "%s ", $1} END {print ""}'

