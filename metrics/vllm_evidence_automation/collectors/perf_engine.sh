#!/usr/bin/env bash
set -Eeuo pipefail

# Interface: OUT_DIR API_PID ENGINE_PID DURATION_SEC ROUND TAG
out_dir=${1:?OUT_DIR required}
api_pid=${2:?API_PID required}
engine_pid=${3:?ENGINE_PID required}
duration_sec=${4:?DURATION_SEC required}
round=${5:?ROUND required}
tag=${6:?TAG required}

perf_freq=${PERF_FREQ:-99}
KEEP_PERF_ENGINE_DATA=${KEEP_PERF_ENGINE_DATA:-0}

mkdir -p "$out_dir"
exit_code_file="$out_dir/exit_code"
rc=0
trap 'rc=$?; printf "%s\n" "$rc" > "$exit_code_file"' EXIT

cat > "$out_dir/meta.env" <<EOF
COLLECTOR=perf_engine
OUT_DIR=$out_dir
API_PID=$api_pid
ENGINE_PID=$engine_pid
DURATION_SEC=$duration_sec
ROUND=$round
TAG=$tag
PERF_FREQ=$perf_freq
KEEP_PERF_ENGINE_DATA=$KEEP_PERF_ENGINE_DATA
STARTED_AT=$(date -Iseconds)
EOF

perf record -F "$perf_freq" -g -p "$engine_pid" -o "$out_dir/perf_engine.data" -- sleep "$duration_sec" || rc=$?
if [[ -f "$out_dir/perf_engine.data" ]]; then
    perf report -i "$out_dir/perf_engine.data" --stdio --sort comm,dso,symbol > "$out_dir/perf_engine.report.txt" || true
    if [[ "$KEEP_PERF_ENGINE_DATA" != "1" ]]; then
        rm -f "$out_dir/perf_engine.data"
    fi
fi
exit "$rc"
