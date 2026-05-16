#!/usr/bin/env bash
set -Eeuo pipefail

# Interface: OUT_DIR API_PID ENGINE_PID DURATION_SEC ROUND TAG
out_dir=${1:?OUT_DIR required}
api_pid=${2:?API_PID required}
engine_pid=${3:?ENGINE_PID required}
duration_sec=${4:?DURATION_SEC required}
round=${5:?ROUND required}
tag=${6:?TAG required}

mkdir -p "$out_dir"
exit_code_file="$out_dir/exit_code"
rc=0
trap 'rc=$?; printf "%s\n" "$rc" > "$exit_code_file"' EXIT

cat > "$out_dir/meta.env" <<EOF
COLLECTOR=perf_sched
OUT_DIR=$out_dir
API_PID=$api_pid
ENGINE_PID=$engine_pid
DURATION_SEC=$duration_sec
ROUND=$round
TAG=$tag
STARTED_AT=$(date -Iseconds)
EOF

perf sched record -a -o "$out_dir/perf_sched.data" -- sleep "$duration_sec" || rc=$?
if [[ -f "$out_dir/perf_sched.data" ]]; then
    perf sched latency -i "$out_dir/perf_sched.data" > "$out_dir/perf_sched_latency.txt" || true
fi
exit "$rc"
