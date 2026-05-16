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
COLLECTOR=pidstat
OUT_DIR=$out_dir
API_PID=$api_pid
ENGINE_PID=$engine_pid
DURATION_SEC=$duration_sec
ROUND=$round
TAG=$tag
STARTED_AT=$(date -Iseconds)
EOF

timeout "$duration_sec" pidstat -t -p "$api_pid,$engine_pid" 1 > "$out_dir/pidstat_threads.txt" || rc=$?
if [[ "$rc" == "124" ]]; then
    rc=0
fi
exit "$rc"
