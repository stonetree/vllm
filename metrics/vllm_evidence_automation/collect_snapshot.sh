#!/usr/bin/env bash
set -Eeuo pipefail

tag=${1:?tag required}
api_pid=${2:?api pid required}
engine_pid=${3:?engine pid required}
out_dir=${4:?output dir required}

mkdir -p "$out_dir"

collect_one_pid() {
    local name=$1
    local pid=$2
    local file="$out_dir/${tag}.${name}.threads.tsv"

    {
        printf 'tid\tcomm\tallowed\tpsr\tpcpu\tvoluntary_ctxt\tnonvoluntary_ctxt\n'
        for tid in $(ls "/proc/$pid/task" 2>/dev/null || true); do
            local comm allowed vol nonvol psr pcpu
            comm=$(cat "/proc/$pid/task/$tid/comm" 2>/dev/null || true)
            allowed=$(awk '/Cpus_allowed_list/ {print $2}' "/proc/$pid/task/$tid/status" 2>/dev/null || true)
            vol=$(awk '/voluntary_ctxt_switches/ {print $2}' "/proc/$pid/task/$tid/status" 2>/dev/null || true)
            nonvol=$(awk '/nonvoluntary_ctxt_switches/ {print $2}' "/proc/$pid/task/$tid/status" 2>/dev/null || true)
            psr=$(ps -T -p "$pid" -o spid,psr 2>/dev/null | awk -v t="$tid" '$1==t {print $2}')
            pcpu=$(ps -T -p "$pid" -o spid,pcpu 2>/dev/null | awk -v t="$tid" '$1==t {print $2}')
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tid" "$comm" "$allowed" "$psr" "$pcpu" "$vol" "$nonvol"
        done
    } > "$file"
}

collect_one_pid API "$api_pid"
collect_one_pid ENGINE "$engine_pid"

ps -eLo pid,tid,psr,pcpu,comm 2>/dev/null \
    | grep -E 'vllm|VLLM|ZMQ|cuda|pt_tcpstore|iou|python' \
    | sort -k4 -nr > "$out_dir/${tag}.system_threads.sorted.txt" || true

ps -T -p "$api_pid" -o spid,psr,pcpu,comm > "$out_dir/${tag}.api.ps_threads.txt" 2>&1 || true
ps -T -p "$engine_pid" -o spid,psr,pcpu,comm > "$out_dir/${tag}.engine.ps_threads.txt" 2>&1 || true

numastat -p "$api_pid" > "$out_dir/${tag}.api.numastat.txt" 2>&1 || true
numastat -p "$engine_pid" > "$out_dir/${tag}.engine.numastat.txt" 2>&1 || true

cat "/proc/$api_pid/numa_maps" > "$out_dir/${tag}.api.numa_maps.txt" 2>/dev/null || true
cat "/proc/$engine_pid/numa_maps" > "$out_dir/${tag}.engine.numa_maps.txt" 2>/dev/null || true
cat "/proc/$api_pid/smaps_rollup" > "$out_dir/${tag}.api.smaps_rollup.txt" 2>/dev/null || true
cat "/proc/$engine_pid/smaps_rollup" > "$out_dir/${tag}.engine.smaps_rollup.txt" 2>/dev/null || true
cat /proc/interrupts > "$out_dir/${tag}.interrupts.txt" 2>/dev/null || true
nvidia-smi -q -d PERFORMANCE,CLOCK,POWER,TEMPERATURE > "$out_dir/${tag}.gpu_q.txt" 2>&1 || true

printf 'snapshot done: %s\n' "$tag"

