#!/usr/bin/env bash
set -Eeuo pipefail

out_dir=${1:?output dir required}
mkdir -p "$out_dir"

nvidia-smi \
    --query-gpu=timestamp,pstate,temperature.gpu,power.draw,clocks.gr,clocks.mem,utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current \
    --format=csv \
    -l 1 > "$out_dir/gpu_watch.csv"

