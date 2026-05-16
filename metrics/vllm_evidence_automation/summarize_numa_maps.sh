#!/usr/bin/env bash
set -Eeuo pipefail

file=${1:?numa_maps file required}

awk '
{
    for (i=1; i<=NF; i++) {
        if ($i ~ /^N[0-9]+=/) {
            split($i, a, "=")
            sum[a[1]] += a[2]
        }
    }
}
END {
    for (n in sum) {
        printf "%s %d pages %.2f MB\n", n, sum[n], sum[n] * 4 / 1024
    }
}
' "$file" | sort

