#!/usr/bin/env python3
import json
import re
import sys
from pathlib import Path


PATTERNS = {
    "successful_requests": r"Successful requests:\s*([0-9.]+)",
    "failed_requests": r"Failed requests:\s*([0-9.]+)",
    "benchmark_duration_s": r"Benchmark duration \(s\):\s*([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s*([0-9.]+)",
    "median_ttft_ms": r"Median TTFT \(ms\):\s*([0-9.]+)",
    "p99_ttft_ms": r"P99 TTFT \(ms\):\s*([0-9.]+)",
    "mean_tpot_ms": r"Mean TPOT \(ms\):\s*([0-9.]+)",
    "median_tpot_ms": r"Median TPOT \(ms\):\s*([0-9.]+)",
    "p99_tpot_ms": r"P99 TPOT \(ms\):\s*([0-9.]+)",
    "mean_itl_ms": r"Mean ITL \(ms\):\s*([0-9.]+)",
    "median_itl_ms": r"Median ITL \(ms\):\s*([0-9.]+)",
    "p99_itl_ms": r"P99 ITL \(ms\):\s*([0-9.]+)",
}


def parse(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    result: dict[str, float] = {}
    for key, pattern in PATTERNS.items():
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            if value.is_integer():
                result[key] = int(value)
            else:
                result[key] = value
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_bench.py BENCH_LOG", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    result = parse(path)
    print(json.dumps(result, sort_keys=True))
    return 0 if result else 1


if __name__ == "__main__":
    raise SystemExit(main())

