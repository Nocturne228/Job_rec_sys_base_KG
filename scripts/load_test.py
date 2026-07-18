#!/usr/bin/env python3
"""Dependency-free concurrent API smoke/load test with percentile reporting."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post(url: str, payload: dict, token: str | None = None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers=headers, method="POST"
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()
        status = response.status
    return status, (time.perf_counter() - start) * 1000


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]


def run(base_url: str, requests: int, concurrency: int):
    _, _ = post(
        f"{base_url}/api/token", {"username": "user_001", "password": "jobrec-demo"}
    )
    token_request = urllib.request.Request(
        f"{base_url}/api/token",
        data=json.dumps({"username": "user_001", "password": "jobrec-demo"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(token_request) as response:
        token = json.loads(response.read())["access_token"]
    payload = {"user_id": "user_001", "resume_text": "Python SQL Docker"}
    latencies, errors = [], 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(post, f"{base_url}/api/recommend", payload, token)
            for _ in range(requests)
        ]
        for future in as_completed(futures):
            try:
                status, latency = future.result()
                latencies.append(latency)
                errors += int(status >= 400)
            except Exception:
                errors += 1
    elapsed = time.perf_counter() - started
    return {
        "requests": requests,
        "concurrency": concurrency,
        "throughput_rps": requests / elapsed,
        "error_rate": errors / requests,
        "mean_ms": statistics.mean(latencies) if latencies else 0,
        "p50_ms": percentile(latencies, 0.50) if latencies else 0,
        "p95_ms": percentile(latencies, 0.95) if latencies else 0,
        "p99_ms": percentile(latencies, 0.99) if latencies else 0,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.base_url, args.requests, args.concurrency), indent=2))
