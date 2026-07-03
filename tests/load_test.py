import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


DEFAULT_PAYLOAD = {
    "amount": 50000,
    "payments": 1000,
    "A4": 50000,
    "A15": 250,
    "A16": 300,
}


def percentile(values, p):
    if not values:
        return None

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def send_request(url, timeout):
    start = time.perf_counter()

    try:
        response = requests.post(url, json=DEFAULT_PAYLOAD, timeout=timeout)
        latency = time.perf_counter() - start

        return {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "latency": latency,
            "error": None,
        }

    except Exception as exc:
        latency = time.perf_counter() - start

        return {
            "success": False,
            "status_code": None,
            "latency": latency,
            "error": str(exc),
        }


def run_load_test(url, concurrency, requests_count, timeout):
    start_time = time.perf_counter()

    results = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_request, url, timeout)
            for _ in range(requests_count)
        ]

        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.perf_counter() - start_time

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    latencies = [r["latency"] for r in successful]

    summary = {
        "url": url,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "concurrency": concurrency,
        "total_requests": requests_count,
        "successful_requests": len(successful),
        "failed_requests": len(failed),
        "total_time_seconds": round(total_time, 4),
        "throughput_requests_per_second": round(requests_count / total_time, 4) if total_time > 0 else None,
        "latency_seconds": {
            "average": round(statistics.mean(latencies), 4) if latencies else None,
            "p50": round(percentile(latencies, 50), 4) if latencies else None,
            "p95": round(percentile(latencies, 95), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "sub_2_second_p95": bool(latencies and percentile(latencies, 95) < 2.0),
        "sample_errors": [r["error"] for r in failed[:5] if r["error"]],
    }

    return summary


def append_report(summary):
    report_path = REPORTS_DIR / "load_test_results.md"

    status = "PASS" if summary["sub_2_second_p95"] else "FAIL"

    content = f"""
## Load Test Run - {summary["timestamp"]}

| Metric | Value |
|---|---:|
| URL | `{summary["url"]}` |
| Concurrency | {summary["concurrency"]} |
| Total requests | {summary["total_requests"]} |
| Successful requests | {summary["successful_requests"]} |
| Failed requests | {summary["failed_requests"]} |
| Total time seconds | {summary["total_time_seconds"]} |
| Throughput requests/sec | {summary["throughput_requests_per_second"]} |
| Average latency seconds | {summary["latency_seconds"]["average"]} |
| p50 latency seconds | {summary["latency_seconds"]["p50"]} |
| p95 latency seconds | {summary["latency_seconds"]["p95"]} |
| Max latency seconds | {summary["latency_seconds"]["max"]} |
| p95 < 2 seconds | {status} |

"""

    if summary["sample_errors"]:
        content += "Sample errors:\n\n"
        for error in summary["sample_errors"]:
            content += f"- `{error}`\n"
        content += "\n"

    if report_path.exists():
        existing = report_path.read_text(encoding="utf-8")
    else:
        existing = "# Load Test Results\n\n"

    report_path.write_text(existing + content, encoding="utf-8")

    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=10.0)

    args = parser.parse_args()

    summary = run_load_test(
        url=args.url,
        concurrency=args.concurrency,
        requests_count=args.requests,
        timeout=args.timeout,
    )

    report_path = append_report(summary)

    print(summary)
    print(f"Saved report to {report_path}")

    if summary["sub_2_second_p95"]:
        print("PASS: p95 latency is under 2 seconds.")
    else:
        print("FAIL: p95 latency is not under 2 seconds.")


if __name__ == "__main__":
    main()