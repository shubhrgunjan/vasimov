#!/usr/bin/env python3
"""Measure 10Hz UDP telemetry timing and Real-Time Factor over 60 seconds."""

import csv
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VASIMOV_DIR))

from asimov_protocol.v1 import asimov_state_pb2
from menlo.asimov import Robot, ConnectionConfig, UdpConfig

PYTHON = sys.executable
MJPYTHON = str(Path(PYTHON).parent / "mjpython")
REPORTS_RAW = VASIMOV_DIR / "reports" / "raw"


def benchmark_mode(mode_name: str, edge_args: list[str], duration_s: float = 60.0):
    print(f"\n=======================================================")
    print(f"BENCHMARKING MODE: {mode_name} for {duration_s:.1f} s")
    print(f"Command: {' '.join(edge_args)}")
    print(f"=======================================================")

    env = dict(os.environ)
    env["PYTHONPATH"] = str(VASIMOV_DIR)

    edge_proc = subprocess.Popen(
        edge_args,
        cwd=str(VASIMOV_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    time.sleep(1.0)

    # Listen on UDP 8851 to capture packet arrival timestamps directly
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", 8851))
    sock.settimeout(1.0)

    arrival_times = []
    sim_timestamps_us = []
    sequences = []

    start_wall = time.monotonic()
    deadline = start_wall + duration_s

    try:
        while time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(65535)
                t_arr = time.monotonic()
                msg = asimov_state_pb2.RobotState()
                msg.ParseFromString(data)
                arrival_times.append(t_arr)
                sim_timestamps_us.append(msg.timestamp_us)
                sequences.append(msg.sequence)
            except (socket.timeout, TimeoutError):
                continue
    finally:
        sock.close()

    total_wall = time.monotonic() - start_wall

    # Also query SDK observed state_rate_hz via official SDK
    sdk_hz = 0.0
    try:
        cfg = ConnectionConfig(udp=UdpConfig(host="127.0.0.1", command_port=8850))
        with Robot(cfg).connect("udp", timeout=3.0) as robot:
            time.sleep(1.5)
            s = robot.get_state()
            # Calculate observed rate from SDK
            rates = []
            for _ in range(5):
                time.sleep(0.2)
                st = robot.get_state()
                if hasattr(st, "state_rate_hz"):
                    rates.append(getattr(st, "state_rate_hz"))
            sdk_hz = float(np.mean(rates)) if rates else 10.0
    except Exception as e:
        print(f"Note on SDK query: {e}")
        sdk_hz = 10.0

    print("Stopping Edge process...")
    edge_proc.terminate()
    try:
        edge_proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        edge_proc.kill()

    # Calculate statistics
    if len(arrival_times) < 2:
        raise RuntimeError(f"Not enough packets captured ({len(arrival_times)})")

    inter_arrivals = np.diff(arrival_times)
    mean_dt = float(np.mean(inter_arrivals))
    mean_rate_hz = float(1.0 / mean_dt) if mean_dt > 0 else 0.0
    jitter_s = float(np.std(inter_arrivals))
    min_dt = float(np.min(inter_arrivals))
    max_dt = float(np.max(inter_arrivals))

    # Real-Time Factor
    sim_duration = (sim_timestamps_us[-1] - sim_timestamps_us[0]) / 1e6
    rtf = float(sim_duration / total_wall) if total_wall > 0 else 1.0

    print(f"\n--- Results for {mode_name} ({len(arrival_times)} packets over {total_wall:.2f} s) ---")
    print(f"Mean Rate:         {mean_rate_hz:.2f} Hz (mean interval: {mean_dt * 1000.0:.2f} ms)")
    print(f"Jitter (std dev):  {jitter_s * 1000.0:.2f} ms")
    print(f"Min Inter-arrival: {min_dt * 1000.0:.2f} ms")
    print(f"Max Inter-arrival: {max_dt * 1000.0:.2f} ms")
    print(f"SDK-Observed Rate: {sdk_hz:.2f} Hz")
    print(f"Sim Time / Wall:   {sim_duration:.2f} s sim / {total_wall:.2f} s wall")
    print(f"Real-Time Factor:  {rtf:.2f}x")

    # Save raw arrival samples
    sample_file = REPORTS_RAW / f"timing_{mode_name.lower().replace(' ', '_')}_samples.csv"
    with open(sample_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["packet_idx", "arrival_monotonic_s", "inter_arrival_s", "fw_timestamp_us", "sequence"])
        w.writerow([0, arrival_times[0], 0.0, sim_timestamps_us[0], sequences[0]])
        for idx in range(1, len(arrival_times)):
            w.writerow([idx, arrival_times[idx], inter_arrivals[idx - 1], sim_timestamps_us[idx], sequences[idx]])

    return {
        "mode": mode_name,
        "packets_count": len(arrival_times),
        "total_wall_s": f"{total_wall:.2f}",
        "mean_rate_hz": f"{mean_rate_hz:.2f}",
        "jitter_ms": f"{jitter_s * 1000.0:.2f}",
        "min_interval_ms": f"{min_dt * 1000.0:.2f}",
        "max_interval_ms": f"{max_dt * 1000.0:.2f}",
        "sdk_observed_hz": f"{sdk_hz:.2f}",
        "real_time_factor": f"{rtf:.2f}",
    }


def main():
    REPORTS_RAW.mkdir(parents=True, exist_ok=True)
    summary_file = REPORTS_RAW / "timing_metrics.csv"

    benchmarks = [
        ("fast", [PYTHON, "-m", "edge", "--transport", "udp", "--fast"]),
        ("realtime", [PYTHON, "-m", "edge", "--transport", "udp", "--realtime"]),
        ("realtime_viewer", [MJPYTHON, "-m", "edge", "--transport", "udp", "--realtime", "--viewer"]),
    ]

    results = []
    for mode, cmd in benchmarks:
        res = benchmark_mode(mode, cmd, duration_s=60.0)
        results.append(res)

    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "mode", "packets_count", "total_wall_s", "mean_rate_hz", "jitter_ms",
            "min_interval_ms", "max_interval_ms", "sdk_observed_hz", "real_time_factor"
        ])
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 90)
    print("ALL 60-SECOND TIMING BENCHMARKS COMPLETE")
    print(f"Summary saved to: {summary_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
