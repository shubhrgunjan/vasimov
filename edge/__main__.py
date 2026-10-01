#!/usr/bin/env python3
"""
vasimov/edge/__main__.py
CLI Entrypoint for Virtual Asimov Edge.

Usage:
    python -m vasimov.edge --transport udp [--realtime] [--fast] [--viewer]
"""

from __future__ import annotations
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

# Add vasimov to sys.path
_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

from edge.core import EdgeCore
from edge.ground_truth import GroundTruthServer
from edge.sim import SimBackend, SimControlServer
from edge.transports.udp_transport import UdpEdgeTransport, COMMAND_PORT, STATE_PORT


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Virtual Asimov Edge: Local Asimov protocol emulator in MuJoCo."
    )
    parser.add_argument(
        "--transport",
        choices=["udp"],
        default="udp",
        help="Transport protocol to expose (default: udp)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=True,
        help="Run physics with wall-clock pacing (default: True)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run simulation as fast as possible (disables realtime pacing)",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch MuJoCo passive viewer window (requires mjpython)",
    )
    parser.add_argument(
        "--command-port",
        type=int,
        default=COMMAND_PORT,
        help=f"UDP command port to bind (default: {COMMAND_PORT})",
    )
    parser.add_argument(
        "--state-port",
        type=int,
        default=STATE_PORT,
        help=f"UDP state port to send to (default: {STATE_PORT})",
    )
    parser.add_argument(
        "--state-host",
        type=str,
        default="127.0.0.1",
        help="Destination IP address for UDP telemetry (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8852,
        help="Port for local HTTP JSON control interface and dashboard (default: 8852)",
    )
    parser.add_argument(
        "--ground-truth-port",
        type=int,
        default=8854,
        help="Port for WebSocket ground-truth stream (default: 8854)",
    )
    parser.add_argument(
        "--enable-l1",
        action="store_true",
        help="Enable Layer 1 velocity-dependent torque derating in motor model",
    )
    parser.add_argument(
        "--no-gantry",
        action="store_true",
        help="Disable automatic virtual gantry release",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(name)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("vasimov.edge")

    log.info("=" * 70)
    log.info("STARTING VIRTUAL ASIMOV EDGE (MUJOCO SIMULATION)")
    log.info("Transport: %s | Realtime: %s", args.transport, not args.fast)
    log.info("Commands: UDP :%d | Telemetry: UDP %s:%d", args.command_port, args.state_host, args.state_port)
    log.info("Dashboard: http://127.0.0.1:%d | Ground Truth: ws://127.0.0.1:%d", args.control_port, args.ground_truth_port)
    log.info("=" * 70)

    # 1. Initialize Edge Core
    core = EdgeCore()

    # 2. Start Ground Truth Server (sim-only 50 Hz WebSocket stream)
    ground_truth_server = GroundTruthServer(ws_port=args.ground_truth_port)
    ground_truth_server.start()

    # 3. Initialize Sim Backend
    backend = SimBackend(
        core=core,
        auto_gantry=not args.no_gantry,
        enable_l1=args.enable_l1,
    )
    backend.ground_truth_server = ground_truth_server

    # 4. Start JSON Control Server & Dashboard
    control_server = SimControlServer(backend=backend, port=args.control_port)
    control_server.start()

    # 5. Start Transport
    if args.transport == "udp":
        transport = UdpEdgeTransport(
            core=core,
            backend=backend,
            command_port=args.command_port,
            state_host=args.state_host,
            state_port=args.state_port,
        )
        transport.start()
    else:
        log.error("Unsupported transport '%s'", args.transport)
        return 1

    # Viewer support
    viewer = None
    if args.viewer:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(backend.model, backend.data)
            log.info("[VIEWER] Launched MuJoCo interactive passive viewer")
        except Exception as e:
            log.warning("[VIEWER] Could not launch viewer: %s", e)

    # Shutdown flag
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        log.info("Caught shutdown signal; stopping Virtual Edge...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    sim_dt = backend.dt
    realtime = not args.fast

    log.info("[LOOP] Physics running at %.1f Hz (dt=%.4fs)", 1.0 / sim_dt, sim_dt)

    try:
        while running:
            step_start = time.perf_counter()
            backend.step()

            if viewer is not None and viewer.is_running():
                viewer.sync()
            elif viewer is not None and not viewer.is_running():
                break

            if realtime:
                step_elapsed = time.perf_counter() - step_start
                sleep_s = sim_dt - step_elapsed
                if sleep_s > 0:
                    time.sleep(sleep_s)
    finally:
        transport.stop()
        control_server.stop()
        ground_truth_server.stop()
        if viewer is not None and viewer.is_running():
            viewer.close()
        log.info("Virtual Asimov Edge stopped cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
