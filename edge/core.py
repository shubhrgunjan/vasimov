#!/usr/bin/env python3
"""
vasimov/edge/core.py
Virtual Asimov Edge Core: State Machine, Controller Arbiter, Drop Rules & Telemetry Builder.

Pure logic module, zero network I/O, completely unit-testable.

SPECIFICATIONS & VERIFIED FACTS:
- Modes:
    * DAMP (boot default; entered on fall, overtemp, or 2 s trajectory timeout)
    * STAND (linear ramp from current pose to default pose over 2.0 s, ASSUMED)
    * MOVE / TRAJECTORY (direct PD streaming)
    * MOVE / POLICY (NoPolicy stub: logs loudly and stays in STAND; does not walk)
    * FAULT_DAMP (latched emergency damping; cleared only by virtual restart)
- Transitions:
    * any -> DAMP: always allowed
    * DAMP -> STAND: allowed if NOT fault-latched
    * STAND -> MOVE/TRAJECTORY: allowed via trajectory command
    * MOVE -> STAND: allowed via STAND command
    * DAMP -> MOVE: DROPPED SILENTLY (no DAMP->MOVE allowed)
- Drop Rules:
    * Non-finite velocity or trajectory positions (NaN / Inf)
    * Trajectory positions length != 25
    * Empty segments or unsupported source
    * STAND while fault-latched
    * Any command from a non-active controller
- Enum Numbering:
    * ControlMode: DAMP=0, STAND=1, MOVE=2, FAULT_DAMP=5
    * Command Mode (Edge Cloud): STAND=0, DAMP=1
    * Command Mode (asimov.io): STAND=1, DAMP=0
- Safety Faults:
    * Fall: projected gravity z > -0.5 (tilt > 60 deg), latches FAULT_DAMP, error_flags = 0x101
    * Overtemp: trips at 80 C (latches FAULT_DAMP), clears at 70 C hysteresis
    * CAN errors: synthetic CAN counter diagnostics (~1 s)
"""

from __future__ import annotations
import enum
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import yaml

from asimov_protocol.v1 import (
    asimov_command_pb2,
    asimov_common_pb2,
    asimov_state_pb2,
    edge_cloud_pb2,
)
from asimov_protocol.alerts import AlertId, AlertSeverity
from adapter import JointAdapter, SIM_JOINTS, FIRMWARE_JOINTS

log = logging.getLogger("vasimov.edge.core")


# ── Modes & Constants ────────────────────────────────────────────────────────
class EdgeMode(enum.IntEnum):
    """Firmware / Edge control modes matching asimov.io.ControlMode."""
    DAMP = asimov_common_pb2.CONTROL_MODE_DAMP          # 0
    STAND = asimov_common_pb2.CONTROL_MODE_STAND        # 1
    MOVE = asimov_common_pb2.CONTROL_MODE_MOVE          # 2
    FAULT_DAMP = asimov_common_pb2.CONTROL_MODE_FAULT_DAMP  # 5


class MoveSubmode(enum.IntEnum):
    NONE = 0
    TRAJECTORY = 1
    POLICY = 2


STAND_RAMP_DURATION_S = 2.0
TRAJECTORY_WATCHDOG_S = 2.0
VELOCITY_HOLD_TIMEOUT_S = 2.0
FALL_GRAVITY_Z_THRESHOLD = -0.50  # Tilt > 60 deg trips fall latch [VERIFIED]
OVERTEMP_TRIP_C = 80.0             # Latches FAULT_DAMP [VERIFIED]
OVERTEMP_CLEAR_C = 70.0            # Hysteresis clearance [VERIFIED]
TEMP_HIGH_WARN_C = 60.0            # Warning alert [VERIFIED]


# ── Edge Core ────────────────────────────────────────────────────────────────
class EdgeCore:
    """
    Virtual Asimov Edge state machine, safety monitor, and telemetry generator.
    """

    def __init__(
        self,
        gains_yaml_path: Optional[Union[str, Path]] = None,
        default_controller: str = "sdk",
    ):
        base_dir = Path(__file__).resolve().parent.parent
        if gains_yaml_path is None:
            gains_yaml_path = base_dir / "config" / "gains.yaml"
        self.gains_yaml_path = Path(gains_yaml_path)

        # Load gains and default pose
        with open(self.gains_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.default_pose_sim = tuple(
            float(cfg["default_pose"]["joints"][name]) for name in SIM_JOINTS
        )
        self.default_pose_fw = JointAdapter.sim_to_firmware_positions(self.default_pose_sim)

        # Extract per-joint default gains
        gains_cfg = cfg.get("gains", {})
        self.default_kp = tuple(
            float(gains_cfg[name]["kp"]) for name in SIM_JOINTS
        )
        self.default_kd = tuple(
            float(gains_cfg[name]["kd"]) for name in SIM_JOINTS
        )
        self.damp_kd = tuple(2.0 for _ in range(25))  # ASSUMED small compliance damping

        # Arbiter
        self.active_controller: str = default_controller

        # State machine
        self.mode: EdgeMode = EdgeMode.DAMP
        self.move_submode: MoveSubmode = MoveSubmode.NONE
        self.boot_time_us: int = int(time.time() * 1_000_000)
        self.sequence: int = 0

        # STAND ramp tracking
        self.stand_ramp_start_time: float = 0.0
        self.stand_ramp_start_pos: Tuple[float, ...] = self.default_pose_sim
        self.stand_settled: bool = False

        # TRAJECTORY tracking
        self.last_trajectory_time: float = 0.0
        self.current_trajectory_targets: Tuple[float, ...] = self.default_pose_sim
        self.current_trajectory_kp: Tuple[float, ...] = self.default_kp
        self.current_trajectory_kd: Tuple[float, ...] = self.default_kd

        # Velocity tracking
        self.last_velocity_time: float = 0.0
        self.current_vx: float = 0.0
        self.current_vy: float = 0.0
        self.current_vyaw: float = 0.0

        # POLICY tracking
        self.current_policy_targets: Tuple[float, ...] = self.default_pose_sim
        self.current_policy_kp: Tuple[float, ...] = self.default_kp
        self.current_policy_kd: Tuple[float, ...] = self.default_kd

        # Safety & Fault Latching
        self.fault_latched: bool = False
        self.fault_fall: bool = False
        self.fault_overtemp: bool = False
        self.fault_can: bool = False
        self.fall_latch_enabled: bool = True  # When False, direct policy runs without forcing FAULT_DAMP
        self.error_flags: int = 0
        self.temperatures: List[float] = [25.0] * 25  # PLACEHOLDER 25.0 C

        # Virtual Battery (PLACEHOLDER)
        self.battery_enabled: bool = True
        self.battery_soc: float = 100.0         # State of Charge % (default: 100.0)
        self.battery_voltage: float = 48.0      # Nominal pack voltage (default: 48.0 V)
        self.battery_temp_c: float = 25.0       # Cell temperature (default: 25.0 C)
        self.battery_discharge_rate: float = 0.0  # Linear discharge rate in %/s (default: 0.0 OFF)
        self.battery_protection_flags: int = 0  # BMS protection bits (default: 0)

        # Event tracking
        self.pending_events: List[edge_cloud_pb2.EdgeEvent] = []
        self.last_diag_time: float = 0.0
        self.sim_time: float = 0.0

    # ── State Machine Transitions ────────────────────────────────────────────
    def command_stand(
        self,
        controller: str = "sdk",
        current_sim_pos: Optional[Sequence[float]] = None,
        current_time: Optional[float] = None,
    ) -> bool:
        """Handle STAND request."""
        if controller != self.active_controller:
            log.warning("STAND command rejected: inactive controller '%s' (active: '%s')", controller, self.active_controller)
            return False

        if self.fault_latched:
            log.warning("STAND command rejected: firmware latched in FAULT_DAMP (error_flags=0x%x)", self.error_flags)
            return False

        if self.mode == EdgeMode.STAND:
            return True

        # Enter STAND: start 2.0s linear ramp to default standing pose
        now = current_time if current_time is not None else (self.sim_time if self.sim_time > 0 else time.monotonic())
        self.mode = EdgeMode.STAND
        self.move_submode = MoveSubmode.NONE
        self.stand_ramp_start_time = now
        self.stand_settled = False
        if current_sim_pos is not None and len(current_sim_pos) == 25:
            self.stand_ramp_start_pos = tuple(float(x) for x in current_sim_pos)
        else:
            self.stand_ramp_start_pos = self.default_pose_sim

        log.info("Entered STAND: ramping to default standing pose over %.1fs", STAND_RAMP_DURATION_S)
        return True

    def command_damp(self, controller: str = "sdk") -> bool:
        """Handle DAMP request."""
        if controller != self.active_controller:
            log.warning("DAMP command rejected: inactive controller '%s'", controller)
            return False

        self.mode = EdgeMode.DAMP
        self.move_submode = MoveSubmode.NONE
        self.stand_settled = False
        log.info("Entered DAMP: actuators compliant")
        return True

    def command_trajectory(
        self,
        positions: Sequence[float],
        kp: Optional[Sequence[float]] = None,
        kd: Optional[Sequence[float]] = None,
        controller: str = "sdk",
        current_time: Optional[float] = None,
    ) -> bool:
        """Handle Trajectory command."""
        if controller != self.active_controller:
            log.debug("Trajectory dropped: inactive controller '%s'", controller)
            return False

        # Drop rule: Trajectory length must be exactly 25
        if len(positions) != 25:
            log.warning("Trajectory dropped: length %d != 25", len(positions))
            return False

        # Drop rule: Non-finite values
        if not all(math.isfinite(x) for x in positions):
            log.warning("Trajectory dropped: contains non-finite values")
            return False

        # Drop rule: In DAMP, trajectories are dropped silently (no DAMP->MOVE)
        if self.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
            log.debug("Trajectory dropped: robot is in DAMP / FAULT_DAMP (stand() it first)")
            return False

        # Transition to MOVE/TRAJECTORY
        now = current_time if current_time is not None else (self.sim_time if self.sim_time > 0 else time.monotonic())
        self.mode = EdgeMode.MOVE
        self.move_submode = MoveSubmode.TRAJECTORY
        self.last_trajectory_time = now

        # Convert wire positions to sim positions (ankles are already pitch/roll on wire)
        self.current_trajectory_targets = JointAdapter.wire_trajectory_to_sim_positions(positions)

        # Gains: must have exactly 25 or edge defaults are used
        if kp is not None and len(kp) == 25 and all(math.isfinite(x) for x in kp):
            self.current_trajectory_kp = tuple(float(x) for x in kp)
        else:
            self.current_trajectory_kp = self.default_kp

        if kd is not None and len(kd) == 25 and all(math.isfinite(x) for x in kd):
            self.current_trajectory_kd = tuple(float(x) for x in kd)
        else:
            self.current_trajectory_kd = self.default_kd

        return True

    def command_velocity(
        self,
        vx: float,
        vy: float,
        vyaw: float,
        controller: Union[str, float] = "sdk",
        current_time: Optional[float] = None,
    ) -> bool:
        """Handle Velocity command: transitions STAND -> MOVE/POLICY with contract clipping."""
        if isinstance(controller, (int, float)):
            current_time = float(controller)
            controller = "sdk"

        if controller != self.active_controller:
            log.debug("Velocity dropped: inactive controller '%s'", controller)
            return False

        # Drop rule: Non-finite values
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vyaw)):
            log.warning("Velocity dropped: non-finite velocity command (%s, %s, %s)", vx, vy, vyaw)
            return False

        # Drop rule: In DAMP, velocity is dropped silently
        if self.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
            log.debug("Velocity dropped: robot in DAMP / FAULT_DAMP")
            return False

        from edge.policy import PolicyController
        cvx, cvy, cvyaw = PolicyController.clip_velocity_command(vx, vy, vyaw)

        # Transition to MOVE/POLICY
        now = self.sim_time if self.sim_time > 0 else time.monotonic()
        self.mode = EdgeMode.MOVE
        self.move_submode = MoveSubmode.POLICY
        self.current_vx = cvx
        self.current_vy = cvy
        self.current_vyaw = cvyaw
        self.last_velocity_time = now
        log.info(
            "[POLICY] Velocity command accepted: (vx=%.2f, vy=%.2f, vyaw=%.2f) -> Mode MOVE/POLICY",
            cvx, cvy, cvyaw
        )
        return True

    # ── Watchdogs & Safety Monitor ───────────────────────────────────────────
    def step_state(self, current_time: float, imu_gravity: Sequence[float]) -> None:
        """
        Step timers, watchdogs, and safety checks on every control tick.
        """
        dt = max(0.0, current_time - self.sim_time) if self.sim_time > 0.0 else 0.0
        self.sim_time = current_time

        # Virtual Battery linear discharge (PLACEHOLDER)
        if self.battery_discharge_rate > 0.0 and dt > 0.0:
            self.battery_soc = max(0.0, self.battery_soc - self.battery_discharge_rate * dt)
        # 1. Fall detection check
        if len(imu_gravity) == 3:
            gz = float(imu_gravity[2])
            if gz > FALL_GRAVITY_Z_THRESHOLD:  # gz > -0.5 (tilt > 60 deg)
                self.fault_fall = True
                if self.fall_latch_enabled:
                    if not self.fault_latched:
                        log.error(
                            "[SAFETY FAULT] Fall detected: projected gravity z=%.2f > %.2f (tilt > 60 deg). "
                            "Latching FAULT_DAMP.",
                            gz, FALL_GRAVITY_Z_THRESHOLD
                        )
                        self.fault_latched = True
                        self.mode = EdgeMode.FAULT_DAMP
                        self.move_submode = MoveSubmode.NONE
                else:
                    log.debug("[POLICY] Tilt detected (gz=%.2f), fall_latch_enabled=False: continuing policy execution.", gz)
                self._recalculate_error_flags()
            else:
                if not self.fall_latch_enabled and self.fault_fall:
                    self.fault_fall = False
                    self._recalculate_error_flags()

        # 2. Overtemp hysteresis check
        max_temp = max(self.temperatures)
        if max_temp >= OVERTEMP_TRIP_C and not self.fault_overtemp:
            log.error("[SAFETY FAULT] Overtemp trip: joint temp %.1f C >= %.1f C. Latching FAULT_DAMP.", max_temp, OVERTEMP_TRIP_C)
            self.fault_overtemp = True
            self.fault_latched = True
            self.mode = EdgeMode.FAULT_DAMP
            self.move_submode = MoveSubmode.NONE
            self._recalculate_error_flags()
        elif max_temp < OVERTEMP_CLEAR_C and self.fault_overtemp:
            log.info("[SAFETY] Joint temperature dropped to %.1f C (< %.1f C). Self-clearing overtemp condition.", max_temp, OVERTEMP_CLEAR_C)
            self.fault_overtemp = False
            if not self.fault_fall:
                self.fault_latched = False
                if self.mode == EdgeMode.FAULT_DAMP:
                    self.mode = EdgeMode.DAMP
            self._recalculate_error_flags()

        # 3. Trajectory watchdog (2.0s silence in MOVE/TRAJECTORY -> auto-DAMP)
        if self.mode == EdgeMode.MOVE and self.move_submode == MoveSubmode.TRAJECTORY:
            if current_time - self.last_trajectory_time > TRAJECTORY_WATCHDOG_S:
                log.warning(
                    "[WATCHDOG] 2.0s trajectory silence in MOVE/TRAJECTORY (elapsed: %.2fs). Triggering auto-DAMP.",
                    current_time - self.last_trajectory_time
                )
                self.mode = EdgeMode.DAMP
                self.move_submode = MoveSubmode.NONE

        # 3b. Policy velocity staleness (2.0s silence in MOVE/POLICY -> zero-velocity hold)
        if self.mode == EdgeMode.MOVE and self.move_submode == MoveSubmode.POLICY:
            if current_time - self.last_velocity_time > VELOCITY_HOLD_TIMEOUT_S:
                if self.current_vx != 0.0 or self.current_vy != 0.0 or self.current_vyaw != 0.0:
                    log.info(
                        "[WATCHDOG] 2.0s velocity silence in MOVE/POLICY (elapsed: %.2fs). Holding zero-velocity balance.",
                        current_time - self.last_velocity_time
                    )
                    self.current_vx = 0.0
                    self.current_vy = 0.0
                    self.current_vyaw = 0.0

        # 4. Stand ramp progress
        if self.mode == EdgeMode.STAND and not self.stand_settled:
            elapsed = current_time - self.stand_ramp_start_time
            if elapsed >= STAND_RAMP_DURATION_S:
                self.stand_settled = True
                log.info("STAND ramp complete: holding default standing pose")

    def _recalculate_error_flags(self) -> None:
        """Update error_flags bitfield based on active critical alerts."""
        flags = 0
        if self.fault_latched:
            flags |= 1  # Bit 0 = latched fault
        if self.fault_fall:
            flags |= (1 << (AlertId.FALL_DETECTED + 1))  # Alert 7 -> bit 8
        if self.fault_overtemp:
            flags |= (1 << (AlertId.MOTOR_OVERTEMP + 1))  # Alert 2 -> bit 3
        if self.fault_can:
            flags |= (1 << (AlertId.CAN_BUS_OFF + 1))
        self.error_flags = flags

    # ── Fault Injection & Virtual Restart ────────────────────────────────────
    def inject_fall(self) -> None:
        """Inject fall fault (callable from tests/control)."""
        log.warning("[FAULT INJECTION] Injecting fall fault")
        self.fault_fall = True
        self.fault_latched = True
        self.mode = EdgeMode.FAULT_DAMP
        self.move_submode = MoveSubmode.NONE
        self._recalculate_error_flags()

    def inject_overtemp(self, temp_c: float = 85.0) -> None:
        """Inject overtemp fault."""
        log.warning("[FAULT INJECTION] Injecting overtemp (%.1f C)", temp_c)
        self.temperatures = [temp_c] * 25
        self.fault_overtemp = True
        self.fault_latched = True
        self.mode = EdgeMode.FAULT_DAMP
        self.move_submode = MoveSubmode.NONE
        self._recalculate_error_flags()

    def clear_overtemp(self, temp_c: float = 25.0) -> None:
        """Cool actuators down."""
        log.info("[FAULT CLEAR] Cooling actuators down to %.1f C", temp_c)
        self.temperatures = [temp_c] * 25
        self.fault_overtemp = False
        if not self.fault_fall:
            self.fault_latched = False
            if self.mode == EdgeMode.FAULT_DAMP:
                self.mode = EdgeMode.DAMP
        self._recalculate_error_flags()

    def inject_can_fault(self) -> None:
        """Inject CAN bus fault."""
        log.warning("[FAULT INJECTION] Injecting CAN bus off fault")
        self.fault_can = True
        self.fault_latched = True
        self.mode = EdgeMode.FAULT_DAMP
        self._recalculate_error_flags()

    def virtual_restart(self) -> None:
        """Virtual restart: clears all fault latches and returns to clean DAMP state."""
        log.info("[RESTART] Virtual firmware restart requested: clearing all fault latches")
        self.fault_latched = False
        self.fault_fall = False
        self.fault_overtemp = False
        self.fault_can = False
        self.error_flags = 0
        self.temperatures = [25.0] * 25
        self.battery_soc = 100.0
        self.battery_discharge_rate = 0.0
        self.battery_protection_flags = 0
        self.mode = EdgeMode.DAMP
        self.move_submode = MoveSubmode.NONE
        self.stand_settled = False
        self.current_policy_targets = self.default_pose_sim
        self.boot_time_us = int(time.time() * 1_000_000)

    def set_battery(
        self,
        soc: Optional[float] = None,
        discharge_rate: Optional[float] = None,
        voltage: Optional[float] = None,
        temp_c: Optional[float] = None,
        protection_flags: Optional[int] = None,
    ) -> None:
        """Configure virtual battery state (PLACEHOLDER)."""
        if soc is not None:
            self.battery_soc = max(0.0, min(100.0, float(soc)))
            log.info("[BATTERY] Virtual battery SoC set to %.1f%%", self.battery_soc)
        if discharge_rate is not None:
            self.battery_discharge_rate = max(0.0, float(discharge_rate))
            log.info("[BATTERY] Virtual battery discharge rate set to %.2f%%/s", self.battery_discharge_rate)
        if voltage is not None:
            self.battery_voltage = float(voltage)
        if temp_c is not None:
            self.battery_temp_c = float(temp_c)
        if protection_flags is not None:
            self.battery_protection_flags = int(protection_flags)

    # ── Controller Arbiter ───────────────────────────────────────────────────
    def set_active_controller(self, controller: str, reason: str = "request") -> None:
        """Switch active controller, recording an event."""
        prev = self.active_controller
        if prev != controller:
            self.active_controller = controller
            log.info("Controller change: '%s' -> '%s' (reason: %s)", prev, controller, reason)
            event = edge_cloud_pb2.EdgeEvent(
                timestamp_us=int(time.time() * 1_000_000),
                sequence=self.sequence,
                controller=edge_cloud_pb2.ControllerEvent(
                    previous=prev,
                    current=controller,
                    reason=reason,
                ),
            )
            self.pending_events.append(event)

    # ── Control Targets for Sim ──────────────────────────────────────────────
    def get_control_targets(
        self,
        current_sim_positions: Sequence[float],
        current_time: float,
    ) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, ...]]:
        """
        Compute (q_target, kp, kd) in SIMULATION joint coordinates for the motor model.
        """
        # 1. DAMP or FAULT_DAMP: kp = 0, compliance damping
        if self.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP):
            zero_kp = tuple(0.0 for _ in range(25))
            return tuple(current_sim_positions), zero_kp, self.damp_kd

        # 2. STAND: ramping or holding default standing pose
        if self.mode == EdgeMode.STAND:
            if not self.stand_settled:
                elapsed = max(0.0, current_time - self.stand_ramp_start_time)
                alpha = min(1.0, elapsed / STAND_RAMP_DURATION_S)
                # Linear blend between ramp start pose and default pose
                blended = tuple(
                    p0 + (p1 - p0) * alpha
                    for p0, p1 in zip(self.stand_ramp_start_pos, self.default_pose_sim, strict=True)
                )
                return blended, self.default_kp, self.default_kd
            else:
                return self.default_pose_sim, self.default_kp, self.default_kd

        # 3. MOVE / TRAJECTORY: stream targets directly
        if self.mode == EdgeMode.MOVE and self.move_submode == MoveSubmode.TRAJECTORY:
            return self.current_trajectory_targets, self.current_trajectory_kp, self.current_trajectory_kd

        # 4. MOVE / POLICY: stream policy targets
        if self.mode == EdgeMode.MOVE and self.move_submode == MoveSubmode.POLICY:
            return self.current_policy_targets, self.current_policy_kp, self.current_policy_kd

        # Fallback
        return self.default_pose_sim, self.default_kp, self.default_kd

    # ── Telemetry Builders ───────────────────────────────────────────────────
    def build_robot_state(
        self,
        sim_joint_pos: Sequence[float],
        sim_joint_vel: Sequence[float],
        imu_quat: Sequence[float],
        imu_gyro: Sequence[float],
        imu_gravity: Sequence[float],
        sim_torques: Optional[Sequence[float]] = None,
    ) -> asimov_state_pb2.RobotState:
        """
        Build an official asimov.io.RobotState message for UDP state datagrams (:8851).
        """
        self.sequence = (self.sequence + 1) & 0xFFFFFFFF
        now_us = int(time.time() * 1_000_000)
        fw_us = max(0, now_us - self.boot_time_us)

        msg = asimov_state_pb2.RobotState()
        msg.protocol_version = 1
        msg.sequence = self.sequence
        msg.timestamp_us = fw_us
        msg.current_mode = int(self.mode)
        msg.error_flags = self.error_flags

        # Joint telemetry in FIRMWARE order (with ankles mapped to motors A/B)
        fw_pos = JointAdapter.sim_to_firmware_positions(sim_joint_pos)
        fw_vel = JointAdapter.sim_to_firmware_velocities(sim_joint_vel)
        msg.joint_pos.extend(fw_pos)
        msg.joint_vel.extend(fw_vel)

        # joint_current: PLACEHOLDER 0.0 unless motor torque constant found
        msg.joint_current.extend([0.0] * 25)

        # joint_temp: PLACEHOLDER constant 25.0 C (or injected values)
        msg.joint_temp.extend(self.temperatures)

        # Base sensors from MuJoCo
        msg.base_quat.extend(float(x) for x in imu_quat)  # [w, x, y, z]
        msg.base_ang_vel.extend(float(x) for x in imu_gyro)
        msg.projected_gravity.extend(float(x) for x in imu_gravity)

        # Active alerts
        self._populate_active_alerts(msg.active_alerts, now_us)

        # Virtual Battery (PLACEHOLDER)
        if self.battery_enabled:
            msg.battery.voltage_v = float(self.battery_voltage)
            msg.battery.current_a = -1.0 if self.battery_discharge_rate > 0.0 else 0.0
            msg.battery.soc_percent = float(self.battery_soc)
            msg.battery.max_cell_temp_c = float(self.battery_temp_c)
            msg.battery.protection_flags = int(self.battery_protection_flags)

        return msg

    def build_edge_telemetry(
        self,
        sim_joint_pos: Sequence[float],
        sim_joint_vel: Sequence[float],
        imu_quat: Sequence[float],
        imu_gyro: Sequence[float],
        imu_gravity: Sequence[float],
    ) -> edge_cloud_pb2.EdgeTelemetry:
        """
        Build an official menlo.edge.EdgeTelemetry message (Edge-Cloud wire format).
        """
        now_us = int(time.time() * 1_000_000)
        fw_us = now_us - self.boot_time_us

        msg = edge_cloud_pb2.EdgeTelemetry()
        msg.timestamp_us = now_us
        msg.fw_timestamp_us = fw_us
        msg.sequence = self.sequence

        # Enum translation for EdgeTelemetry (FW_MODE_DAMP=0, FW_MODE_STAND=1, FW_MODE_MOVE=2)
        if self.mode == EdgeMode.STAND:
            msg.fw_mode = edge_cloud_pb2.FW_MODE_STAND
        elif self.mode == EdgeMode.MOVE:
            msg.fw_mode = edge_cloud_pb2.FW_MODE_MOVE
        else:
            msg.fw_mode = edge_cloud_pb2.FW_MODE_DAMP

        fw_pos = JointAdapter.sim_to_firmware_positions(sim_joint_pos)
        fw_vel = JointAdapter.sim_to_firmware_velocities(sim_joint_vel)
        msg.joint_pos.extend(fw_pos)
        msg.joint_vel.extend(fw_vel)
        msg.joint_current.extend([0.0] * 25)
        msg.joint_temp.extend(self.temperatures)

        msg.imu_quat.extend(float(x) for x in imu_quat)
        msg.imu_gyro.extend(float(x) for x in imu_gyro)
        msg.imu_gravity.extend(float(x) for x in imu_gravity)

        msg.error_flags = self.error_flags
        self._populate_firmware_alerts(msg.active_alerts, now_us)
        msg.alert_count = len(msg.active_alerts)
        msg.fw_age_ms = 0
        msg.last_video_timestamp_us = 0
        msg.last_audio_timestamp_us = 0

        return msg

    def _populate_active_alerts(self, alerts_field: Any, now_us: int) -> None:
        """Populate active alerts on asimov.io.RobotState."""
        if self.fault_fall:
            a = alerts_field.add()
            a.id = int(AlertId.FALL_DETECTED)
            a.severity = int(AlertSeverity.CRITICAL)
            a.value = 1
            a.threshold = 1
            a.source_id = 0
            a.first_set_us = now_us
        if self.fault_overtemp:
            a = alerts_field.add()
            a.id = int(AlertId.MOTOR_OVERTEMP)
            a.severity = int(AlertSeverity.CRITICAL)
            a.value = int(round(max(self.temperatures)))
            a.threshold = int(round(OVERTEMP_TRIP_C))
            a.source_id = 1
            a.first_set_us = now_us
        elif max(self.temperatures) >= TEMP_HIGH_WARN_C:
            a = alerts_field.add()
            a.id = int(AlertId.MOTOR_TEMP_HIGH)
            a.severity = int(AlertSeverity.WARNING)
            a.value = int(round(max(self.temperatures)))
            a.threshold = int(round(TEMP_HIGH_WARN_C))
            a.source_id = 1
            a.first_set_us = now_us
        if self.fault_can:
            a = alerts_field.add()
            a.id = int(AlertId.CAN_BUS_OFF)
            a.severity = int(AlertSeverity.WARNING)
            a.value = 1
            a.threshold = 0
            a.source_id = 2
            a.first_set_us = now_us

    def _populate_firmware_alerts(self, alerts_field: Any, now_us: int) -> None:
        """Populate active alerts on menlo.edge.EdgeTelemetry."""
        if self.fault_fall:
            a = alerts_field.add()
            a.id = int(AlertId.FALL_DETECTED)
            a.severity = int(AlertSeverity.CRITICAL)
            a.value = 1
            a.threshold = 1
            a.source_id = 0
            a.first_set_us = now_us
        if self.fault_overtemp:
            a = alerts_field.add()
            a.id = int(AlertId.MOTOR_OVERTEMP)
            a.severity = int(AlertSeverity.CRITICAL)
            a.value = int(max(self.temperatures))
            a.threshold = int(OVERTEMP_TRIP_C)
            a.source_id = 1
            a.first_set_us = now_us
        if self.fault_can:
            a = alerts_field.add()
            a.id = int(AlertId.CAN_BUS_OFF)
            a.severity = int(AlertSeverity.WARNING)
            a.value = 1
            a.threshold = 0
            a.source_id = 2
            a.first_set_us = now_us

    def build_diagnostics(self) -> edge_cloud_pb2.EdgeDiagnostics:
        """
        Build ~1 s diagnostics message with synthetic CAN counters.
        """
        diag = edge_cloud_pb2.EdgeDiagnostics()
        diag.controller = self.active_controller
        diag.provisioned = True
        diag.cloud_connected = True
        diag.ble_connected = False
        diag.camera_active = False
        diag.mic_active = False

        # Synthetic CAN counters [FLAGGED: SYNTHETIC / APPROXIMATE]
        can = diag.can_health.add()
        can.frames_sent = self.sequence * 25
        can.frames_received = self.sequence * 25
        can.errors = 1 if self.fault_can else 0
        can.bus_offs = 1 if self.fault_can else 0

        return diag
