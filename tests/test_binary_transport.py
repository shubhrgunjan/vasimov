import asyncio
import os
import sys
import time
import pytest
from pathlib import Path
import websockets

_VASIMOV_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_VASIMOV_DIR))

os.environ.setdefault("MUJOCO_GL", "egl")

from edge.core import EdgeCore, EdgeMode
from edge.sim import SimBackend
from web.server.gateway import WebGateway
from web.server.binary_protocol import (
    STATE_PACKET_SIZE, COMMAND_PACKET_SIZE, PROTOCOL_VERSION,
    unpack_state_packet, pack_command_packet, CommandType
)

@pytest.mark.asyncio
async def test_binary_state_streaming_and_command():
    core = EdgeCore()
    backend = SimBackend(core=core)
    
    # Use non-default ports for testing
    http_port = 8872
    ws_port = 8874
    gateway = WebGateway(backend=backend, core=core, http_port=http_port, ws_port=ws_port)
    gateway.start()
    
    try:
        await asyncio.sleep(0.5)
        
        uri = f"ws://127.0.0.1:{ws_port}"
        async with websockets.connect(uri) as ws:
            # Subscribe to binary stream
            await ws.send("{\"action\": \"subscribe\", \"params\": {\"format\": \"binary\"}}")
            resp = await ws.recv()
            assert "subscribed" in resp
            
            # Step the simulation
            with backend._lock:
                for _ in range(10):
                    backend.step()
            
            # Collect 5 binary packets
            packets = []
            for _ in range(5):
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                assert isinstance(msg, bytes), f"Expected bytes, got {type(msg)}"
                assert len(msg) == STATE_PACKET_SIZE, f"Expected {STATE_PACKET_SIZE} bytes, got {len(msg)}"
                state = unpack_state_packet(msg)
                packets.append(state)
            
            assert len(packets) == 5
            for i in range(1, len(packets)):
                assert packets[i]["sequence_number"] > packets[i-1]["sequence_number"]
                assert packets[i]["timestamp"] >= packets[i-1]["timestamp"]
                assert len(packets[i]["joint_positions"]) == 25
                assert len(packets[i]["joint_velocities"]) == 25
                assert len(packets[i]["base_position"]) == 3
                assert len(packets[i]["base_orientation"]) == 4

            # 1. Arm to STAND first
            stand_bytes = pack_command_packet(
                sequence=1,
                timestamp=time.time(),
                command_type=CommandType.STAND,
            )
            await ws.send(stand_bytes)
            await asyncio.sleep(0.05)
            assert core.mode == EdgeMode.STAND

            # 2. Send WALK command
            walk_bytes = pack_command_packet(
                sequence=2,
                timestamp=time.time(),
                command_type=CommandType.WALK,
                linear_x=0.4,
                linear_y=0.0,
                yaw_rate=-0.2,
            )
            await ws.send(walk_bytes)
            await asyncio.sleep(0.05)

            # Verify that velocity command reached EdgeCore and transitioned to MOVE
            assert abs(core.current_vx - 0.4) < 1e-3
            assert abs(core.current_vyaw - (-0.2)) < 1e-3
            assert core.mode == EdgeMode.MOVE

            # 3. Test ESTOP / DAMP command
            estop_bytes = pack_command_packet(
                sequence=3,
                timestamp=time.time(),
                command_type=CommandType.ESTOP,
            )
            await ws.send(estop_bytes)
            await asyncio.sleep(0.05)
            assert core.mode in (EdgeMode.DAMP, EdgeMode.FAULT_DAMP)

    finally:
        gateway.stop()
