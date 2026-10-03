"""
vasimov/web/server/camera_streamer.py
MuJoCo Offscreen Camera Streaming Worker.

Features:
- Enumerate real MuJoCo model cameras (front_camera, side_camera, etc.) from m.ncam.
- Support separate free/interactive orbit camera (viewer_camera) tracked to pelvis.
- Renders offscreen using mujoco.Renderer without blocking the 200 Hz physics loop.
- Supports SOLID and X-RAY rendering modes via MjvOption.
- High-performance JPEG compression with frame metadata (sequence, sim_time, timestamp).
- Thread-safe frame cache for instantaneous HTTP MJPEG and snapshot serving.
"""

from __future__ import annotations
import io
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure appropriate headless rendering backend is used for offscreen worker threads:
# macOS uses CGL (Core OpenGL), Linux uses EGL, Windows uses OSMesa.
import sys
if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "cgl")
elif sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
else:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco

from PIL import Image

log = logging.getLogger("vasimov.web.camera")

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
THUMB_WIDTH = 320
THUMB_HEIGHT = 240


class CameraStreamer:
    """Manages offscreen rendering for all model cameras and the viewer camera."""

    def __init__(self, backend, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT):
        self.backend = backend
        self.width = width
        self.height = height
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Model and rendering resources
        self.model = backend.model
        self.render_data = mujoco.MjData(self.model)
        self.renderer: Optional[mujoco.Renderer] = None

        # Free viewer camera settings
        self.viewer_cam = mujoco.MjvCamera()
        self.viewer_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer_cam.trackbodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        self.viewer_cam.distance = 3.0
        self.viewer_cam.azimuth = 135.0
        self.viewer_cam.elevation = -20.0

        # Enumerate model cameras
        self.model_cameras: List[str] = []
        for i in range(self.model.ncam):
            cname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera_{i}"
            self.model_cameras.append(cname)

        # Available cameras list
        self.all_cameras = list(self.model_cameras) + ["viewer_camera"]

        # Cache of latest JPEG frames and metadata
        # Key: (camera_name, xray_bool) -> {"jpeg": bytes, "sim_time": float, "seq": int, "wall_time": float}
        self.frame_cache: Dict[Tuple[str, bool], Dict[str, Any]] = {}
        self._render_lock = threading.Lock()
        self.frame_sequences: Dict[str, int] = {c: 0 for c in self.all_cameras}
        self.active_subscribers: Dict[str, int] = {}
        self.primary_camera: str = self.model_cameras[0] if self.model_cameras else "viewer_camera"
        self.primary_xray: bool = False
        self.frame_listeners: List[Callable[[str, np.ndarray, float], None]] = []
        self.target_fps: int = 25

        # Options for Normal vs X-Ray
        self.opt_solid = mujoco.MjvOption()
        self.opt_xray = mujoco.MjvOption()
        self.opt_xray.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 1
        self.opt_xray.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 1
        self.opt_xray.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 1
        self.opt_xray.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1

    def add_frame_listener(self, listener: Callable[[str, np.ndarray, float], None]) -> None:
        """Register a callback for uncompressed raw RGB frames (e.g. WebRTC VideoSource)."""
        if listener not in self.frame_listeners:
            self.frame_listeners.append(listener)
            self.target_fps = 60  # Elevate capture loop to 60 FPS target when WebRTC is active

    def remove_frame_listener(self, listener: Callable[[str, np.ndarray, float], None]) -> None:
        if listener in self.frame_listeners:
            self.frame_listeners.remove(listener)
            if not self.frame_listeners:
                self.target_fps = 25

    def start(self) -> None:
        """Start the background rendering worker thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True, name="vasimov-camera-streamer")
        self._thread.start()
        log.info("[CAMERA] Streamer started for cameras: %s", self.all_cameras)

    def stop(self) -> None:
        """Stop background rendering."""
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        with self._render_lock:
            if self.renderer is not None:
                try:
                    self.renderer.close()
                except Exception:
                    pass
                self.renderer = None
        log.info("[CAMERA] Streamer stopped.")

    def set_orbit_camera(self, azimuth: Optional[float] = None, elevation: Optional[float] = None, distance: Optional[float] = None) -> None:
        """Adjust orbit viewer camera parameters."""
        with self._lock:
            if azimuth is not None:
                self.viewer_cam.azimuth = float(azimuth)
            if elevation is not None:
                self.viewer_cam.elevation = float(elevation)
            if distance is not None:
                self.viewer_cam.distance = max(0.5, min(20.0, float(distance)))

    def _render_single(self, camera_name: str, xray: bool, sim_time: float) -> Optional[bytes]:
        """Render one frame offscreen and encode to JPEG."""
        with self._render_lock:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, self.height, self.width)

            vopt = self.opt_xray if xray else self.opt_solid

            try:
                if camera_name == "viewer_camera":
                    self.renderer.update_scene(self.render_data, camera=self.viewer_cam, scene_option=vopt)
                else:
                    self.renderer.update_scene(self.render_data, camera=camera_name, scene_option=vopt)

                rgb = self.renderer.render()

                # Dispatch uncompressed raw RGB frame to realtime listeners (WebRTC)
                if not xray:
                    for listener in list(self.frame_listeners):
                        try:
                            listener(camera_name, rgb, sim_time)
                        except Exception as e:
                            log.debug("[CAMERA] Frame listener error: %s", e)

                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="JPEG", quality=75)
                jpeg_bytes = buf.getvalue()

                seq = self.frame_sequences.get(camera_name, 0) + 1
                self.frame_sequences[camera_name] = seq

                with self._lock:
                    self.frame_cache[(camera_name, xray)] = {
                        "jpeg": jpeg_bytes,
                        "sim_time": sim_time,
                        "seq": seq,
                        "wall_time": time.time(),
                        "camera": camera_name,
                        "xray": xray,
                        "width": self.width,
                        "height": self.height,
                    }
                return jpeg_bytes
            except Exception as e:
                log.debug("[CAMERA] Render error for %s: %s", camera_name, e)
                return None

    def _sync_model_if_changed(self) -> None:
        """If simulation environment loaded a new MjModel, synchronize render resources."""
        if self.model != self.backend.model:
            with self._render_lock:
                self.model = self.backend.model
                self.render_data = mujoco.MjData(self.model)
                if self.renderer is not None:
                    try:
                        self.renderer.close()
                    except Exception:
                        pass
                    try:
                        self.renderer = mujoco.Renderer(self.model, self.height, self.width)
                    except Exception as e:
                        log.error("[CAMERA] Failed to recreate renderer after model change: %s", e)
                        self.renderer = None

                # Re-enumerate cameras
                self.model_cameras = []
                for i in range(self.model.ncam):
                    cname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i) or f"camera_{i}"
                    self.model_cameras.append(cname)
                self.all_cameras = list(self.model_cameras) + ["viewer_camera"]
                if self.primary_camera not in self.all_cameras:
                    self.primary_camera = self.model_cameras[0] if self.model_cameras else "viewer_camera"
                log.info("[CAMERA] Synchronized offscreen renderer to new model (%d cameras, %d geoms)", len(self.model_cameras), self.model.ngeom)

    def _render_loop(self) -> None:
        """Background render loop running independently from physics."""
        # Warmup renderer inside thread
        try:
            self.renderer = mujoco.Renderer(self.model, self.height, self.width)
        except Exception as e:
            log.error("[CAMERA] Could not initialize MuJoCo Renderer: %s", e)
            return

        last_secondary_render = 0.0

        while self.running:
            t0 = time.perf_counter()

            # Atomically copy current simulation state in <0.4ms
            with self.backend._lock:
                self._sync_model_if_changed()
                sim_time = float(self.backend.data.time)
                try:
                    mujoco.mj_copyData(self.render_data, self.model, self.backend.data)
                except Exception as e:
                    log.debug("[CAMERA] mj_copyData error during state copy: %s", e)
                    continue

            # 1. Always render primary live camera (solid + xray if active)
            self._render_single(self.primary_camera, self.primary_xray, sim_time)
            if self.primary_xray:
                # Also maintain solid for preview
                self._render_single(self.primary_camera, False, sim_time)

            # 2. Render other camera thumbnails periodically (~5-10 Hz)
            now = time.perf_counter()
            if now - last_secondary_render >= 0.15:
                last_secondary_render = now
                for cam in self.all_cameras:
                    if cam != self.primary_camera:
                        self._render_single(cam, False, sim_time)

            elapsed = time.perf_counter() - t0
            target_dt = 1.0 / float(self.target_fps)  # adaptive (25 FPS or 60 FPS)
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_latest_frame(self, camera_name: str = "front_camera", xray: bool = False) -> Optional[Dict[str, Any]]:
        """Retrieve latest cached JPEG frame and metadata."""
        if camera_name not in self.all_cameras:
            camera_name = self.primary_camera

        with self._lock:
            frame = self.frame_cache.get((camera_name, xray))
            if frame is None and xray:
                # Fallback to solid if xray not rendered yet
                frame = self.frame_cache.get((camera_name, False))
            if frame is not None:
                return frame

        # On-demand fallback if cache is cold
        try:
            with self.backend._lock:
                sim_time = float(self.backend.data.time)
                mujoco.mj_copyData(self.render_data, self.model, self.backend.data)
            self._render_single(camera_name, xray, sim_time)
            with self._lock:
                return self.frame_cache.get((camera_name, xray))
        except Exception as e:
            log.warning("[CAMERA] On-demand frame render failed for %s: %s", camera_name, e)
            return None

    def get_camera_metadata(self) -> List[Dict[str, Any]]:
        """Return metadata list for all real cameras."""
        meta = []
        with self._lock:
            for cname in self.all_cameras:
                is_viewer = (cname == "viewer_camera")
                seq = self.frame_sequences.get(cname, 0)
                meta.append({
                    "name": cname,
                    "type": "VIEWER CAMERA" if is_viewer else "MODEL CAMERA",
                    "resolution": [self.width, self.height],
                    "fps": 25.0,
                    "sequence": seq,
                    "is_model_camera": not is_viewer,
                })
        return meta
