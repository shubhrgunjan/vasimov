"""
vasimov/web/server/replay_manager.py
Session Replay Manager for Virtual Asimov 1.

Scans reports/raw/*.json, loads recorded telemetry sessions, and provides
scrubbable playback with recorded joint state, base pose, contacts, and policy I/O.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("vasimov.web.replay")
_REPORTS_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / "raw"


class ReplayManager:
    """Manages recording playback and listing."""

    def __init__(self):
        self.active: bool = False
        self.playing: bool = False
        self.playback_speed: float = 1.0
        self.current_recording_name: Optional[str] = None
        self.frames: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.current_frame_idx: int = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def list_recordings(self) -> List[Dict[str, Any]]:
        """List available JSON recordings in reports/raw/."""
        recordings = []
        if not _REPORTS_RAW_DIR.exists():
            return recordings

        for p in sorted(_REPORTS_RAW_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                # Read header
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("metadata", {})
                frames = data.get("frames", [])
                duration = 0.0
                if len(frames) > 1:
                    t0 = frames[0].get("sim_time", 0.0)
                    t1 = frames[-1].get("sim_time", 0.0)
                    duration = max(0.0, t1 - t0)

                recordings.append({
                    "name": p.stem,
                    "filename": p.name,
                    "frame_count": len(frames),
                    "duration_s": duration,
                    "created_time": p.stat().st_mtime,
                })
            except Exception:
                pass
        return recordings

    def load_recording(self, name: str) -> bool:
        """Load recording into memory for replay."""
        clean_name = name.replace(".json", "")
        p = _REPORTS_RAW_DIR / f"{clean_name}.json"
        if not p.exists():
            return False

        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self._lock:
                self.current_recording_name = clean_name
                self.metadata = data.get("metadata", {})
                self.frames = data.get("frames", [])
                self.current_frame_idx = 0
                self.active = True
                self.playing = False
            log.info("[REPLAY] Loaded '%s' with %d frames", clean_name, len(self.frames))
            return True
        except Exception as e:
            log.error("[REPLAY] Failed to load %s: %s", p, e)
            return False

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def seek(self, frame_index: int) -> Dict[str, Any]:
        with self._lock:
            if not self.frames:
                return {}
            self.current_frame_idx = max(0, min(len(self.frames) - 1, int(frame_index)))
            return self.get_current_frame()

    def step(self, delta: int = 1) -> Dict[str, Any]:
        with self._lock:
            if not self.frames:
                return {}
            self.current_frame_idx = max(0, min(len(self.frames) - 1, self.current_frame_idx + delta))
            return self.get_current_frame()

    def get_current_frame(self) -> Dict[str, Any]:
        with self._lock:
            if not self.frames or self.current_frame_idx >= len(self.frames):
                return {}
            fr = self.frames[self.current_frame_idx]
            return {
                "is_replay": True,
                "name": self.current_recording_name,
                "frame_index": self.current_frame_idx,
                "total_frames": len(self.frames),
                "playing": self.playing,
                "data": fr,
            }

    def stop_replay(self) -> None:
        with self._lock:
            self.active = False
            self.playing = False
            self.frames = []
            self.current_recording_name = None
