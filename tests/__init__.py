import os
import sys

# Platform-aware headless offscreen rendering: 'cgl' on macOS, 'egl' on Linux, 'osmesa' on Windows
if sys.platform == "darwin":
    default_gl = "cgl"
elif sys.platform.startswith("linux"):
    default_gl = "egl"
else:
    default_gl = "osmesa"

os.environ.setdefault("MUJOCO_GL", default_gl)
