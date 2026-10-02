import os
# Ensure headless EGL offscreen rendering is used by default across test suites
os.environ.setdefault("MUJOCO_GL", "egl")
