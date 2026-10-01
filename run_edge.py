#!/usr/bin/env python3
"""
Virtual Asimov Edge Launcher.
Usage:
    mjpython run_edge.py --realtime --viewer
    python run_edge.py --realtime
    python run_edge.py --fast
"""
import sys
from pathlib import Path

# Ensure vasimov directory is in python search path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from edge.__main__ import main

if __name__ == "__main__":
    main()
