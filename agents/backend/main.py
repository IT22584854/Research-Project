"""Compatibility wrapper that re-exports the canonical FastAPI app from agents/app.py."""
from __future__ import annotations

import sys
from pathlib import Path
 
_AGENTS_DIR = Path(__file__).resolve().parents[1]
if str(_AGENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENTS_DIR))

from app import app
