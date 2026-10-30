"""Tests always run on the demo fixtures without trained models, so results do not depend on local data."""

import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["COPILOT_DATA_DIR"] = str(ROOT / "samples")
os.environ["COPILOT_MODELS_DIR"] = tempfile.mkdtemp(prefix="copilot-models-")
os.environ["COPILOT_LLM"] = "off"
