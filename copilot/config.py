"""Paths and tunables. Everything can be overridden with environment variables."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path):
    """Minimal .env support (KEY=VALUE lines) so Windows users don't have to export variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if value.strip():
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("COPILOT_DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("COPILOT_MODELS_DIR", ROOT / "models"))
REPORTS_DIR = Path(os.getenv("COPILOT_REPORTS_DIR", ROOT / "reports"))
SAMPLES_DIR = ROOT / "samples"
RESOURCES_DIR = Path(__file__).resolve().parent / "resources"

# LLM: "auto" = use Claude when ANTHROPIC_API_KEY is set, otherwise offline templates.
LLM_MODE = os.getenv("COPILOT_LLM", "auto")  # auto | anthropic | off
LLM_MODEL = os.getenv("COPILOT_LLM_MODEL", "claude-haiku-5-5")
LLM_JUDGE_MODEL = os.getenv("COPILOT_JUDGE_MODEL", "claude-sonnet-5-5")
# Hard deadline for one LLM call; on timeout we fall back to the offline draft,
# which keeps p95 latency under the 5 s limit from the brief.
LLM_TIMEOUT_S = float(os.getenv("COPILOT_LLM_TIMEOUT", "3.5"))
LLM_MAX_TOKENS = int(os.getenv("COPILOT_LLM_MAX_TOKENS", "900"))
LLM_WORKERS = int(os.getenv("COPILOT_WORKERS", "8"))

# USD per 1M tokens: (input, output). Cache reads cost 0.1x input, 5-min cache writes 1.25x.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

DEDUP_WINDOW_MIN = int(os.getenv("COPILOT_DEDUP_WINDOW_MIN", "60"))
DEDUP_SIMILARITY = float(os.getenv("COPILOT_DEDUP_SIM", "0.55"))
# Below this confidence a ticket is routed to a human even if nothing else requires it.
AUTO_REPLY_MIN_CONFIDENCE = float(os.getenv("COPILOT_AUTO_MIN_CONF", "0.70"))

SUPPORTED_LANGS = ("ru", "kk", "uz", "en", "es", "pt")


def data_file(name: str) -> Path:
    """Real data from data/ if present, else the demo fixture from samples/."""
    p = DATA_DIR / name
    if p.exists():
        return p
    return SAMPLES_DIR / name
