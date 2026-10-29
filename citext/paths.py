import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]          # the repository root (this file is <root>/citext/paths.py)
CODE_DIR = PROJECT_ROOT
RESULTS_DIR = Path(os.environ.get("CI_TEXT_RESULTS_DIR", CODE_DIR / "results"))   # set to run a partial job into a separate directory
PAPER_DIR = Path(os.environ.get("CI_TEXT_PAPER_DIR", PROJECT_ROOT / "paper"))   # figures and result macros for the paper; set to write them elsewhere
FIG_DIR = PAPER_DIR / "figures"
CACHE_DIR = Path.home() / ".cache" / "ci_text"

for _d in (RESULTS_DIR, FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
