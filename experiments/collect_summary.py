"""Merge the per-experiment summaries into results/summary.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from citext.paths import RESULTS_DIR  # noqa: E402

out = {}
for name in ("exp1", "exp2", "exp3"):
    f = RESULTS_DIR / f"summary_{name}.json"
    if f.exists():
        out[name] = json.loads(f.read_text())
(RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=1))
print("wrote", RESULTS_DIR / "summary.json", "with", list(out))
