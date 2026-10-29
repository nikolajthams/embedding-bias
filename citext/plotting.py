"""Matplotlib style for the paper figures (follows the dataviz reference palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .paths import FIG_DIR, RESULTS_DIR  # noqa: E402

# categorical slots, fixed order (light surface)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
# blue sequential ramp for ordered parameters (light -> dark)
SEQ = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef", 300: "#6da7ec",
       350: "#5598e7", 400: "#3987e5", 450: "#2a78d6", 500: "#256abf", 550: "#1c5cab",
       600: "#184f95", 650: "#104281", 700: "#0d366b"}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

SINGLE = (3.25, 2.3)   # inches, ~0.48 textwidth
DOUBLE = (6.6, 2.4)


def seq_colors(k: int):
    """k ordered colors from the blue ramp, light to dark, all >= step 250."""
    steps = [250, 400, 550] if k == 3 else [250, 350, 450, 550, 650][:k] if k <= 5 else \
        sorted(SEQ)[-k:]
    if k == 2:
        steps = [300, 550]
    if k == 4:
        steps = [250, 400, 500, 650]
    return [SEQ[s] for s in steps]


def setup():
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.5, "lines.markersize": 5, "lines.markeredgewidth": 0.8,
        "lines.markeredgecolor": SURFACE,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.borderaxespad": 0.2,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK,
    })


def savefig(fig, name: str, preview: bool = True):
    """Save vector PDF to paper/figures and a PNG preview to code/results/preview."""
    fig.savefig(FIG_DIR / name, bbox_inches="tight", pad_inches=0.02)
    if preview:
        pdir = RESULTS_DIR / "preview"
        pdir.mkdir(exist_ok=True)
        fig.savefig(pdir / name.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def hline(ax, y, label=None, color=MUTED, x_text=None, **kw):
    ax.axhline(y, color=color, linewidth=0.8, linestyle="--", zorder=1, **kw)
    if label:
        xt = x_text if x_text is not None else ax.get_xlim()[0]
        ax.annotate(label, (xt, y), xytext=(2, 2), textcoords="offset points", fontsize=6.5,
                    color=INK2, ha="left", va="bottom")


def end_label(ax, x, y, text, color=INK2, dx=3, dy=0):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                color=color, ha="left", va="center")
