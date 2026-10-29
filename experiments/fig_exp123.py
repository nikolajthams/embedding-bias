"""Main-text figure: rejection rates under H0 of Experiments 1 and 2 as two stacked heatmaps (rows = tests, columns = sample
size, colour and number = rejection rate) with a shared colour bar. Cells whose regressions are ridge regressions under the
regressor rule (Experiment 2, the 1152-dimensional set at n < d) carry a dashed frame.
Reads saved results only (no recomputation):
  results/exp1_typeI_vs_n.csv, exp1_robust_test.csv, exp2_text_typeI.csv (column reg_used), exp2_text_rv_benchmark.csv
  (column reg_used_test), summary_exp2.json
Writes <paper>/figures/exp123_main.tex (TikZ; the paper inputs it).
Usage: python experiments/fig_exp123.py   (paper dir from citext.paths, i.e. CI_TEXT_PAPER_DIR if set)
       python fig_exp123.py <code dir> <paper dir>   from outside the code tree
History: replaces fig_exp12.py (30 Sept 2026). The right panel with the Experiment 3 learning curve was dropped on
7 Oct 2026 when that experiment left the paper, and the dashed frames were added with the regressor rule of that date.
"""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
try:
    from citext import plotting as P; from citext.paths import RESULTS_DIR, FIG_DIR
except ImportError:   # run from outside the code tree: python fig_exp123.py <code dir> <paper dir>
    CODE = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(CODE))
    from citext import plotting as P
    RESULTS_DIR = CODE / "results"; FIG_DIR = Path(sys.argv[2]).resolve() / "figures"

ALPHA = 0.05; NS = [500, 1000, 2000, 4000, 10000]
SUMMARY2 = json.load(open(RESULTS_DIR / "summary_exp2.json")); REG2 = SUMMARY2["primary_regressor"]
ROWS = [("suff", r"standard, sufficient $\psi$"),
        ("rich", r"standard, sufficient $\psi$, %d-dim."),
        ("ins0", r"standard, insufficient $\psi$, $\lambda = 0$"),
        ("ins1", r"standard, insufficient $\psi$, $\lambda = 1$"),
        ("rob_true", r"robust, $B$ = true insufficiency"),
        ("rob_half", r"robust, $B$ = half of it")]
CW, CH, GAP, INSET = 0.80, 0.40, 0.025, 0.045   # cell width, height, half of the surface gap between cells, inset of the dashed frame (cm)


def hex2rgb(h): h = h.lstrip("#"); return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def ramp(v):
    """Colour of a rate v in [0, 1] on the blue sequential ramp (piecewise linear through its steps)."""
    steps = sorted(P.SEQ); pos = steps[0] + (steps[-1] - steps[0]) * float(np.clip(v, 0, 1))
    for a, b in zip(steps, steps[1:]):
        if pos <= b:
            t = (pos - a) / (b - a); ca, cb = hex2rgb(P.SEQ[a]), hex2rgb(P.SEQ[b])
            return tuple(int(round(x + t * (y - x))) for x, y in zip(ca, cb))
    return hex2rgb(P.SEQ[steps[-1]])


def tikz_rgb(rgb): return "{rgb,255:red,%d;green,%d;blue,%d}" % rgb


def exp1():
    t = pd.read_csv(RESULTS_DIR / "exp1_typeI_vs_n.csv"); r = pd.read_csv(RESULTS_DIR / "exp1_robust_test.csv").set_index("n")
    g = lambda tag: t[t.tag == tag].set_index("n").rej
    return {"suff": g("sufficient (q=20)"), "ins0": g("q=5, lam=0"), "ins1": g("q=5, lam=1"), "rob_true": r.robust_bound, "rob_half": r.robust_half}, set()


def exp2():
    """Rates per row, the dimension of the rich set, and the (row, n) cells fitted by ridge regression under the rule."""
    t = pd.read_csv(RESULTS_DIR / "exp2_text_typeI.csv"); rej = f"rej_{REG2}"
    sel = lambda psi, lam: t[(t.psi == psi) & (t.lam == lam)].set_index("n")
    a = pd.read_csv(RESULTS_DIR / "exp2_text_rv_benchmark.csv").set_index("n")
    dim = int(SUMMARY2["rich_encoder_dim"])
    blocks = {"suff": sel("oracle", 1.0), "rich": sel("rich", 1.0), "ins0": sel("E1", 0.0), "ins1": sel("E1", 1.0)}
    data = {k: b[rej] for k, b in blocks.items()}; data.update({"rob_true": a.robust_bound_rej, "rob_half": a.robust_half_rej})
    used = {k: b["reg_used"] for k, b in blocks.items() if "reg_used" in b}
    if "reg_used_test" in a:
        used.update({"rob_true": a.reg_used_test, "rob_half": a.reg_used_test})
    ridge = {(k, int(n)) for k, s in used.items() for n, r in s.items() if r == "ridge"}
    return data, dim, ridge


def val(series, n):
    try:
        v = float(series.loc[n])
    except KeyError:
        return np.nan
    return v


def heatmap(out, data, rows, y_top, title, ridge=frozenset()):
    """Cells of one block; rows from the top. Returns the y of the bottom edge."""
    out.append(r"\node[anchor=south west, font=\footnotesize, inner sep=1pt] at (0,%.3f) {%s};" % (y_top + 0.05, title))
    for j, (key, lab) in enumerate(rows):
        y1 = y_top - j * CH; y0 = y1 - CH; yc = (y0 + y1) / 2
        out.append(r"\node[anchor=east, font=\scriptsize, text=cInk, inner sep=1pt] at (-0.12,%.3f) {%s};" % (yc, lab))
        for i, n in enumerate(NS):
            x0, x1 = i * CW + GAP, (i + 1) * CW - GAP; v = val(data.get(key, pd.Series(dtype=float)), n)
            if np.isnan(v):
                out.append(r"\draw[cGrid, line width=0.4pt] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (x0, y0 + GAP, x1, y1 - GAP))
                continue
            ink = "white" if v > 0.5 else "cInk"
            out.append(r"\fill[color=%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (tikz_rgb(ramp(v)), x0, y0 + GAP, x1, y1 - GAP))
            if (key, n) in ridge:   # ridge regression under the rule: dashed frame inside the cell
                out.append(r"\draw[%s, dash pattern=on 1.1pt off 0.9pt, line width=0.45pt] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                           % (ink, x0 + INSET, y0 + GAP + INSET, x1 - INSET, y1 - GAP - INSET))
            out.append(r"\node[font=\scriptsize, text=%s, inner sep=0pt] at (%.3f,%.3f) {%s};" % (ink, (x0 + x1) / 2, yc, ("%.2f" % v)))
    return y_top - len(rows) * CH


def write(data1, ridge1, data2, rich_dim, ridge2):
    out = ["%% Generated by code/experiments/fig_exp123.py from code/results/*.csv (Experiment 2 regressor: %s; dashed cells: ridge regression under the rule). Do not edit by hand." % REG2,
           r"\definecolor{cInk}{HTML}{%s}" % P.INK2.lstrip("#").upper(), r"\definecolor{cMuted}{HTML}{%s}" % P.MUTED.lstrip("#").upper(),
           r"\definecolor{cAxis}{HTML}{%s}" % P.AXIS.lstrip("#").upper(), r"\definecolor{cGrid}{HTML}{%s}" % P.GRID.lstrip("#").upper()]
    out.append(r"\begin{tikzpicture}[baseline=(current bounding box.south)]")
    rows1 = [(k, l) for k, l in ROWS if k != "rich"]; rows2 = [(k, l % rich_dim if "%d" in l else l) for k, l in ROWS]
    y_top1 = 0.0; y_bot1 = heatmap(out, data1, rows1, y_top1, "Synthetic (Experiment~1)", ridge1)
    y_top2 = y_bot1 - 0.62; y_bot2 = heatmap(out, data2, rows2, y_top2, "Text (Experiment~2)", ridge2)
    for i, n in enumerate(NS):    # shared x labels under the bottom block
        out.append(r"\node[anchor=north, font=\scriptsize, text=cInk, inner sep=2pt] at (%.3f,%.3f) {%s};" % ((i + 0.5) * CW, y_bot2 - 0.02, "{:,}".format(n).replace(",", "{,}")))
    out.append(r"\node[anchor=north, font=\footnotesize, inner sep=1pt] at (%.3f,%.3f) {sample size $n$};" % (len(NS) * CW / 2, y_bot2 - 0.42))
    # colour bar
    xb = len(NS) * CW + 0.28; y0, y1 = y_bot2, y_top1; K = 40
    for k in range(K):
        v0, v1 = k / K, (k + 1) / K
        out.append(r"\fill[color=%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (tikz_rgb(ramp((v0 + v1) / 2)), xb, y0 + v0 * (y1 - y0), xb + 0.22, y0 + v1 * (y1 - y0) + 0.005))
    for v in (0, 0.25, 0.5, 0.75, 1):
        y = y0 + v * (y1 - y0)
        out.append(r"\draw[cAxis, line width=0.4pt] (%.3f,%.3f) -- (%.3f,%.3f);" % (xb + 0.22, y, xb + 0.30, y))
        out.append(r"\node[anchor=west, font=\tiny, text=cInk, inner sep=1pt] at (%.3f,%.3f) {%.2f};" % (xb + 0.30, y, v))
    out.append(r"\node[anchor=south, font=\scriptsize, text=cInk, inner sep=1pt, rotate=90] at (%.3f,%.3f) {rejection rate under $H_0$};" % (xb + 1.02, (y0 + y1) / 2))
    out += [r"\end{tikzpicture}", ""]
    path = FIG_DIR / "exp123_main.tex"; path.write_text("\n".join(out)); print(path)


def main():
    data1, ridge1 = exp1(); data2, rich_dim, ridge2 = exp2()
    for name, d in (("exp1", data1), ("exp2", data2)):
        print(name, {k: [None if np.isnan(val(s, n)) else round(val(s, n), 3) for n in NS] for k, s in d.items()})
    print("ridge cells", sorted(ridge2))
    write(data1, ridge1, data2, rich_dim, ridge2)


if __name__ == "__main__":
    main()
