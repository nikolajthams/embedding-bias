"""Write the Experiment 1 block (\\resOne*) of paper/results_macros.tex from results/summary_exp1.json and
results/exp1_robust_test_decisions.csv (both written by exp1_synthetic.py --only plots).
Usage: python experiments/exp1_macros.py [--write]   (without --write the block is printed only)"""
import sys, json, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from citext.paths import RESULTS_DIR, PAPER_DIR
MACROS_TEX = PAPER_DIR / "results_macros.tex"
BEGIN, END = "% ---------- Experiment 1 ----------", "% ---------- Experiment 2 ----------"
REGRESSOR_TEXT = {"hybrid": "least squares regressions", "ols": "least squares regressions", "ridge": "ridge regressions"}


def block():
    S = json.load(open(RESULTS_DIR / "summary_exp1.json")); dec = pd.read_csv(RESULTS_DIR / "exp1_robust_test_decisions.csv")
    ns = S["ns"]; n_lo, n_hi = ns[0], ns[-1]; q, p = S["q_insufficient"], S["p"]; r2 = S["r2"]; bound = float(np.sqrt(r2 * r2))
    f1 = S["fig1"]; rows3 = {r["n"]: r for r in S["fig3"]["rows"]}; f4 = {round(r["beta"], 3): r for r in S["fig4"]}; f5 = S["fig5"]
    budgets = sorted(float(k.split("_")[1]) for k in S["fig4"][0] if k.startswith("robust_"))
    ref = 0.1 if 0.1 in budgets else budgets[len(budgets) // 2]
    coverage = float(np.mean(dec.RV_alpha < bound))
    m = [
        ("resOneDimZ", str(p)), ("resOneQsuff", str(S["q_sufficient"])), ("resOneQinsuff", str(q)),
        ("resOneQbench", str(int(S["bench_encoder"].split("_")[1].split()[0]))),
        ("resOneNmin", f"{n_lo:,}".replace(",", "{,}")), ("resOneNmax", f"{n_hi:,}".replace(",", "{,}")), ("resOneNrep", str(S["n_rep"])),
        ("resOneRegressor", REGRESSOR_TEXT.get(S["regressor"], S["regressor"])),
        ("resOneRtwoA", f"{r2:g}"), ("resOneRtwoB", f"{r2:g}"), ("resOneBound", f"{bound:g}"),
        ("resOneTypeISuff", f"{f1[f'sufficient (q={p})|n={n_hi}']:.3f}"), ("resOneTypeIInsuffOrth", f"{f1[f'q={q}, lam=0|n={n_hi}']:.3f}"),
        ("resOneTypeIHalfSmallN", f"{f1[f'q={q}, lam=0.5|n={n_lo}']:.2f}"), ("resOneTypeIHalfLargeN", f"{f1[f'q={q}, lam=0.5|n={n_hi}']:.2f}"),
        ("resOneTypeIAlignedSmallN", f"{f1[f'q={q}, lam=1|n={n_lo}']:.2f}"), ("resOneTypeIAlignedLargeN", f"{f1[f'q={q}, lam=1|n={n_hi}']:.2f}"),
        ("resOneRobustTypeI", f"{rows3[n_hi]['robust_bound']:.3f}"), ("resOneRobustTypeIMax", f"{max(r['robust_bound'] for r in rows3.values()):.3f}"),
        ("resOneRobustTypeIHalf", f"{rows3[n_hi]['robust_half']:.2f}"), ("resOneRobustTypeIBench", f"{rows3[n_hi]['robust_bench']:.2f}"),
        ("resOnePowerN", str(S["power_n"])), ("resOnePowerBudgets", "\\{" + ", ".join(f"{b:g}" for b in budgets) + "\\}"), ("resOnePowerBudgetRef", f"{ref:g}"),
        ("resOnePowerNaive", f"{f4[0.1]['naive']:.2f}"), ("resOnePowerRobust", f"{f4[0.1][f'robust_{ref:g}']:.2f}"), ("resOnePowerRobustTwo", f"{f4[0.2][f'robust_{ref:g}']:.2f}"),
        ("resOneRVTrue", f"{bound:g}"), ("resOneRVBias", f"{max(abs(v['RV_median'] - bound) for v in f5.values()):.3f}"),
        ("resOneRVCoverage", f"{100 * coverage:.0f}\\%"),
        ("resOneRValphaSmall", f"{f5[f'n={n_lo}']['RV_alpha_median']:.3f}"), ("resOneRValphaLarge", f"{f5[f'n={n_hi}']['RV_alpha_median']:.3f}"),
    ]
    lines = [BEGIN, f"% written by code/experiments/exp1_macros.py from results/summary_exp1.json (regressor: {S['regressor']}); do not edit by hand"]
    lines += ["\\newcommand{\\%s}{%s}" % (k, v) for k, v in m]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--write", action="store_true"); args = ap.parse_args()
    new = block(); print(new)
    if args.write:
        s = MACROS_TEX.read_text(); i, j = s.index(BEGIN), s.index(END)
        MACROS_TEX.write_text(s[:i] + new + s[j:]); print("written to", MACROS_TEX)


if __name__ == "__main__":
    main()
