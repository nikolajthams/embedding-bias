"""Write the Experiment 2 block (\\resTwo*) of paper/results_macros.tex from results/summary_exp2.json
(written by exp2_text.py --only plots). Without --write the block is printed only.
Usage: python experiments/exp2_macros.py [--write]
"""
import sys, json, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from citext.paths import RESULTS_DIR, PAPER_DIR
MACROS_TEX = PAPER_DIR / "results_macros.tex"
BEGIN, END = "% ---------- Experiment 2 ----------", "% ---------- Experiment 3 ----------"


def thousands(n):
    return f"{int(n):,}".replace(",", "{,}")


def block():
    S = json.load(open(RESULTS_DIR / "summary_exp2.json"))
    P1, S1 = S["primary_regressor"], S["secondary_regressor"]
    NS = S["ns"]; n_lo, n_hi = NS[0], NS[-1]; K = S["K_folds"]; ratio = S["hybrid_min_rows_per_col"]
    typeI, rvb, power = pd.DataFrame(S["typeI"]), pd.DataFrame(S["rv_benchmark"]), pd.DataFrame(S["power"])
    pop, pop2 = S["population"][f"lam1.0_{P1}"], S["population"][f"lam1.0_{S1}"]
    d1, d2 = S["dim_E1"], S["dim_E2"]

    def t(psi, lam, n, col):
        return float(typeI[(typeI.psi == psi) & (typeI.lam == lam) & (typeI.n == n)][col].iloc[0])

    def b(n, col):
        return float(rvb[rvb.n == n][col].iloc[0])

    def pw(lam, gamma, col):
        return float(power[(power.lam == lam) & (np.isclose(power.gamma, gamma))][col].iloc[0])

    def train_rows(n):
        """Rows in the smallest training fold at sample size n."""
        return n - int(np.ceil(n / K))

    def ols_from(d):
        """Smallest n in the grid at which a d-dimensional conditioning set is fitted by least squares under the rule."""
        return min(n for n in NS if train_rows(n) >= ratio * d)

    def f2(v):
        return "--" if not np.isfinite(v) else f"{v:.2f}"

    def f3(v):
        return "--" if not np.isfinite(v) else f"{v:.3f}"

    n_e1, n_rich = ols_from(d1), ols_from(S["rich_encoder_dim"])   # smallest n with least squares for psi = E1 and for the rich set
    n_ridge_rich = [n for n in NS if n < n_rich]                      # the rich set is fitted by ridge regression at these n
    big = [n for n in NS if n >= 2000]
    nrep = S["n_rep_per_setting"]["typeI"]
    m = [
        ("resTwoN", thousands(S["n_corpus_posts"])),
        ("resTwoPopN", str(S["population_n_max"])),
        ("resTwoNrep", f"{nrep['E1|lam=1.0']} ({nrep['rich|lam=1.0']} for the high-dimensional encoder)"),
        ("resTwoNmin", thousands(n_lo)), ("resTwoNmax", thousands(n_hi)),
        ("resTwoDimWeak", str(d1)), ("resTwoDimStrong", str(d2)), ("resTwoBothDim", str(d1 + d2)),
        ("resTwoRichDim", str(S["rich_encoder_dim"])),
        ("resTwoHybridMinRowsPerCol", f"{ratio:g}"),
        ("resTwoTrainRowsSmall", str(train_rows(n_lo))), ("resTwoTrainRowsOneK", str(train_rows(1000))),
        ("resTwoRichRowsPerColAtOLS", f"{train_rows(n_rich) / S['rich_encoder_dim']:.1f}"),
        ("resTwoRichRowsPerColFourK", f"{train_rows(4000) / S['rich_encoder_dim']:.1f}"),
        ("resTwoOLSFromWeak", thousands(n_e1)), ("resTwoOLSFromRich", thousands(n_rich)),
        ("resTwoKRRMaxN", thousands(S["krr_max_n"])),
        ("resTwoRegressor", "least squares regressions"),
        ("resTwoRegressorSecondary", "kernel ridge regression with a Gaussian kernel and median-heuristic bandwidth"),
        ("resTwoPopRtwo", f"{pop['r2_A']:.3f}"), ("resTwoPopBound", f"{pop['bound']:.3f}"), ("resTwoPopBench", f"{pop['bench']:.3f}"),
        ("resTwoPopRtwoKRR", f"{pop2['r2_A']:.3f}"), ("resTwoPopBenchKRR", f"{pop2['bench']:.3f}"),
        ("resTwoTypeIOrth", f2(t('E1', 0.0, n_hi, f'rej_{P1}'))),
        ("resTwoTypeIOrthMin", f2(np.nanmin([t('E1', 0.0, n, f'rej_{P1}') for n in NS]))), ("resTwoTypeIOrthMax", f2(np.nanmax([t('E1', 0.0, n, f'rej_{P1}') for n in NS]))),
        ("resTwoTypeIOrthSmall", f2(t('E1', 0.0, n_lo, f'rej_{P1}'))), ("resTwoTypeIOrthOneK", f2(t('E1', 0.0, 1000, f'rej_{P1}'))),
        ("resTwoTypeIOrthFromTwoKMin", f2(np.nanmin([t('E1', 0.0, n, f'rej_{P1}') for n in big]))), ("resTwoTypeIOrthFromTwoKMax", f2(np.nanmax([t('E1', 0.0, n, f'rej_{P1}') for n in big]))),
        ("resTwoTypeIOrthOLSLarge", f2(t('E1', 0.0, n_hi, 'rej_ols'))), ("resTwoTypeIOrthRidgeLarge", f2(t('E1', 0.0, n_hi, 'rej_ridge'))),
        ("resTwoTypeIOrthOLSTwoK", f2(t('E1', 0.0, 2000, 'rej_ols'))), ("resTwoTypeIOrthRidgeTwoK", f2(t('E1', 0.0, 2000, 'rej_ridge'))),
        ("resTwoTypeIOrthOLSFourK", f2(t('E1', 0.0, 4000, 'rej_ols'))), ("resTwoTypeIOrthRidgeFourK", f2(t('E1', 0.0, 4000, 'rej_ridge'))),
        ("resTwoTypeIOracleMax", f2(np.nanmax([t('oracle', 1.0, n, f'rej_{P1}') for n in NS]))),
        ("resTwoNrepBench", str(S["n_rep_per_setting"]["bench"])), ("resTwoNrepPower", str(S["n_rep_per_setting"]["power"])),
        ("resTwoTypeIWeakSmall", f2(t('E1', 1.0, n_e1, f'rej_{P1}'))), ("resTwoTypeIWeakLarge", f2(t('E1', 1.0, n_hi, f'rej_{P1}'))),
        ("resTwoRhoWeak", f3(np.nanmean([t('E1', 1.0, n, f'rho_{P1}') for n in NS]))),
        ("resTwoRhoWeakSmall", f3(t('E1', 1.0, n_e1, f'rho_{P1}'))), ("resTwoRhoWeakLarge", f3(t('E1', 1.0, n_hi, f'rho_{P1}'))),
        ("resTwoRhoWeakMin", f3(np.nanmin([t('E1', 1.0, n, f'rho_{P1}') for n in NS]))), ("resTwoRhoWeakMax", f3(np.nanmax([t('E1', 1.0, n, f'rho_{P1}') for n in NS]))),
        ("resTwoTypeIOracle", f2(t('oracle', 1.0, n_hi, f'rej_{P1}'))),
        ("resTwoTypeIRichSmall", f2(t('rich', 1.0, n_rich, f'rej_{P1}'))), ("resTwoTypeIRichLarge", f2(t('rich', 1.0, n_hi, f'rej_{P1}'))),
        ("resTwoTypeIRichOLSFrom", f2(t('rich', 1.0, n_rich, f'rej_{P1}'))),
        ("resTwoTypeIRichFourK", f2(t('rich', 1.0, 4000, f'rej_{P1}'))),
        ("resTwoTypeIRichBeforeOLS", f2(t('rich', 1.0, max(n for n in NS if n < n_rich), 'rej_ridge'))),
        ("resTwoRhoRichSmall", f3(t('rich', 1.0, n_rich, f'rho_{P1}'))), ("resTwoRhoRichLarge", f3(t('rich', 1.0, n_hi, f'rho_{P1}'))),
        ("resTwoTypeIRichRidgeLarge", f2(t('rich', 1.0, n_hi, 'rej_ridge'))), ("resTwoRhoRichRidgeLarge", f3(t('rich', 1.0, n_hi, 'rho_ridge'))),
        ("resTwoTypeIRichRidgeFourK", f2(t('rich', 1.0, 4000, 'rej_ridge'))),
        ("resTwoTypeIRichRidgeSmall", f2(t('rich', 1.0, n_lo, f'rej_{P1}'))), ("resTwoTypeIRichRidgeOneK", f2(t('rich', 1.0, 1000, f'rej_{P1}'))),
        ("resTwoTypeIRichRidgeMin", f2(np.nanmin([t('rich', 1.0, n, f'rej_{P1}') for n in n_ridge_rich]))), ("resTwoTypeIRichRidgeMax", f2(np.nanmax([t('rich', 1.0, n, f'rej_{P1}') for n in n_ridge_rich]))),
        ("resTwoRhoRichRidgeSmall", f3(t('rich', 1.0, n_lo, f'rho_{P1}'))), ("resTwoRhoRichRidgeOneK", f3(t('rich', 1.0, 1000, f'rho_{P1}'))),
        ("resTwoTypeIRichOLSSmall", f2(t('rich', 1.0, n_lo, 'rej_ols'))), ("resTwoTypeIRichOLSOneK", f2(t('rich', 1.0, 1000, 'rej_ols'))),
        ("resTwoRVmed", f3(b(n_hi, 'RV_med'))), ("resTwoRVmedSmall", f3(b(n_e1, 'RV_med'))),
        ("resTwoRValphaSmall", f3(b(n_e1, 'RVa_med'))), ("resTwoRValphaLarge", f3(b(n_hi, 'RVa_med'))),
        ("resTwoBenchSampleMed", f3(rvb.bench_med.max())), ("resTwoBenchSampleMedLarge", f3(b(n_hi, 'bench_med'))),
        ("resTwoFracRobust", f"{100 * b(n_hi, 'frac_RValpha_above_bench'):.0f}\\%"),
        ("resTwoRobustBoundTypeI", f2(b(n_hi, 'robust_bound_rej'))), ("resTwoRobustBoundTypeIMax", f2(rvb.robust_bound_rej.max())),
        ("resTwoRobustBoundTypeISmall", f2(b(n_lo, 'robust_bound_rej'))), ("resTwoRobustBoundTypeIOneK", f2(b(1000, 'robust_bound_rej'))),
        ("resTwoRobustBoundTypeIFromTwoKMax", f2(max(b(n, 'robust_bound_rej') for n in big))),
        ("resTwoRobustHalfOneK", f2(b(1000, 'robust_half_rej'))),
        ("resTwoRobustHalfSmall", f2(b(n_e1, 'robust_half_rej'))), ("resTwoRobustHalfLarge", f2(b(n_hi, 'robust_half_rej'))),
        ("resTwoPowerN", str(S["power_n"])),
        ("resTwoPowerNaive", f2(pw(1.0, 0.0, 'naive'))), ("resTwoPowerNaiveOrth", f2(pw(0.0, 0.0, 'naive'))),
        ("resTwoPowerRobustBench", f2(pw(1.0, 0.1, 'robust_bench'))),
        ("resTwoPowerRobustBound", f2(pw(1.0, 0.0, 'robust_bound'))), ("resTwoPowerRobustBoundNext", f2(pw(1.0, 0.05, 'robust_bound'))),
        ("resTwoTypeIWeakPowerN", f2(pw(1.0, 0.0, 'naive'))),
        ("resTwoRobustBoundNull", f2(pw(1.0, 0.0, 'robust_bound'))), ("resTwoRobustBoundPower", f2(pw(1.0, 0.1, 'robust_bound'))),
    ]
    lines = [BEGIN, f"% written by code/experiments/exp2_macros.py from results/summary_exp2.json (primary regressor: {P1}); do not edit by hand"]
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
