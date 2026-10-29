"""Variance components of the application (Section 4.4): how much of X and Y the post identity explains, how aligned the
post-level components are, and the bias that Theorem 3.3 predicts for the embedded test with psi = E1.

Unbalanced one-way ANOVA (posts as groups) for each variable and for the cross-product; lambda_hat is the between-post
covariance divided by the between-post standard deviations. R2_psi(X) (key R2_A) = share of Var(X) explained by the post minus
the share explained by the embedding (from app_cmv_results.csv), predicted bias = lambda_hat * sqrt(R2_psi(X) R2_psi(Y)),
measured bias = rho_psi - rho_id for psi = E1.
Output: results/app_cmv_components.json.  Usage: python experiments/app_cmv_components.py
"""
import sys, json; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from citext.paths import RESULTS_DIR, CACHE_DIR
CMV = CACHE_DIR / "cmv"

def anova(df, cols, g):
    """Between/within variance components for cols (and covariance for pairs) with unbalanced groups g."""
    N, G = len(df), df[g].nunique(); ng = df.groupby(g).size().values
    n0 = (N - np.sum(ng**2) / N) / (G - 1)
    means = df.groupby(g)[cols].transform("mean"); grand = df[cols].mean()
    dev_w = df[cols] - means; dev_b = means - grand
    out = {}
    for a in cols:
        for b in cols:
            if cols.index(b) < cols.index(a): continue
            ssb = float((dev_b[a] * dev_b[b]).sum()); ssw = float((dev_w[a] * dev_w[b]).sum())
            msb, msw = ssb / (G - 1), ssw / (N - G)
            out[(a, b)] = {"between": (msb - msw) / n0, "within": msw}
    return out, {"N": int(N), "G": int(G), "n0": float(n0)}

def main():
    df = pd.read_parquet(CMV / "rows.parquet")
    rat = pd.DataFrame([json.loads(l) for l in (CMV / "ratings_haiku.jsonl").read_text().splitlines() if l.strip()])
    rat = rat.dropna(subset=["rating"]).drop_duplicates("id"); df = df.merge(rat[["id", "rating"]], on="id", how="left")
    res = pd.read_csv(RESULTS_DIR / "app_cmv_results.csv")
    out = {}
    for xn, xcol in (("politeness", "rating"), ("log_length", "log_len"), ("lexicon", "x_lex")):
        for yn, ycol in (("score", "y_score"), ("delta", "delta")):
            d = df.dropna(subset=[xcol, ycol]).copy(); d = d[d.groupby("op")["id"].transform("size") >= 2]
            comp, meta = anova(d, [xcol, ycol], "op")
            vx, vy, cxy = comp[(xcol, xcol)], comp[(ycol, ycol)], comp[(xcol, ycol)]
            share_x = vx["between"] / (vx["between"] + vx["within"]); share_y = vy["between"] / (vy["between"] + vy["within"])
            # lambda is undefined when a between-post variance component is estimated as non-positive (e.g. the delta indicator)
            lam = cxy["between"] / np.sqrt(vx["between"] * vy["between"]) if min(vx["between"], vy["between"]) > 0 else float("nan")
            r = res[(res.X == xn) & (res.Y == yn)].set_index("psi")
            rec = {**meta, "share_between_X": share_x, "share_between_Y": share_y, "lambda_hat": float(lam),
                   "within_corr": cxy["within"] / np.sqrt(vx["within"] * vy["within"])}
            if "E1" in r.index and "stratified (OP identity)" in r.index:
                # a negative out-of-fold R^2 is estimation error (least squares with few posts per column); the population
                # R^2 of the embedding is nonnegative, so the share explained by the embedding is truncated at zero, as the
                # benchmark estimates are
                r2x_emb, r2y_emb = max(0.0, float(r.loc["E1", "R2_X"])), max(0.0, float(r.loc["E1", "R2_Y"]))
                R2A, R2B = max(share_x - r2x_emb, 0.0), max(share_y - r2y_emb, 0.0)
                # The paper's R2_psi(X) divides by sigma_psi^2(X) = Var(X)(1 - share explained by the embedding); the recursion
                # formula predicts rho_psi = rho_id sqrt((1-R2_psi(X))(1-R2_psi(Y))) + lambda sqrt(R2_psi(X) R2_psi(Y)) for psi = E1.
                R2A_def, R2B_def = R2A / max(1.0 - r2x_emb, 1e-12), R2B / max(1.0 - r2y_emb, 1e-12)
                rho_s, rho_e1 = float(r.loc["stratified (OP identity)", "rho"]), float(r.loc["E1", "rho"])
                pred_rho = rho_s * np.sqrt((1 - R2A_def) * (1 - R2B_def)) + lam * np.sqrt(R2A_def * R2B_def)
                rec.update({"emb_R2_X_E1": r2x_emb, "emb_R2_Y_E1": r2y_emb, "R2_A_E1": R2A, "R2_B_E1": R2B,
                            "bound_E1": float(np.sqrt(R2A * R2B)), "predicted_bias_E1": float(lam * np.sqrt(R2A * R2B)),
                            "R2_A_E1_def": float(R2A_def), "R2_B_E1_def": float(R2B_def), "bound_E1_def": float(np.sqrt(R2A_def * R2B_def)),
                            "shrink_E1": float(np.sqrt((1 - R2A_def) * (1 - R2B_def))),
                            "predicted_rho_E1_recursion": float(pred_rho), "predicted_bias_E1_recursion": float(pred_rho - rho_s),
                            "measured_bias_E1": rho_e1 - rho_s, "rho_strat": rho_s, "rho_E1": rho_e1})
            out[f"{xn}/{yn}"] = rec
            print(f"{xn}/{yn}: N={meta['N']} G={meta['G']} shareX={share_x:.3f} shareY={share_y:.3f} lambda={lam:.3f}"
                  + (f" pred_bias={rec['predicted_bias_E1']:.4f} meas_bias={rec['measured_bias_E1']:.4f}" if "predicted_bias_E1" in rec else ""))
    json.dump(out, open(RESULTS_DIR / "app_cmv_components.json", "w"), indent=1)

if __name__ == "__main__":
    main()
