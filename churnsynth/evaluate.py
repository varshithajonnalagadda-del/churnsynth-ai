"""Quality evaluation: statistical similarity, validity, relationships, churn patterns,
downstream churn-model usefulness (TSTR) and a basic privacy/copy check."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .config import CATEGORICAL_COLS, NUMERIC_COLS, SOURCE_COLUMNS
from .rules import validate_dataframe

CORR_COLS = NUMERIC_COLS + ["auto_renew", "discount_used", "churned"]
GROUP_COLS = ["plan_type", "billing_cycle", "acquisition_channel", "country", "auto_renew"]
MODEL_NUMERIC = ["tenure_months", "monthly_price", "monthly_logins", "avg_session_minutes",
                 "support_tickets", "payment_failures", "last_login_days_ago"]
MODEL_CATEGORICAL = ["plan_type", "billing_cycle", "country", "acquisition_channel",
                     "auto_renew", "discount_used"]  # subscription_status excluded: it leaks the label

WEIGHTS = {"statistical_similarity": 0.30, "relationship_preservation": 0.15,
           "validity": 0.15, "churn_pattern": 0.25, "churn_model_utility": 0.15}


def numeric_similarity(real, syn) -> dict:
    return {c: float(1 - ks_2samp(real[c], syn[c]).statistic) for c in NUMERIC_COLS}


def categorical_similarity(real, syn) -> dict:
    out = {}
    for c in CATEGORICAL_COLS:
        p = real[c].value_counts(normalize=True)
        q = syn[c].value_counts(normalize=True)
        idx = p.index.union(q.index)
        out[c] = float(1 - 0.5 * np.abs(p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).sum())
    return out


def correlation_similarity(real, syn):
    cr, cs = real[CORR_COLS].corr().to_numpy(), syn[CORR_COLS].corr().to_numpy()
    iu = np.triu_indices_from(cr, k=1)
    mad = float(np.nanmean(np.abs(cr[iu] - cs[iu])))
    return {"mean_abs_corr_diff": mad, "score": float(max(0.0, 1 - mad / 0.10))}, cr, cs


def churn_preservation(real, syn) -> dict:
    gaps, by_group = [], {}
    for g in GROUP_COLS:
        r = real.groupby(g)["churned"].mean()
        s = syn.groupby(g)["churned"].mean().reindex(r.index)
        diff = (r - s).abs().fillna(1.0)
        gaps.extend(diff.tolist())
        by_group[g] = pd.DataFrame({"real": r, "synthetic": s}).round(4)
    mad = float(np.mean(gaps))

    def ratio(df, col):
        a = df.loc[df["churned"] == 1, col].mean()
        b = df.loc[df["churned"] == 0, col].mean()
        return float(a / b) if b else float("nan")

    signals = {c: {"real_churned_vs_active_ratio": ratio(real, c),
                   "synthetic_churned_vs_active_ratio": ratio(syn, c)}
               for c in ["monthly_logins", "support_tickets", "payment_failures", "last_login_days_ago"]}
    return {
        "real_churn_rate": float(real["churned"].mean()),
        "synthetic_churn_rate": float(syn["churned"].mean()),
        "overall_rate_gap": float(abs(real["churned"].mean() - syn["churned"].mean())),
        "mean_abs_group_churn_gap": mad,
        "score": float(max(0.0, 1 - mad / 0.10)),
        "signal_ratios": signals,
        "by_group": by_group,
    }


def _features(df):
    X = df[MODEL_NUMERIC].copy()
    for c in MODEL_CATEGORICAL:
        X[c] = df[c].astype(str)
    return pd.get_dummies(X, columns=MODEL_CATEGORICAL, dtype=float)


def churn_model_utility(real_train, real_holdout, syn, seed=0) -> dict:
    """TRTR vs TSTR: train a churn model on real vs synthetic, test both on held-out real data."""
    Xh = _features(real_holdout)

    def fit_auc(train):
        X = _features(train).reindex(columns=Xh.columns, fill_value=0.0)
        clf = HistGradientBoostingClassifier(max_iter=150, random_state=seed).fit(X, train["churned"])
        return float(roc_auc_score(real_holdout["churned"], clf.predict_proba(Xh)[:, 1]))

    auc_real, auc_syn = fit_auc(real_train), fit_auc(syn)
    return {"auc_trained_on_real": auc_real, "auc_trained_on_synthetic": auc_syn,
            "score": float(min(1.0, auc_syn / auc_real)) if auc_real else 0.0}


def privacy_check(real, syn) -> dict:
    cols = [c for c in SOURCE_COLUMNS if c != "customer_id"]
    def keys(df):
        txt = df[cols].apply(lambda col: col.map(lambda v: "NA" if pd.isna(v) else str(v)))
        return txt.agg("|".join, axis=1)

    r, s = keys(real), keys(syn)
    rate = float(s.isin(set(r)).mean())
    return {"exact_copy_rate": rate, "passed": rate < 0.01}


def evaluate_synthetic(real_train, syn, real_holdout=None, seed=0) -> dict:
    num, cat = numeric_similarity(real_train, syn), categorical_similarity(real_train, syn)
    stat = float(np.mean(list(num.values()) + list(cat.values())))
    corr, cr, cs = correlation_similarity(real_train, syn)
    val = validate_dataframe(syn)
    churn = churn_preservation(real_train, syn)
    res = {
        "n_real": len(real_train), "n_synthetic": len(syn),
        "numeric_similarity": num, "categorical_similarity": cat,
        "statistical_similarity": stat,
        "correlation": corr, "validity": val,
        "churn": churn, "privacy": privacy_check(real_train, syn),
        "_corr_real": cr, "_corr_syn": cs,
    }
    comp = {"statistical_similarity": stat, "relationship_preservation": corr["score"],
            "validity": val["overall_row_pass_rate"], "churn_pattern": churn["score"]}
    if real_holdout is not None:
        res["churn_model"] = churn_model_utility(real_train, real_holdout, syn, seed)
        comp["churn_model_utility"] = res["churn_model"]["score"]
    w = {k: WEIGHTS[k] for k in comp}
    res["components"] = comp
    res["overall_score"] = float(sum(comp[k] * w[k] for k in comp) / sum(w.values()))
    return res


def public_summary(res: dict) -> dict:
    """JSON-safe subset of an evaluation result."""
    out = {k: v for k, v in res.items() if not k.startswith("_")}
    out["churn"] = {k: v for k, v in res["churn"].items() if k != "by_group"}
    out["validity"] = {"overall_row_pass_rate": res["validity"]["overall_row_pass_rate"],
                       "all_rules_passed": res["validity"]["all_rules_passed"]}
    return out


# ----------------------------------------------------------------------------- figures
def make_figures(real, syn, res, variants, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_REAL, C_SYN = "#2f5fd0", "#f08a24"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9})

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5))
    for ax, c in zip(axes[0], ["monthly_logins", "avg_session_minutes", "last_login_days_ago", "tenure_months"]):
        bins = np.histogram_bin_edges(pd.concat([real[c], syn[c]]), bins=25)
        ax.hist(real[c], bins=bins, alpha=.6, density=True, color=C_REAL, label="Real")
        ax.hist(syn[c], bins=bins, alpha=.6, density=True, color=C_SYN, label="Synthetic")
        ax.set_title(c)
    for ax, c in zip(axes[1], ["plan_type", "billing_cycle", "subscription_status", "acquisition_channel"]):
        p = real[c].value_counts(normalize=True).sort_index()
        q = syn[c].value_counts(normalize=True).reindex(p.index, fill_value=0)
        x = np.arange(len(p))
        ax.bar(x - .2, p.values, .4, color=C_REAL); ax.bar(x + .2, q.values, .4, color=C_SYN)
        ax.set_xticks(x); ax.set_xticklabels(p.index, rotation=30, ha="right"); ax.set_title(c)
    axes[0][0].legend()
    fig.suptitle("Real vs synthetic distributions", fontweight="bold"); fig.tight_layout()
    fig.savefig(f"{outdir}/distributions.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, mat, t, cmap, lim in [(axes[0], res["_corr_real"], "Real", "RdBu_r", 1),
                                   (axes[1], res["_corr_syn"], "Synthetic", "RdBu_r", 1),
                                   (axes[2], res["_corr_real"] - res["_corr_syn"], "Difference", "PuOr", .3)]:
        im = ax.imshow(mat, cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(CORR_COLS))); ax.set_yticks(range(len(CORR_COLS)))
        ax.set_xticklabels(CORR_COLS, rotation=60, ha="right", fontsize=7)
        ax.set_yticklabels(CORR_COLS, fontsize=7); ax.set_title(t); fig.colorbar(im, ax=ax, shrink=.8)
    fig.suptitle("Correlation structure", fontweight="bold"); fig.tight_layout()
    fig.savefig(f"{outdir}/correlations.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, g in zip(axes, ["plan_type", "billing_cycle", "acquisition_channel"]):
        t = res["churn"]["by_group"][g]; x = np.arange(len(t))
        ax.bar(x - .2, t["real"], .4, color=C_REAL, label="Real"); ax.bar(x + .2, t["synthetic"], .4, color=C_SYN, label="Synthetic")
        ax.set_xticks(x); ax.set_xticklabels(t.index, rotation=30, ha="right"); ax.set_title(f"Churn rate by {g}")
    axes[0].legend(); fig.tight_layout(); fig.savefig(f"{outdir}/churn_patterns.png", dpi=130); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = list(variants)
    cols = ["#9aa5b8", C_SYN]
    axes[0].bar(names, [variants[n]["overall_score"] for n in names], color=cols[:len(names)])
    axes[0].set_ylim(0, 1); axes[0].set_title("Overall quality score")
    axes[1].bar(names, [variants[n]["churn_model"]["auc_trained_on_synthetic"] for n in names], color=cols[:len(names)])
    axes[1].axhline(res["churn_model"]["auc_trained_on_real"], color=C_REAL, ls="--", label="Trained on real")
    axes[1].set_ylim(.4, 1); axes[1].set_title("Churn-model AUC on real holdout"); axes[1].legend()
    for a in axes: a.tick_params(axis="x", labelsize=8)
    fig.tight_layout(); fig.savefig(f"{outdir}/refinement_comparison.png", dpi=130); plt.close(fig)
