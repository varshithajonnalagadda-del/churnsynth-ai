"""End-to-end pipeline: source data -> SDV/copula generation -> evaluation -> reports.

Run:  python scripts/run_pipeline.py
Outputs: data/*.csv, data/*.json, reports/*.md, reports/figures/*.png
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from churnsynth.config import COLUMN_DESCRIPTIONS, SOURCE_COLUMNS  # noqa: E402
from churnsynth.evaluate import WEIGHTS, evaluate_synthetic, make_figures, public_summary  # noqa: E402
from churnsynth.generator import GaussianCopulaGenerator, get_generator  # noqa: E402
from churnsynth.rules import RULE_DESCRIPTIONS, validate_dataframe  # noqa: E402
from churnsynth.source_data import generate_source_dataset  # noqa: E402

DATA, REPORTS, FIGS = ROOT / "data", ROOT / "reports", ROOT / "reports" / "figures"
for p in (DATA, REPORTS, FIGS):
    p.mkdir(parents=True, exist_ok=True)
SEED = 42


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ Week 1: source data
src = generate_source_dataset(6000, seed=SEED)
src.to_csv(DATA / "source_subscription_data.csv", index=False)
src_val = validate_dataframe(src)
schema = {c: {"dtype": str(src[c].dtype), "description": COLUMN_DESCRIPTIONS[c],
              "missing": int(src[c].isna().sum()), "unique": int(src[c].nunique())} for c in SOURCE_COLUMNS}
(DATA / "schema.json").write_text(json.dumps(schema, indent=2))

profile = pd.DataFrame({k: {"dtype": v["dtype"], "unique": v["unique"], "missing": v["missing"],
                            "description": v["description"]} for k, v in schema.items()}).T
profile.index.name = "column"
churn_by_plan = src.groupby("plan_type")["churned"].agg(["count", "mean"]).rename(columns={"mean": "churn_rate"})
churn_by_cycle = src.groupby("billing_cycle")["churned"].agg(["count", "mean"]).rename(columns={"mean": "churn_rate"})
signals = src.groupby("churned")[["monthly_logins", "avg_session_minutes", "support_tickets",
                                  "payment_failures", "last_login_days_ago", "tenure_months"]].mean()
signals.index = ["Active (0)", "Churned (1)"]
rules_tbl = pd.DataFrame({k: {"rule": v["description"], "pass rate": v["pass_rate"]}
                          for k, v in src_val["rules"].items()}).T

week1 = f"""# Subscription Churn Data Requirements and Dataset Preparation Report (Week 1)

*Project: ChurnSynth AI - Synthetic Subscription Churn Generator*  |  Generated {datetime.now(timezone.utc):%Y-%m-%d}

## 1. Problem and objective
Subscription businesses want to test churn-prediction models across varied customer lifecycle scenarios,
but real churn data for new product lines is limited. ChurnSynth AI learns statistical patterns from an
**approved sample dataset** with the Synthetic Data Vault (SDV) framework and generates synthetic customer
subscription records that preserve the relationships between customer lifecycle, subscription activity,
engagement, payment behaviour and churn outcomes.

## 2. Customer lifecycle scenarios covered
| Scenario | How it appears in the data |
|---|---|
| New / young customers | `tenure_months` small, recent `signup_date` |
| Loyal long-tenure customers | active, high `monthly_logins`, auto-renew on, annual billing |
| Payment-trouble customers | `payment_failures` > 0, higher churn risk |
| Disengaged customers | low `monthly_logins`/`avg_session_minutes`, high `last_login_days_ago` |
| Support-heavy customers | many `support_tickets`, higher churn risk |
| Voluntary churn | `subscription_status = cancelled` |
| Passive churn | `subscription_status = expired` (more likely when auto-renew is off) |
| Paused customers | `subscription_status = paused` (not churned) |

## 3. Data requirements
**Customer attributes:** `customer_id`, `country`, `acquisition_channel`, `signup_date`.
**Subscription attributes:** `plan_type`, `billing_cycle`, `monthly_price`, `auto_renew`, `discount_used`, `tenure_months`.
**Engagement:** `monthly_logins`, `avg_session_minutes`, `last_login_days_ago`, `support_tickets`.
**Payment:** `payment_failures`.
**Churn indicators:** `subscription_status`, `churned` (label), `churn_date`.

## 4. Source dataset (approved sample)
- **Rows:** {len(src):,}  |  **Columns:** {len(src.columns)}  |  **Snapshot date:** 2026-01-01
- **Overall churn rate:** {src['churned'].mean():.1%}  |  **Missing values:** only `churn_date` (empty for non-churned customers, by design)
- The sample is a simulated stand-in for an approved dataset (no real customer data is used), built so that churn depends on support tickets, payment failures, auto-renew, billing cycle, plan and channel.
- File: `data/source_subscription_data.csv`

### Validated schema
{md_table(profile)}

## 5. Business rules (checked on every dataset, source and synthetic)
{md_table(rules_tbl, "{:.1%}")}

All rules pass on the source data: **{src_val['all_rules_passed']}**.

## 6. Exploratory findings
**Churn by plan**

{md_table(churn_by_plan)}

**Churn by billing cycle**

{md_table(churn_by_cycle)}

**Mean behaviour of active vs churned customers**

{md_table(signals)}

Key takeaways: churned customers log in less, have longer gaps since last login, raise more support tickets and
suffer more payment failures; annual and auto-renew customers churn less. Basic plans churn more than Premium.
These are exactly the relationships the synthetic data must preserve.

## 7. Data-quality requirements for generated data
1. 100% of generated rows satisfy all business rules in section 5.
2. Per-column distributions close to the source (similarity >= 0.95 on average).
3. Correlation structure and churn-by-segment patterns preserved.
4. A churn model trained on synthetic data must still rank real customers sensibly (AUC close to a model trained on real data).
5. No exact copies of source records.

## 8. Synthetic-data generation objectives
Generate any number of records (up to 100,000), reproducibly (seed), optionally at a **target churn rate**,
returning metadata with every dataset. Design details are in `README.md`.
"""
(REPORTS / "week1_requirements_and_dataset_report.md").write_text(week1)
print("Week 1 done: source rows", len(src), "churn", round(src.churned.mean(), 3))

# ------------------------------------------------------------------ Week 2: generation pipeline
train, holdout = train_test_split(src, test_size=0.3, stratify=src["churned"], random_state=SEED)
train, holdout = train.reset_index(drop=True), holdout.reset_index(drop=True)

baseline = GaussianCopulaGenerator(independent=True, seed=SEED).fit(train)
syn_base = baseline.sample(len(train), seed=SEED)

final_gen = get_generator("auto", seed=SEED).fit(train)
syn_final = final_gen.sample(len(train), seed=SEED)

# release dataset: model fitted on the full approved sample
release_gen = get_generator("auto", seed=SEED).fit(src)
release = release_gen.sample(5000, seed=SEED)
release.to_csv(DATA / "synthetic_churn_data.csv", index=False, date_format="%Y-%m-%d")
release_val = validate_dataframe(release)

gen_meta = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "generator": release_gen.name,
    "sdv_installed": "sdv" in sys.modules or __import__("importlib").util.find_spec("sdv") is not None,
    "seed": SEED,
    "source_file": "data/source_subscription_data.csv",
    "source_rows": len(src),
    "synthetic_rows": len(release),
    "source_churn_rate": round(float(src.churned.mean()), 4),
    "synthetic_churn_rate": round(float(release.churned.mean()), 4),
    "modelled_columns": ["signup_date", "tenure (as ratio of lifetime)", "plan_type", "billing_cycle", "country",
                         "acquisition_channel", "auto_renew", "discount_used", "monthly_logins",
                         "avg_session_minutes", "support_tickets", "payment_failures",
                         "last_login_days_ago", "subscription_status"],
    "derived_columns": ["customer_id", "monthly_price", "tenure_months", "churned", "churn_date"],
    "business_rules_passed": release_val["all_rules_passed"],
    "rows_passing_all_rules": release_val["overall_row_pass_rate"],
}
(DATA / "generation_metadata.json").write_text(json.dumps(gen_meta, indent=2))
print("Week 2 done:", gen_meta["generator"], "release rows", len(release))

# ------------------------------------------------------------------ Week 3: evaluation
res_base = evaluate_synthetic(train, syn_base, holdout, SEED)
res_final = evaluate_synthetic(train, syn_final, holdout, SEED)
variants = {"v1 independent columns": res_base, "v2 copula + rules": res_final}
make_figures(train, syn_final, res_final, variants, FIGS)

summary = {"final": public_summary(res_final), "baseline": public_summary(res_base)}
(REPORTS / "evaluation_results.json").write_text(json.dumps(summary, indent=2, default=float))

comp_tbl = pd.DataFrame({
    "weight": WEIGHTS,
    "v1 independent": res_base["components"],
    "v2 final": res_final["components"],
})
comp_tbl.loc["OVERALL (weighted)"] = [1.0, res_base["overall_score"], res_final["overall_score"]]
comp_tbl.index.name = "component"
num_tbl = pd.DataFrame({"similarity (1-KS)": res_final["numeric_similarity"]}); num_tbl.index.name = "numeric column"
cat_tbl = pd.DataFrame({"similarity (1-TVD)": res_final["categorical_similarity"]}); cat_tbl.index.name = "categorical column"
rule_tbl = pd.DataFrame({k: {"rule": RULE_DESCRIPTIONS[k], "pass rate": v["pass_rate"]}
                         for k, v in validate_dataframe(syn_final)["rules"].items()}).T
sig = pd.DataFrame(res_final["churn"]["signal_ratios"]).T
sig.index.name = "feature (churned / active mean)"
cm_b, cm_f = res_base["churn_model"], res_final["churn_model"]
worst_cat = min(res_final["categorical_similarity"], key=res_final["categorical_similarity"].get)
worst_num = min(res_final["numeric_similarity"], key=res_final["numeric_similarity"].get)

week3 = f"""# Synthetic Subscription Churn Data Quality and Evaluation Report (Week 3)

*Generator:* `{final_gen.name}`  |  *Source (train split):* {len(train):,} rows  |  *Synthetic:* {len(syn_final):,} rows  |  *Held-out real data:* {len(holdout):,} rows

## 1. Method
The generator is fitted on a 70% train split of the source data. Evaluation compares synthetic data with that train
split, and uses the untouched 30% holdout only for the downstream churn-model test, so no model ever sees the holdout.

| Dimension | Test | Weight |
|---|---|---|
| Statistical similarity | 1 - KS statistic (numeric), 1 - total variation distance (categorical) | 30% |
| Relationship preservation | mean absolute difference of pairwise correlations (score = 1 - diff/0.10) | 15% |
| Validity | share of rows passing all 12 business rules | 15% |
| Churn-pattern preservation | churn-rate gap per plan / cycle / channel / country / auto-renew (score = 1 - gap/0.10) | 25% |
| Churn-model utility (TSTR) | AUC of a model trained on synthetic data, scored on real holdout, relative to a model trained on real data | 15% |

## 2. Results summary
{md_table(comp_tbl)}

Churn-model AUC on real holdout - trained on **real**: {cm_f['auc_trained_on_real']:.3f}; trained on **synthetic v1**: {cm_b['auc_trained_on_synthetic']:.3f};
trained on **synthetic v2 (final)**: {cm_f['auc_trained_on_synthetic']:.3f}.

![Refinement comparison](figures/refinement_comparison.png)

## 3. Statistical similarity
{md_table(num_tbl)}

{md_table(cat_tbl)}

Weakest numeric column: `{worst_num}` ({res_final['numeric_similarity'][worst_num]:.3f}); weakest categorical column: `{worst_cat}` ({res_final['categorical_similarity'][worst_cat]:.3f}).

![Distributions](figures/distributions.png)

## 4. Relationship preservation
Mean absolute correlation difference: **{res_final['correlation']['mean_abs_corr_diff']:.3f}** (v1: {res_base['correlation']['mean_abs_corr_diff']:.3f}).

![Correlations](figures/correlations.png)

## 5. Validity (business rules)
{md_table(rule_tbl, "{:.1%}")}

## 6. Churn-pattern preservation
Churn rate: real **{res_final['churn']['real_churn_rate']:.1%}** vs synthetic **{res_final['churn']['synthetic_churn_rate']:.1%}**.
Mean absolute churn-rate gap across segments: **{res_final['churn']['mean_abs_group_churn_gap']:.3f}**.

{md_table(sig)}

![Churn patterns](figures/churn_patterns.png)

## 7. Privacy / copy check
Exact copies of source records (all columns except the id): **{res_final['privacy']['exact_copy_rate']:.2%}** - {"passed" if res_final['privacy']['passed'] else "FAILED"}.

## 8. Issues identified and refinement activities
| # | Issue found in v1 (independent column sampling) | Refinement applied in v2 | Effect |
|---|---|---|---|
| 1 | Column relationships lost: correlation gap {res_base['correlation']['mean_abs_corr_diff']:.3f}, so churn drivers (logins, tickets, payment failures) no longer predict churn | Gaussian copula learns the joint dependence between all columns | gap {res_base['correlation']['mean_abs_corr_diff']:.3f} -> {res_final['correlation']['mean_abs_corr_diff']:.3f} |
| 2 | Churn model trained on v1 data generalises poorly (AUC {cm_b['auc_trained_on_synthetic']:.3f} vs {cm_f['auc_trained_on_real']:.3f} on real) | Same copula fix | AUC {cm_b['auc_trained_on_synthetic']:.3f} -> {cm_f['auc_trained_on_synthetic']:.3f} |
| 3 | Categorical columns are treated as *ordered* by a copula, which breaks category-churn dependence (e.g. `paused` sits between `active` and `cancelled` alphabetically) | Categories ordered by observed churn rate before encoding | stronger churn-pattern and relationship scores |
| 4 | Derived fields could contradict each other (price vs plan, churn flag vs status, tenure vs signup date) | Price, churn flag, tenure bounds, churn date and ids are rebuilt from business rules, never sampled | validity {res_final['validity']['overall_row_pass_rate']:.0%} |
| 5 | Tenure of active customers is fully determined by signup date and distorts the learned tenure distribution | Tenure modelled as a ratio of possible lifetime, learned from churned customers only | tenure similarity {res_final['numeric_similarity']['tenure_months']:.3f} |
| 6 | Need to test models at other churn rates | Conditional (rejection) sampling on the churn label to hit an exact requested rate | `target_churn_rate` API parameter |

**Remaining limitations:** the relationship score is the weakest dimension (a Gaussian copula captures monotonic,
pairwise dependence, not complex interactions); correlations involving discrete columns are slightly attenuated; and the
source data is itself a simulated approximation of real subscription behaviour, so conclusions drawn from the synthetic
data are only as good as that sample.

## 9. Conclusion
Overall quality score **{res_final['overall_score']:.3f}** (v1: {res_base['overall_score']:.3f}).
The final generator produces rule-consistent records that preserve distributions and churn patterns well enough to
exercise churn-prediction pipelines, with no copied records.
"""
(REPORTS / "week3_quality_and_evaluation_report.md").write_text(week3)
print("Week 3 done: overall v1 %.3f -> v2 %.3f" % (res_base["overall_score"], res_final["overall_score"]))
print(json.dumps({k: round(v, 3) for k, v in res_final["components"].items()}))
print("AUC real/v1/v2: %.3f %.3f %.3f" % (cm_f["auc_trained_on_real"], cm_b["auc_trained_on_synthetic"], cm_f["auc_trained_on_synthetic"]))
