"""
05_findings_report.py — writes the findings report (the communication
deliverable) from the actual pipeline outputs.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

metrics = json.loads((OUT / "model_metrics.json").read_text())
eda = json.loads((OUT / "eda" / "eda_summary.json").read_text())
profiles = pd.read_csv(OUT / "cluster_profiles.csv")

cls = pd.DataFrame(metrics["classification"])
reg = pd.DataFrame(metrics["regression"])
best_cls = cls.loc[cls["roc_auc"].idxmax()]
best_reg = reg.loc[reg["rmse_min"].idxmin()]


def to_md(df: pd.DataFrame) -> str:
    """Minimal dependency-free DataFrame -> markdown table."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


others = cls[cls["model"] != best_cls["model"]].sort_values("roc_auc",
                                                             ascending=False)
lead_txt = (f"The **{best_cls['model']}** leads on ROC-AUC "
            f"({best_cls['roc_auc']:.3f}), ahead of " +
            ", ".join(f"{r['model']} ({r['roc_auc']:.3f})"
                      for _, r in others.iterrows()))

cls_table = to_md(cls)
reg_table = to_md(reg)
prof_table = to_md(profiles)

by_dept = "\n".join(f"- **{k}**: {v:.1%}" for k, v in
                    eda["repeat_rate_by_department"].items())

md = f"""# Findings Report — Servicing Repeat-Contact Prediction & Customer Segmentation

**Prepared for:** Analyst – Data Science, Global Servicing Decision Science (GSDS)
**Dataset:** 12 months of synthetic global servicing interactions
({eda['n_interactions']:,} interactions, Phone / Chat / Email across Payments,
Billing, Technical and Account Management)

---

## 1. Executive summary

- **{eda['repeat_rate_overall']:.1%}** of servicing interactions are followed by
  another contact from the same customer within 30 days — a large, addressable
  pool of repeat work.
- First-contact resolution is the strongest behavioural lever: customers whose
  issue was **not** resolved at first contact repeat at
  **{eda['repeat_rate_fcr_0']:.1%}** vs **{eda['repeat_rate_fcr_1']:.1%}** for
  resolved customers (chi-square p = {eda['chi2_fcr_vs_repeat_pvalue']:.1e}).
- A **{best_cls['model']}** predicts repeat contact with **ROC-AUC
  {best_cls['roc_auc']:.3f}** (accuracy {best_cls['accuracy']:.1%}), giving
  servicing teams a ranked, explainable list of at-risk customers.
- Handle time is predictable from operational features
  (best RMSE **{best_reg['rmse_min']:.2f} min**, R² **{best_reg['r2']:.2f}**),
  supporting workforce planning.
- Customers fall into **4 distinct segments** (K-Means), each with a clear
  operational profile (see section 5).

## 2. Problem & data

**Problem.** Servicing organisations carry heavy repeat-contact volume. This
project builds (a) a classifier that flags interactions likely to generate a
repeat contact within 30 days, (b) a regressor that estimates handle time, and
(c) a segmentation of the customer base — the three foundational decision
science building blocks for servicing intelligence.

**Data.** Synthetic but behaviourally realistic: repeat contact is driven by
first-contact resolution, severity, department, trailing-90-day contact
frequency, tenure, channel and satisfaction — the same drivers documented in
servicing analytics practice. All behavioural features are computed from
**past data only** (SQL window functions), so the model-ready dataset is free
of target leakage.

## 3. Approach

| Task | Models |
|---|---|
| Repeat-contact classification | Logistic Regression · Decision Tree · Random Forest · Neural Network (MLP, 2 hidden layers) |
| Handle-time regression | Linear Regression · Random Forest Regressor |
| Customer segmentation | K-Means (k = 4, chosen by elbow method) on SQL-derived customer features |

Train/test split: 75/25, stratified, fixed seed. Categorical features one-hot
encoded; numeric features standardised for the linear model and the neural
network.

## 4. Results

### 4.1 Repeat-contact classification (n_test = {metrics['n_test']:,})

{cls_table}

{lead_txt}; the Random Forest
additionally provides feature importance (see `outputs/feature_importance.png`).
Top predictive features: {", ".join(metrics["top_features"][:8])}.

### 4.2 Handle-time regression (n_test = {metrics['n_test']:,})

{reg_table}

### 4.3 Exploratory findings

Repeat-contact rate by department:

{by_dept}

- Handle time is significantly longer for interactions **not** resolved at
  first contact ({eda['handle_time_mean_fcr_0']:.1f} vs
  {eda['handle_time_mean_fcr_1']:.1f} min, Welch t-test
  p = {eda['ttest_handle_time_fcr_pvalue']:.1e}).

## 5. Customer segments (K-Means, k = 4)

{prof_table}

**How to read the segments** (see `outputs/cluster_pca.png` for the 2-D view):
each row is the average profile of one segment. Segments differ mainly on
contact frequency, first-contact resolution and issue mix — i.e. on
**how much repeat work they generate** and **how expensive they are to serve**.

## 6. Key insights

1. **First-contact resolution is the #1 lever.** Repeat rate drops from
   {eda['repeat_rate_fcr_0']:.1%} to {eda['repeat_rate_fcr_1']:.1%} when the issue is
   resolved on the first contact — the single largest measurable driver.
2. **Billing & Payments dominate repeat risk** (see department table above),
   consistent with high-frequency, high-friction issue types.
3. **A ranked at-risk list is feasible.** The best classifier separates
   repeat-contact customers with ROC-AUC {best_cls['roc_auc']:.3f} — good enough
   to prioritise proactive outreach and quality checks.
4. **Handle time is forecastable** (RMSE {best_reg['rmse_min']:.2f} min), enabling
   better agent rostering and SLA setting.
5. **Four customer segments** give operations a simple lens: which customers
   are high-frequency / low-resolution / high-severity, and therefore where
   process fixes will save the most repeat contacts.

## 7. Limitations & next steps

- Data is **synthetic**; drivers are plausible but should be re-estimated on
  real servicing data before any operational use.
- The classifier uses interaction-time features only; adding customer-level
  history (e.g. product Life-cycle events) should improve ranking.
- Next steps: calibrate probabilities, set an operating threshold on the
  precision/recall trade-off, and monitor segment drift over time.

---
*Generated automatically by the pipeline (`src/05_findings_report.py`) from
`outputs/model_metrics.json`, `outputs/eda/eda_summary.json` and
`outputs/cluster_profiles.csv` — every number above is reproducible with
`python run_all.py`.*
"""
(OUT / "findings_report.md").write_text(md)
print("Findings report written -> outputs/findings_report.md")
