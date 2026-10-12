"""
02_eda.py — exploratory data analysis on the model-ready dataset,
including statistical hypothesis tests (chi-square, t-test).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "model_ready.csv")
summary = {}

# ---------------- 1. repeat rate by department ----------------
by_dept = (df.groupby("department")["repeat_contact_30d"].mean()
           .sort_values(ascending=False))
fig, ax = plt.subplots(figsize=(7, 4.2))
by_dept.plot.bar(ax=ax, color="#1F4E79")
ax.axhline(df["repeat_contact_30d"].mean(), color="#C00000", ls="--",
           label=f"overall {df['repeat_contact_30d'].mean():.1%}")
ax.set_title("Repeat-contact rate (30d) by department")
ax.set_ylabel("repeat rate"); ax.set_xlabel("")
ax.legend(); plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
fig.tight_layout(); fig.savefig(OUT / "repeat_rate_by_department.png", dpi=130)
plt.close(fig)

# ---------------- 2. repeat rate by first-contact resolution ----------------
by_fcr = (df.groupby("first_contact_resolution")["repeat_contact_30d"]
          .mean())
fig, ax = plt.subplots(figsize=(5.5, 4.2))
by_fcr.plot.bar(ax=ax, color=["#C00000", "#2E7D32"])
ax.set_xticklabels(["not resolved at first contact", "resolved at first contact"],
                   rotation=12, ha="right")
ax.set_title("Repeat-contact rate vs first-contact resolution")
ax.set_ylabel("repeat rate"); ax.set_xlabel("")
fig.tight_layout(); fig.savefig(OUT / "repeat_rate_by_fcr.png", dpi=130)
plt.close(fig)

# chi-square test: is repeat rate independent of FCR?
ct = pd.crosstab(df["first_contact_resolution"], df["repeat_contact_30d"])
chi2, p_chi, _, _ = stats.chi2_contingency(ct)

# ---------------- 3. handle time by department (boxplot) ----------------
fig, ax = plt.subplots(figsize=(7, 4.2))
data = [df.loc[df["department"] == d, "handle_time_min"].values
        for d in ["Technical", "Billing", "Account Management", "Payments"]]
ax.boxplot(data, tick_labels=["Technical", "Billing", "Account\nManagement",
                               "Payments"], showfliers=False)
ax.set_title("Handle time (minutes) by department")
ax.set_ylabel("minutes")
fig.tight_layout(); fig.savefig(OUT / "handle_time_by_department.png", dpi=130)
plt.close(fig)

# t-test: handle time, resolved vs not resolved at first contact
g0 = df.loc[df["first_contact_resolution"] == 0, "handle_time_min"]
g1 = df.loc[df["first_contact_resolution"] == 1, "handle_time_min"]
tstat, p_t = stats.ttest_ind(g0, g1, equal_var=False)

# ---------------- 4. correlation heatmap (numeric features) ----------------
num = df.select_dtypes(include=[np.number]).drop(
    columns=["interaction_id", "customer_id"])
corr = num.corr()
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("Feature correlation (numeric)")
fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=130)
plt.close(fig)

# ---------------- summary ----------------
summary = {
    "n_interactions": int(len(df)),
    "repeat_rate_overall": round(float(df["repeat_contact_30d"].mean()), 4),
    "fcr_rate_overall": round(float(df["first_contact_resolution"].mean()), 4),
    "avg_handle_time_min": round(float(df["handle_time_min"].mean()), 2),
    "avg_satisfaction": round(float(df["satisfaction_score"].mean()), 2),
    "repeat_rate_by_department": {k: round(float(v), 4)
                                  for k, v in by_dept.items()},
    "repeat_rate_fcr_0": round(float(by_fcr[0]), 4),
    "repeat_rate_fcr_1": round(float(by_fcr[1]), 4),
    "chi2_fcr_vs_repeat_pvalue": float(p_chi),
    "ttest_handle_time_fcr_pvalue": float(p_t),
    "handle_time_mean_fcr_0": round(float(g0.mean()), 2),
    "handle_time_mean_fcr_1": round(float(g1.mean()), 2),
}
(OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2))

print("EDA complete -> outputs/eda/")
print(f"  repeat rate overall      : {summary['repeat_rate_overall']:.1%}")
print(f"  repeat rate FCR=0 / FCR=1: {summary['repeat_rate_fcr_0']:.1%} / "
      f"{summary['repeat_rate_fcr_1']:.1%}  (chi2 p = {p_chi:.2e})")
print(f"  handle time FCR=0 / FCR=1: {g0.mean():.1f} / {g1.mean():.1f} min "
      f"(t-test p = {p_t:.2e})")
