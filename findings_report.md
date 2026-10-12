# Findings Report — Servicing Repeat-Contact Prediction & Customer Segmentation

**Prepared for:** Analyst – Data Science, Global Servicing Decision Science (GSDS)
**Dataset:** 12 months of synthetic global servicing interactions
(6,000 interactions, Phone / Chat / Email across Payments,
Billing, Technical and Account Management)

---

## 1. Executive summary

- **25.9%** of servicing interactions are followed by
  another contact from the same customer within 30 days — a large, addressable
  pool of repeat work.
- First-contact resolution is the strongest behavioural lever: customers whose
  issue was **not** resolved at first contact repeat at
  **42.4%** vs **18.2%** for
  resolved customers (chi-square p = 1.2e-87).
- A **Logistic Regression** predicts repeat contact with **ROC-AUC
  0.780** (accuracy 72.1%), giving
  servicing teams a ranked, explainable list of at-risk customers.
- Handle time is predictable from operational features
  (best RMSE **2.55 min**, R² **0.76**),
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

### 4.1 Repeat-contact classification (n_test = 1,500)

| model | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|
| Logistic Regression | 0.7207 | 0.4736 | 0.7165 | 0.5703 | 0.7803 |
| Decision Tree | 0.72 | 0.4681 | 0.6057 | 0.5281 | 0.684 |
| Random Forest | 0.7593 | 0.5316 | 0.5851 | 0.5571 | 0.7679 |
| Neural Network (MLP) | 0.7033 | 0.4293 | 0.4459 | 0.4374 | 0.6772 |

The **Logistic Regression** leads on ROC-AUC (0.780), ahead of Random Forest (0.768), Decision Tree (0.684), Neural Network (MLP) (0.677); the Random Forest
additionally provides feature importance (see `outputs/feature_importance.png`).
Top predictive features: tenure_months, severity, first_contact_resolution, handle_time_min, satisfaction_score, age, agent_experience_months, prior_avg_handle_time_90d.

### 4.2 Handle-time regression (n_test = 1,500)

| model | rmse_min | mae_min | r2 |
|---|---|---|---|
| Linear Regression | 2.551 | 2.009 | 0.7612 |
| Random Forest Regressor | 2.71 | 2.135 | 0.7305 |

### 4.3 Exploratory findings

Repeat-contact rate by department:

- **Billing**: 28.5%
- **Payments**: 28.5%
- **Technical**: 23.4%
- **Account Management**: 21.8%

- Handle time is significantly longer for interactions **not** resolved at
  first contact (15.7 vs
  14.2 min, Welch t-test
  p = 4.7e-23).

## 5. Customer segments (K-Means, k = 4)

| cluster | customers | share | tenure_months | n_products | total_interactions | avg_handle_time_min | fcr_rate | avg_severity | billing_payments_share | phone_share |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.0 | 381.0 | 0.227 | 66.927 | 3.882 | 2.496 | 18.52 | 0.509 | 3.21 | 0.424 | 0.625 |
| 1.0 | 524.0 | 0.312 | 50.479 | 3.302 | 2.586 | 14.466 | 0.769 | 1.917 | 0.537 | 0.806 |
| 2.0 | 449.0 | 0.268 | 66.648 | 3.227 | 2.486 | 12.002 | 0.721 | 2.107 | 0.631 | 0.198 |
| 3.0 | 324.0 | 0.193 | 63.117 | 3.79 | 7.957 | 14.51 | 0.691 | 2.312 | 0.567 | 0.547 |

**How to read the segments** (see `outputs/cluster_pca.png` for the 2-D view):
each row is the average profile of one segment. Segments differ mainly on
contact frequency, first-contact resolution and issue mix — i.e. on
**how much repeat work they generate** and **how expensive they are to serve**.

## 6. Key insights

1. **First-contact resolution is the #1 lever.** Repeat rate drops from
   42.4% to 18.2% when the issue is
   resolved on the first contact — the single largest measurable driver.
2. **Billing & Payments dominate repeat risk** (see department table above),
   consistent with high-frequency, high-friction issue types.
3. **A ranked at-risk list is feasible.** The best classifier separates
   repeat-contact customers with ROC-AUC 0.780 — good enough
   to prioritise proactive outreach and quality checks.
4. **Handle time is forecastable** (RMSE 2.55 min), enabling
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
