"""
03_train_models.py — foundational machine learning, exactly the model families
named in the role's minimum qualifications:

  Classification (repeat contact within 30 days):
    * Logistic Regression        (classification)
    * Decision Tree              (tree-based)
    * Random Forest              (tree-based ensemble)
    * Neural Network (MLP)       (deep-learning exposure: single hidden layer)
  Regression (handle time in minutes):
    * Linear Regression          (regression)
    * Random Forest Regressor    (tree-based)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

TARGET_CLS = "repeat_contact_30d"
TARGET_REG = "handle_time_min"
ID_COLS = ["interaction_id", "customer_id"]

df = pd.read_csv(DATA / "model_ready.csv")

# ---------------- feature sets ----------------
# classification: everything except ids and the classification target
# (handle_time_min is known right after the interaction -> legitimate feature)
cls_df = df.drop(columns=ID_COLS + [TARGET_CLS])
# regression: exclude BOTH targets (repeat_contact_30d happens AFTER handle time)
reg_df = df.drop(columns=ID_COLS + [TARGET_CLS, TARGET_REG])

Xc = pd.get_dummies(cls_df, drop_first=True)
yc = df[TARGET_CLS]
Xr = pd.get_dummies(reg_df, drop_first=True)
yr = df[TARGET_REG]

Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(
    Xc, yc, test_size=0.25, random_state=42, stratify=yc)
Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(
    Xr, yr, test_size=0.25, random_state=42)

# impute missing values (satisfaction_score has ~10% missing) before scaling
imp_c = SimpleImputer(strategy="median")
Xtr_c = imp_c.fit_transform(Xtr_c)
Xte_c = imp_c.transform(Xte_c)
imp_r = SimpleImputer(strategy="median")
Xtr_r = imp_r.fit_transform(Xtr_r)
Xte_r = imp_r.transform(Xte_r)

scaler = StandardScaler().fit(Xtr_c)
Xtr_cs, Xte_cs = scaler.transform(Xtr_c), scaler.transform(Xte_c)

# ================= CLASSIFICATION =================
classifiers = {
    "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42,
                                               class_weight="balanced"), True),
    "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42,
                                             class_weight="balanced"), False),
    "Random Forest": (RandomForestClassifier(n_estimators=200, max_depth=8,
                                             random_state=42, n_jobs=-1,
                                             class_weight="balanced"), False),
    "Neural Network (MLP)": (MLPClassifier(hidden_layer_sizes=(64, 32),
                                            max_iter=1000,
                                            random_state=42), True),
}

cls_rows, roc_data = [], {}
for name, (model, needs_scale) in classifiers.items():
    Xt, Xe = (Xtr_cs, Xte_cs) if needs_scale else (Xtr_c, Xte_c)
    if name == "Neural Network (MLP)":
        # balance classes by random oversampling of the minority class
        # (MLPClassifier supports neither class_weight nor sample_weight)
        rng_bal = np.random.default_rng(42)
        pos_idx = np.where(ytr_c.values == 1)[0]
        neg_idx = np.where(ytr_c.values == 0)[0]
        extra = rng_bal.choice(pos_idx, size=len(neg_idx) - len(pos_idx),
                               replace=True)
        up = np.concatenate([np.arange(len(ytr_c)), extra])
        model.fit(Xt[up], ytr_c.values[up])
    else:
        model.fit(Xt, ytr_c)
    pred = model.predict(Xte_c if not needs_scale else Xte_cs)
    proba = model.predict_proba(Xte_c if not needs_scale else Xte_cs)[:, 1]
    cls_rows.append({
        "model": name,
        "accuracy": round(float(accuracy_score(yte_c, pred)), 4),
        "precision": round(float(precision_score(yte_c, pred)), 4),
        "recall": round(float(recall_score(yte_c, pred)), 4),
        "f1": round(float(f1_score(yte_c, pred)), 4),
        "roc_auc": round(float(roc_auc_score(yte_c, proba)), 4),
    })
    fpr, tpr, _ = roc_curve(yte_c, proba)
    roc_data[name] = (fpr.tolist(), tpr.tolist())

cls_metrics = pd.DataFrame(cls_rows)
cls_metrics.to_csv(OUT / "classification_metrics.csv", index=False)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, (fpr, tpr) in roc_data.items():
    auc = next(r["roc_auc"] for r in cls_rows if r["model"] == name)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("ROC curves — repeat-contact prediction")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "roc_curves.png", dpi=130)
plt.close(fig)

# feature importance (Random Forest)
rf = classifiers["Random Forest"][0].fit(Xtr_c, ytr_c)
imp = (pd.Series(rf.feature_importances_, index=Xc.columns)
       .sort_values(ascending=False).head(12))
fig, ax = plt.subplots(figsize=(7, 4.5))
imp[::-1].plot.barh(ax=ax, color="#1F4E79")
ax.set_title("Top 12 features — Random Forest importance")
ax.set_xlabel("importance")
fig.tight_layout(); fig.savefig(OUT / "feature_importance.png", dpi=130)
plt.close(fig)

# ================= REGRESSION =================
regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=200, max_depth=10,
                                                     random_state=42, n_jobs=-1),
}
reg_rows = []
reg_preds = {}
for name, model in regressors.items():
    model.fit(Xtr_r, ytr_r)
    pred = model.predict(Xte_r)
    reg_rows.append({
        "model": name,
        "rmse_min": round(float(np.sqrt(mean_squared_error(yte_r, pred))), 3),
        "mae_min": round(float(mean_absolute_error(yte_r, pred)), 3),
        "r2": round(float(r2_score(yte_r, pred)), 4),
    })
    reg_preds[name] = pred

reg_metrics = pd.DataFrame(reg_rows)
reg_metrics.to_csv(OUT / "regression_metrics.csv", index=False)

best_reg = reg_metrics.loc[reg_metrics["rmse_min"].idxmin(), "model"]
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(yte_r, reg_preds[best_reg], s=8, alpha=0.4)
ax.plot([yte_r.min(), yte_r.max()], [yte_r.min(), yte_r.max()], "r--")
ax.set_xlabel("actual handle time (min)"); ax.set_ylabel("predicted (min)")
ax.set_title(f"Actual vs predicted — {best_reg}")
fig.tight_layout(); fig.savefig(OUT / "regression_actual_vs_pred.png", dpi=130)
plt.close(fig)

# ---------------- save combined metrics ----------------
payload = {
    "classification": cls_rows,
    "regression": reg_rows,
    "n_train": int(len(Xtr_c)),
    "n_test": int(len(Xte_c)),
    "n_features": int(Xc.shape[1]),
    "positive_rate_test": round(float(yte_c.mean()), 4),
    "top_features": imp.index.tolist(),
}
(OUT / "model_metrics.json").write_text(json.dumps(payload, indent=2))

print("Classification (repeat_contact_30d):")
print(cls_metrics.to_string(index=False))
print("\nRegression (handle_time_min):")
print(reg_metrics.to_string(index=False))
print(f"\nBest classifier by ROC-AUC : "
      f"{cls_metrics.loc[cls_metrics['roc_auc'].idxmax(), 'model']}")
print(f"Best regressor by RMSE     : {best_reg}")
