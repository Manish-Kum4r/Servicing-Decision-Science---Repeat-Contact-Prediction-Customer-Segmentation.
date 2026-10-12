"""
00_generate_data.py — synthetic Global Servicing interaction dataset.

Seeded and fully reproducible (seed = 42). Simulates 12 months of customer
servicing interactions (phone / chat / email) across Payments, Billing,
Technical and Account Management, with a realistic behavioural structure:
customers who fail first-contact resolution, face higher severity issues or
contact more often are more likely to contact again within 30 days.
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
N_CUST, N_INT, N_AGENTS = 2000, 6000, 120

# ---------------- customers ----------------
propensity = rng.gamma(2.0, 1.0, N_CUST)  # latent contact frequency (sampling weight only)
customers = pd.DataFrame({
    "customer_id": np.arange(1, N_CUST + 1),
    "tenure_months": rng.integers(1, 121, N_CUST).astype(float),
    "age": rng.integers(22, 76, N_CUST).astype(float),
    "segment": rng.choice(["Retail", "Small Business"], N_CUST, p=[0.70, 0.30]),
    "n_products": rng.integers(1, 7, N_CUST).astype(float),
    "card_tier": rng.choice(["Blue", "Gold", "Platinum"], N_CUST, p=[0.50, 0.35, 0.15]),
    "region": rng.choice(["North", "South", "East", "West", "Central", "North East"], N_CUST),
})

# ---------------- agents ----------------
agents = pd.DataFrame({
    "agent_id": np.arange(1, N_AGENTS + 1),
    "agent_experience_months": rng.integers(1, 61, N_AGENTS).astype(float),
})

# ---------------- interactions ----------------
inter = pd.DataFrame({
    "interaction_id": np.arange(1, N_INT + 1),
    "customer_id": rng.choice(customers["customer_id"], N_INT, p=propensity / propensity.sum()),
    "agent_id": rng.integers(1, N_AGENTS + 1, N_INT),
})
inter = inter.merge(customers, on="customer_id")
inter = inter.merge(agents, on="agent_id")

dates = pd.Series(pd.to_datetime("2025-01-01")
                  + pd.to_timedelta(rng.integers(0, 365, N_INT), unit="D"))
inter["interaction_date"] = dates.dt.strftime("%Y-%m-%d")
inter["is_weekend"] = (dates.dt.dayofweek >= 5).astype(int)
inter["contact_hour"] = rng.integers(8, 21, N_INT)

inter["channel"] = rng.choice(["Phone", "Chat", "Email"], N_INT, p=[0.55, 0.30, 0.15])
inter["department"] = rng.choice(["Payments", "Billing", "Technical", "Account Management"],
                                 N_INT, p=[0.25, 0.30, 0.25, 0.20])
ISSUES = {
    "Payments": ["Payment Failed", "Refund Request", "Card Declined", "EMI Query"],
    "Billing": ["Statement Dispute", "Late Fee", "Minimum Due Query", "Billing Cycle"],
    "Technical": ["Login Issue", "App Crash", "OTP Not Received", "KYC Failure"],
    "Account Management": ["Address Change", "Limit Change", "Card Upgrade", "Closure Request"],
}
inter["issue_category"] = [rng.choice(ISSUES[d]) for d in inter["department"]]
inter["severity"] = rng.choice([1, 2, 3, 4, 5], N_INT, p=[0.30, 0.30, 22 / 100, 0.12, 0.06])

# handle time (minutes) — operational drivers + noise
base = inter["department"].map({"Payments": 6.0, "Billing": 9.0,
                                "Technical": 14.0, "Account Management": 7.0})
chan_adj = inter["channel"].map({"Phone": 2.0, "Chat": 0.0, "Email": -3.0})
inter["handle_time_min"] = (
    base + 2.2 * inter["severity"] + chan_adj
    - 0.04 * inter["agent_experience_months"]
    + 1.5 * inter["is_weekend"] + 0.15 * inter["n_products"]
    + rng.normal(0, 2.5, N_INT)
).clip(lower=1.5)

# first-contact resolution probability
fcr_p = (0.78 - 0.07 * inter["severity"]
         - 0.05 * (inter["department"] == "Technical").astype(float)
         + 0.06 * (inter["channel"] == "Email").astype(float)
         + 0.002 * inter["agent_experience_months"]
         + 0.03 * (inter["segment"] == "Retail").astype(float))
inter["first_contact_resolution"] = rng.binomial(1, fcr_p.clip(0.15, 0.95))

# satisfaction (1-5, ~10% missing)
sat = (4.2 - 0.18 * inter["severity"]
       - 0.25 * (1 - inter["first_contact_resolution"])
       - 0.02 * (inter["handle_time_min"] - 8.0)
       + rng.normal(0, 0.7, N_INT))
inter["satisfaction_score"] = sat.clip(1, 5).round().astype(float)
inter.loc[rng.random(N_INT) < 0.10, "satisfaction_score"] = np.nan

# prior contacts in trailing 90 days (target driver; recomputed later in SQL)
inter = inter.sort_values(["customer_id", "interaction_date", "interaction_id"])
prior = np.zeros(N_INT, dtype=int)
for _, grp in inter.groupby("customer_id", sort=False):
    d = pd.to_datetime(grp["interaction_date"]).values.astype("datetime64[D]").astype(int)
    idx = grp.index.values
    for j in range(len(d)):
        lo = np.searchsorted(d, d[j] - 90, side="left")
        prior[idx[j]] = j - lo
inter["_prior90"] = prior

# target: customer contacts again within 30 days
logit = (-0.75
         + 1.10 * (1 - inter["first_contact_resolution"])
         + 0.45 * (inter["severity"] - 3)
         + 0.45 * inter["department"].isin(["Billing", "Payments"]).astype(float)
         + 0.08 * inter["_prior90"]
         - 0.020 * inter["tenure_months"]
         + 0.40 * (inter["channel"] == "Phone").astype(float)
         - 0.40 * (inter["satisfaction_score"].fillna(3.5) - 3.5)
         + 0.22 * inter["is_weekend"]
         - 0.15 * (inter["card_tier"] == "Platinum").astype(float)
         + rng.normal(0, 0.5, N_INT))
inter["repeat_contact_30d"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))

# ---------------- save ----------------
# interactions table keeps ONLY interaction-level columns;
# customer/agent attributes live in their own tables (SQL views do the joins)
INT_COLS = ["interaction_id", "customer_id", "agent_id", "interaction_date",
            "channel", "department", "issue_category", "severity",
            "handle_time_min", "first_contact_resolution", "satisfaction_score",
            "contact_hour", "is_weekend", "repeat_contact_30d"]
customers.to_csv(DATA / "customers.csv", index=False)
agents.to_csv(DATA / "agents.csv", index=False)
out = inter[INT_COLS].sort_values("interaction_id")
out.to_csv(DATA / "service_interactions.csv", index=False)

print(f"customers: {len(customers)} | agents: {len(agents)} | interactions: {len(out)}")
print(f"repeat_contact_30d rate : {out['repeat_contact_30d'].mean():.1%}")
print(f"avg handle time         : {out['handle_time_min'].mean():.1f} min")
print(f"first-contact-resolution: {out['first_contact_resolution'].mean():.1%}")
