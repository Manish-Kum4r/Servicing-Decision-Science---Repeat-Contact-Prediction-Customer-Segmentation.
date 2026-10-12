"""
01_build_sqlite.py — loads the CSVs into SQLite, executes sql/*.sql
(schema + feature-engineering views), and exports the model-ready dataset.
"""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SQL_DIR = ROOT / "sql"
DB = DATA / "gsds_servicing.db"

TABLES = [("customers", "customers.csv"),
          ("agents", "agents.csv"),
          ("interactions", "service_interactions.csv")]


def main():
    if DB.exists():
        DB.unlink()
    conn = sqlite3.connect(DB)

    print("Executing SQL scripts ...")
    conn.executescript((SQL_DIR / "01_schema.sql").read_text())
    print("  01_schema.sql OK (tables created)")

    print("Loading CSVs into SQLite ...")
    for table, fname in TABLES:
        df = pd.read_csv(DATA / fname)
        df.to_sql(table, conn, if_exists="append", index=False)
        print(f"  {table:<14}: {len(df):,} rows")

    conn.executescript((SQL_DIR / "02_features.sql").read_text())
    print("  02_features.sql OK (views created)")

    query = (SQL_DIR / "03_model_ready.sql").read_text()
    model_ready = pd.read_sql_query(query, conn)
    model_ready.to_csv(DATA / "model_ready.csv", index=False)

    n_int = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
    n_feat = conn.execute(
        "SELECT COUNT(*) FROM v_interaction_features").fetchone()[0]
    conn.close()

    print(f"\nModel-ready dataset : {len(model_ready):,} rows x "
          f"{model_ready.shape[1]} cols -> data/model_ready.csv")
    print(f"Interactions in DB  : {n_int:,} | enriched view rows: {n_feat:,}")
    print("Sample engineered features (first 3 rows):")
    print(model_ready[["interaction_id", "prior_contacts_90d",
                       "prior_fcr_rate_90d", "repeat_contact_30d"]].head(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
