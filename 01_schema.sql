-- 01_schema.sql — Global Servicing interaction warehouse (standard SQL / SQLite dialect)

CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    tenure_months   REAL    NOT NULL,
    age             REAL    NOT NULL,
    segment         TEXT    NOT NULL,
    n_products      REAL    NOT NULL,
    card_tier       TEXT    NOT NULL,
    region          TEXT    NOT NULL
);

CREATE TABLE agents (
    agent_id                  INTEGER PRIMARY KEY,
    agent_experience_months   REAL NOT NULL
);

CREATE TABLE interactions (
    interaction_id          INTEGER PRIMARY KEY,
    customer_id             INTEGER NOT NULL REFERENCES customers(customer_id),
    agent_id                INTEGER NOT NULL REFERENCES agents(agent_id),
    interaction_date        TEXT    NOT NULL,          -- ISO date YYYY-MM-DD
    channel                 TEXT    NOT NULL,          -- Phone / Chat / Email
    department              TEXT    NOT NULL,          -- Payments / Billing / Technical / Account Management
    issue_category          TEXT    NOT NULL,
    severity                INTEGER NOT NULL,          -- 1 (low) .. 5 (critical)
    handle_time_min         REAL    NOT NULL,
    first_contact_resolution INTEGER NOT NULL,         -- 0 / 1
    satisfaction_score      REAL,                      -- 1..5, nullable
    contact_hour            INTEGER NOT NULL,          -- 8..20
    is_weekend              INTEGER NOT NULL,          -- 0 / 1
    repeat_contact_30d      INTEGER NOT NULL           -- TARGET: contacted again within 30 days
);

CREATE INDEX idx_interactions_cust_date ON interactions(customer_id, interaction_date);
CREATE INDEX idx_interactions_dept      ON interactions(department);
