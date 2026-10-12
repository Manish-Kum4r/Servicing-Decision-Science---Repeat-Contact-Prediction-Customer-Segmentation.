-- 03_model_ready.sql — model-ready dataset for the repeat-contact classifier
--
-- One row per interaction. Target: repeat_contact_30d.
-- All behavioural features are computed from PAST data only (see 02_features.sql),
-- so the dataset is safe to use for training without target leakage.

SELECT
    interaction_id,
    customer_id,
    repeat_contact_30d,
    -- interaction attributes (known at / right after the interaction)
    channel,
    department,
    issue_category,
    severity,
    handle_time_min,
    first_contact_resolution,
    satisfaction_score,
    contact_hour,
    is_weekend,
    -- static customer attributes
    tenure_months,
    age,
    segment,
    n_products,
    card_tier,
    region,
    -- agent attribute
    agent_experience_months,
    -- SQL-engineered trailing-90-day behavioural features
    COALESCE(prior_contacts_90d, 0)          AS prior_contacts_90d,
    COALESCE(prior_avg_severity_90d, 3.0)     AS prior_avg_severity_90d,
    COALESCE(prior_avg_handle_time_90d, 8.0) AS prior_avg_handle_time_90d,
    COALESCE(prior_fcr_rate_90d, 0.7)        AS prior_fcr_rate_90d
FROM v_interaction_features
ORDER BY interaction_id;
