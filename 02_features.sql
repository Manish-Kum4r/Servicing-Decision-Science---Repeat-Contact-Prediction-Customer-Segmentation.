-- 02_features.sql — feature engineering in SQL
--
-- v_interaction_features : one row per interaction, enriched with
--   * static customer & agent attributes
--   * trailing-90-day behavioural features (window functions, strictly past data)
-- v_customer_features    : one row per customer with lifetime aggregates
--                           (used for customer segmentation)

CREATE VIEW v_interaction_features AS
SELECT
    i.interaction_id,
    i.customer_id,
    i.agent_id,
    i.interaction_date,
    i.channel,
    i.department,
    i.issue_category,
    i.severity,
    i.handle_time_min,
    i.first_contact_resolution,
    i.satisfaction_score,
    i.contact_hour,
    i.is_weekend,
    i.repeat_contact_30d,
    -- static attributes
    c.tenure_months,
    c.age,
    c.segment,
    c.n_products,
    c.card_tier,
    c.region,
    a.agent_experience_months,
    -- trailing 90-day behavioural features (past data only -> no target leakage)
    COUNT(*)  OVER w90 AS prior_contacts_90d,
    AVG(i.severity)               OVER w90 AS prior_avg_severity_90d,
    AVG(i.handle_time_min)        OVER w90 AS prior_avg_handle_time_90d,
    AVG(i.first_contact_resolution) OVER w90 AS prior_fcr_rate_90d
FROM interactions i
JOIN customers c ON c.customer_id = i.customer_id
JOIN agents    a ON a.agent_id    = i.agent_id
WINDOW w90 AS (
    PARTITION BY i.customer_id
    ORDER BY julianday(i.interaction_date)
    RANGE BETWEEN 90 PRECEDING AND 1 PRECEDING
);

CREATE VIEW v_customer_features AS
SELECT
    c.customer_id,
    c.tenure_months,
    c.segment,
    c.n_products,
    c.card_tier,
    COUNT(i.interaction_id)                                        AS total_interactions,
    AVG(i.handle_time_min)                                         AS avg_handle_time_min,
    AVG(i.first_contact_resolution)                                AS fcr_rate,
    AVG(i.severity)                                                AS avg_severity,
    AVG(i.satisfaction_score)                                      AS avg_satisfaction,
    AVG(CASE WHEN i.department IN ('Billing', 'Payments')
             THEN 1.0 ELSE 0.0 END)                                AS billing_payments_share,
    AVG(CASE WHEN i.channel = 'Phone' THEN 1.0 ELSE 0.0 END)       AS phone_share
FROM customers c
JOIN interactions i ON i.customer_id = c.customer_id
GROUP BY c.customer_id, c.tenure_months, c.segment, c.n_products, c.card_tier;
