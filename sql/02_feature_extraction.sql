-- ============================================================
-- 02_feature_extraction.sql
-- Healthcare Claims Cohort and Care Management Outcome Evaluation
--
-- Purpose:
--   Join cohort staging results with patient demographics,
--   chronic condition burden, and prior 12-month utilization
--   history to build the analytical feature set for logistic
--   regression and chi-square testing.
--
-- Input: cohort_staging (output of 01_cohort_selection.sql)
--
-- Schema:
--   claims_db.patient_demographics (patient_id, age, gender)
--   claims_db.patient_conditions   (patient_id, condition_code,
--                                    chronic_condition_flag)
--   claims_db.inpatient_claims     (patient_id, claim_id,
--                                    admission_date)
--
-- SQL techniques:
--   Multi-table JOIN, LEFT JOIN with aggregated subqueries,
--   COALESCE for null-safe aggregation, CASE WHEN for bucketing,
--   COUNT DISTINCT for condition burden
-- ============================================================

SELECT
    -- Core cohort fields
    c.patient_id,
    c.index_claim_id,
    c.index_discharge_date,
    c.primary_diagnosis,
    c.care_management_flag,
    c.length_of_stay,
    c.index_cost,
    c.readmit_cost,

    -- Target variable
    c.readmitted_30d,

    -- Patient demographics
    p.age,
    p.gender,

    -- Age buckets for categorical analysis
    CASE
        WHEN p.age < 40             THEN '18-39'
        WHEN p.age BETWEEN 40 AND 59 THEN '40-59'
        WHEN p.age BETWEEN 60 AND 74 THEN '60-74'
        ELSE '75+'
    END                             AS age_group,

    -- Chronic condition burden (count of distinct conditions)
    COALESCE(cc.chronic_condition_count, 0) AS chronic_condition_count,

    -- High-risk flag: >= 2 chronic conditions
    CASE
        WHEN COALESCE(cc.chronic_condition_count, 0) >= 2
        THEN 1 ELSE 0
    END                             AS high_chronic_burden_flag,

    -- Prior 12-month inpatient utilization (history of readmissions)
    COALESCE(hist.prior_admissions_12m, 0) AS prior_admissions_12m,

    -- Length of stay risk tier
    CASE
        WHEN c.length_of_stay <= 2  THEN 'Short (1-2 days)'
        WHEN c.length_of_stay <= 5  THEN 'Moderate (3-5 days)'
        ELSE 'Extended (6+ days)'
    END                             AS los_tier

FROM cohort_staging c

-- Join demographics: every cohort member must have a demographic record
JOIN claims_db.patient_demographics p
    ON c.patient_id = p.patient_id

-- Count distinct chronic conditions per patient
LEFT JOIN (
    SELECT
        patient_id,
        COUNT(DISTINCT condition_code) AS chronic_condition_count
    FROM claims_db.patient_conditions
    WHERE chronic_condition_flag = 1
    GROUP BY patient_id
) cc ON c.patient_id = cc.patient_id

-- Count prior inpatient admissions in the 12 months before the index period
LEFT JOIN (
    SELECT
        patient_id,
        COUNT(claim_id)                AS prior_admissions_12m
    FROM claims_db.inpatient_claims
    WHERE admission_date >= '2024-01-01'
      AND admission_date <  '2025-01-01'   -- 12 months before observation window
    GROUP BY patient_id
) hist ON c.patient_id = hist.patient_id

ORDER BY c.patient_id, c.index_discharge_date;
