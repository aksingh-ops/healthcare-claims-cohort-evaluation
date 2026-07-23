-- ============================================================
-- 01_cohort_selection.sql
-- Healthcare Claims Cohort and Care Management Outcome Evaluation
--
-- Purpose:
--   Identify index inpatient admissions within the observation
--   period and flag 30-day hospital readmissions using a
--   LEFT JOIN date-range window.
--
-- Business context:
--   CMS penalizes health plans and hospitals for excess 30-day
--   readmissions under the Hospital Readmissions Reduction
--   Program (HRRP). A readmission is defined as any inpatient
--   admission occurring 1 to 30 days after an index discharge.
--
-- Schema:
--   claims_db.inpatient_claims (claim_id, patient_id,
--     admission_date, discharge_date, primary_diagnosis,
--     care_management_flag, length_of_stay, total_claim_cost)
--
-- SQL techniques:
--   CTEs, LEFT JOIN on date range, CASE WHEN, COALESCE,
--   MIN window aggregation for earliest readmission
-- ============================================================

WITH index_admissions AS (
    -- Step 1: Pull all inpatient discharges in the observation window
    -- These become the index events we track readmissions from
    SELECT
        patient_id,
        claim_id                        AS index_claim_id,
        admission_date                  AS index_admission_date,
        discharge_date                  AS index_discharge_date,
        primary_diagnosis,
        care_management_flag,
        length_of_stay,
        total_claim_cost
    FROM claims_db.inpatient_claims
    WHERE discharge_date BETWEEN '2025-01-01' AND '2025-11-30'
      AND length_of_stay >= 1              -- exclude same-day discharges
),

subsequent_admissions AS (
    -- Step 2: Pull all inpatient admissions that could be readmissions
    -- We pull the full year to capture any readmission after Nov 30
    SELECT
        patient_id,
        claim_id                        AS readmit_claim_id,
        admission_date                  AS readmit_admission_date,
        total_claim_cost                AS readmit_cost
    FROM claims_db.inpatient_claims
    WHERE admission_date >= '2025-01-01'
)

SELECT
    idx.patient_id,
    idx.index_claim_id,
    idx.index_admission_date,
    idx.index_discharge_date,
    idx.primary_diagnosis,
    idx.care_management_flag,
    idx.length_of_stay,
    idx.total_claim_cost                AS index_cost,

    -- Flag: did this patient have any admission within 30 days of discharge?
    -- Uses MIN to find the earliest subsequent admission
    CASE
        WHEN MIN(sub.readmit_admission_date) IS NOT NULL
         AND MIN(sub.readmit_admission_date) - idx.index_discharge_date
             BETWEEN 1 AND 30
        THEN 1
        ELSE 0
    END                                 AS readmitted_30d,

    -- Cost of the first readmission (0 if no readmission)
    COALESCE(
        CASE
            WHEN MIN(sub.readmit_admission_date) - idx.index_discharge_date
                 BETWEEN 1 AND 30
            THEN MIN(sub.readmit_cost)
        END,
        0
    )                                   AS readmit_cost

FROM index_admissions idx
LEFT JOIN subsequent_admissions sub
    ON  idx.patient_id = sub.patient_id
    -- Must be after the index discharge (not the same admission)
    AND sub.readmit_admission_date > idx.index_discharge_date
    -- Must be within the 30-day readmission window
    AND sub.readmit_admission_date <= idx.index_discharge_date + INTERVAL '30 days'
GROUP BY
    idx.patient_id,
    idx.index_claim_id,
    idx.index_admission_date,
    idx.index_discharge_date,
    idx.primary_diagnosis,
    idx.care_management_flag,
    idx.length_of_stay,
    idx.total_claim_cost
ORDER BY
    idx.patient_id,
    idx.index_discharge_date;
