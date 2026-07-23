# Healthcare Claims Cohort Analysis and Care Management Outcome Evaluation

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)
![SQL](https://img.shields.io/badge/SQL-ANSI%20%7C%20PostgreSQL-336791?style=flat-square&logo=postgresql&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-Logistic%20Regression-orange?style=flat-square)
![scipy](https://img.shields.io/badge/scipy-Chi--Square%20Test-8CAAE6?style=flat-square)
![Domain](https://img.shields.io/badge/Domain-Healthcare%20%7C%20Pharmacy%20Analytics-1A5276?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)

End-to-end analytical framework evaluating 30-day inpatient readmission drivers
and quantifying the clinical and financial impact of Care Management program
enrollment. Built on a CMS SynPUF-modeled synthetic claims dataset using SQL
cohort construction, chi-square hypothesis testing, and multivariate logistic
regression to produce actionable cost savings recommendations.

---

## Business Problem

Unplanned 30-day hospital readmissions represent one of the largest drivers of
preventable healthcare expenditure. Under the CMS Hospital Readmissions Reduction
Program (HRRP), Medicare Advantage plans and health systems face financial
penalties for excess readmissions across five target conditions: Heart Failure,
COPD, Pneumonia, Hip/Knee Replacement, and CABG.

Care Management programs provide post-discharge follow-up, medication
reconciliation, and social support for high-risk patients. However, health
plan operations leadership needs empirical evidence to answer two questions
before expanding program investment:

**Question 1:** Does Care Management enrollment statistically reduce 30-day
readmission rates when controlling for patient age, severity, and comorbidity burden?

**Question 2:** What is the net financial value of enrolling the current
unmanaged high-risk patient cohort in the Care Management program?

---

## Key Findings

| Metric | Value | Note |
|---|---|---|
| Overall 30-day readmission rate | 24.0% | Across 5,000 patient cohort |
| Readmission rate -- Not enrolled | 28.5% | Unmanaged baseline |
| Readmission rate -- Enrolled | 15.9% | Care Management group |
| Absolute Risk Reduction (ARR) | 12.6 percentage points | Clinically meaningful |
| Chi-square p-value | p < 0.001 | Reject null hypothesis |
| Care Management Odds Ratio | 0.446 (95% CI: 0.391-0.509) | 48.9% reduction in odds |
| Projected annual savings (10K members) | $14.4M | At $11,405 avg readmit cost |
| Break-even program cost | $1,437 per enrolled member | Per 1,000 patient projection |
| Heart Failure readmission rate | 30.1% | Highest-risk diagnosis |

---

## Project Output -- Charts

### Chart 1 -- Readmission Rates by Diagnosis and Age Group

Side-by-side comparison of 30-day readmission rates between Care Management
enrolled and unenrolled patients, segmented by primary diagnosis and patient
age group. Heart Failure shows the largest absolute reduction.

![Readmission Rates Overview](https://raw.githubusercontent.com/aksingh-ops/healthcare-claims-cohort-evaluation/main/reports/01_readmission_rates_overview.png)

---

### Chart 2 -- Chi-Square Test Visualization

Contingency table heatmap (row percentages) and grouped bar chart confirming
the statistically significant association between Care Management enrollment
and reduced readmission rates (Chi2=99.65, p < 0.001, Phi=0.143).

![Chi-Square Test](https://raw.githubusercontent.com/aksingh-ops/healthcare-claims-cohort-evaluation/main/reports/02_chi_square_test.png)

---

### Chart 3 -- Logistic Regression Odds Ratios

Forest plot showing adjusted odds ratios with 95% confidence intervals for all
model predictors. Care Management (OR=0.446) is the only protective factor.
Age, length of stay, and chronic condition count are all significant risk factors.

![Odds Ratios](https://raw.githubusercontent.com/aksingh-ops/healthcare-claims-cohort-evaluation/main/reports/03_odds_ratios.png)

---

### Chart 4 -- Executive Savings Dashboard

KPI summary tiles, projected savings by diagnosis, and break-even analysis
showing the program cost per member below which Care Management expansion
is financially justified.

![Executive Savings Dashboard](https://raw.githubusercontent.com/aksingh-ops/healthcare-claims-cohort-evaluation/main/reports/04_executive_savings_dashboard.png)

---

## How to Run

```bash
git clone https://github.com/aksingh-ops/healthcare-claims-cohort-evaluation
cd healthcare-claims-cohort-evaluation
pip install -r requirements.txt

# Step 1: Generate synthetic claims dataset
python src/generate_data.py

# Step 2: Run full statistical analysis and generate all charts
python src/statistical_analysis.py
```

All 4 charts save to `reports/`. Dataset saves to `data/synthetic_claims_data.csv`.

---

## Project Structure

```
healthcare-claims-cohort-evaluation/
|
|-- src/
|   |-- generate_data.py          Synthetic CMS SynPUF-modeled claims dataset
|   `-- statistical_analysis.py   Chi-square, logistic regression, savings model, charts
|
|-- sql/
|   |-- 01_cohort_selection.sql   Index admission + 30-day readmission window (CTE, LEFT JOIN)
|   `-- 02_feature_extraction.sql Demographics, chronic burden, prior utilization (multi-join)
|
|-- data/
|   `-- synthetic_claims_data.csv Generated by generate_data.py
|
|-- reports/
|   |-- 01_readmission_rates_overview.png
|   |-- 02_chi_square_test.png
|   |-- 03_odds_ratios.png
|   `-- 04_executive_savings_dashboard.png
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## Phase-by-Phase Breakdown

<table>
  <thead>
    <tr>
      <th>Phase</th>
      <th>File</th>
      <th>What it does</th>
      <th>Key output</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>1 -- Cohort Selection</strong></td>
      <td>sql/01_cohort_selection.sql</td>
      <td>Identifies index inpatient admissions in the observation window and flags 30-day readmissions using a LEFT JOIN date-range window. Groups by patient to find earliest readmission.</td>
      <td>readmitted_30d flag per patient, readmit cost</td>
    </tr>
    <tr>
      <td><strong>2 -- Feature Extraction</strong></td>
      <td>sql/02_feature_extraction.sql</td>
      <td>Joins cohort staging results with demographics, chronic condition counts (COUNT DISTINCT), and prior 12-month utilization history. Derives age groups and LOS tiers.</td>
      <td>Full analytical feature set for Python modeling</td>
    </tr>
    <tr>
      <td><strong>3 -- Data Generation</strong></td>
      <td>src/generate_data.py</td>
      <td>Generates 5,000 synthetic patient records modeled on CMS SynPUF schema. Readmission probability driven by a logistic function with clinically validated coefficients (age, LOS, chronic burden, care management).</td>
      <td>data/synthetic_claims_data.csv -- 5,000 records</td>
    </tr>
    <tr>
      <td><strong>4 -- Descriptive Statistics</strong></td>
      <td>src/statistical_analysis.py</td>
      <td>Cohort profiling: readmission rates by diagnosis, age group, and care management status. Identifies Heart Failure (30.1%) as the highest-risk diagnosis.</td>
      <td>Phase 1 console output</td>
    </tr>
    <tr>
      <td><strong>5 -- Hypothesis Testing</strong></td>
      <td>src/statistical_analysis.py</td>
      <td>Chi-square test of independence on a 2x2 contingency table (Care Management enrollment vs readmission status). Reports chi2 statistic, p-value, degrees of freedom, and Phi effect size.</td>
      <td>Chi2=99.65, p &lt; 0.001 -- Reject H0</td>
    </tr>
    <tr>
      <td><strong>6 -- Logistic Regression</strong></td>
      <td>src/statistical_analysis.py</td>
      <td>Multivariate logistic regression controlling for age, LOS, and chronic condition count. Converts log-odds to Odds Ratios with 95% CI for stakeholder communication.</td>
      <td>Care Management OR=0.446 (48.9% reduction in odds)</td>
    </tr>
    <tr>
      <td><strong>7 -- Savings Model</strong></td>
      <td>src/statistical_analysis.py</td>
      <td>Absolute risk reduction applied to projected enrollment at scale. Break-even program cost calculation. Diagnosis-level savings decomposition.</td>
      <td>$14.4M annual savings per 10,000 members</td>
    </tr>
    <tr>
      <td><strong>8 -- Charts</strong></td>
      <td>src/statistical_analysis.py</td>
      <td>4 publication-quality charts: readmission rate comparison, chi-square visualization, odds ratio forest plot, executive savings dashboard.</td>
      <td>4 PNG files in reports/</td>
    </tr>
  </tbody>
</table>

---

## Statistical Methodology

### Chi-Square Test of Independence

Tests whether readmission status and Care Management enrollment are
statistically independent at alpha=0.05:

```
H0: Readmission status is independent of Care Management enrollment
H1: Readmission status is dependent on Care Management enrollment

Result: Chi2=99.65, df=1, p < 0.001, Phi=0.143
Decision: Reject H0 -- statistically significant association confirmed
```

### Multivariate Logistic Regression

Controls for clinical confounders to isolate the Care Management effect:

```
logit(P[readmit]) = b0
                  + b1 * care_management_flag
                  + b2 * age
                  + b3 * length_of_stay
                  + b4 * chronic_condition_count
```

Odds Ratios with 95% Confidence Intervals:

| Variable | OR | 95% CI | p-value |
|---|---|---|---|
| Care Management (enrolled=1) | 0.446 | 0.391 -- 0.509 | < 0.001 |
| Age (per year) | 1.022 | 1.019 -- 1.025 | < 0.001 |
| Length of Stay (per day) | 1.160 | 1.125 -- 1.195 | < 0.001 |
| Chronic Conditions (per additional) | 1.334 | 1.277 -- 1.393 | < 0.001 |

### Absolute Risk Reduction and Savings Model

```
ARR = Readmission rate (unmanaged) - Readmission rate (managed)
    = 28.5% - 15.9% = 12.6 percentage points

Projected avoided readmissions per 1,000 newly enrolled patients:
  = 1,000 * 0.126 = 126 readmissions

Projected savings per 1,000 patients:
  = 126 * $11,405 = $1.44M

Annual savings per 10,000 members:
  = 1,261 * $11,405 = $14.4M
```

---

## Data Source

The dataset uses a synthetic schema modeled on CMS Synthetic Public Use Files
(SynPUFs) available at:
https://www.cms.gov/Research-Statistics-Data-and-Systems/Downloadable-Public-Use-Files/SynPUFs

Three source tables are simulated:
- `inpatient_claims` -- admissions, discharges, diagnoses, costs
- `patient_demographics` -- age, gender
- `patient_conditions` -- chronic condition codes and flags

The readmission probability model uses logistic regression coefficients
grounded in published CMS HRRP benchmarks and peer-reviewed readmission
risk literature.

---

## Strategic Recommendation

Prioritize Care Management outreach for patients meeting all three criteria:

1. Two or more chronic conditions
2. Index length of stay exceeding 4 days
3. Primary diagnosis of Heart Failure or COPD

This high-risk segment shows the highest absolute risk reduction and
generates the best program ROI per enrolled member. At the current
$11,405 average readmission cost, the break-even program cost is
$1,437 per enrolled member per year -- well below typical Care
Management program costs of $300 to $800 per member annually.

---

## Author

**Akash Singh**
M.S. Business Analytics -- Iowa State University
[github.com/aksingh-ops](https://github.com/aksingh-ops) | [Portfolio](https://aksingh-ops.github.io)
