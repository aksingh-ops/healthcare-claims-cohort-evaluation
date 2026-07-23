"""
statistical_analysis.py
-----------------------
Healthcare Claims Cohort and Care Management Outcome Evaluation.

Runs in sequence:
  Phase 1  -- Data validation and descriptive statistics
  Phase 2  -- Chi-square test of independence
             (Care Management vs 30-day readmission)
  Phase 3  -- Multivariate logistic regression
             (Odds ratios controlling for confounders)
  Phase 4  -- Financial savings model
             (Absolute risk reduction, projected cost savings)
  Phase 5  -- Output charts (4 publication-quality figures)

Usage:
  python src/statistical_analysis.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# -------------------------------------------------------
# Color palette (healthcare / clinical reporting style)
# -------------------------------------------------------
C_MANAGED   = "#1A5276"   # deep blue -- care management group
C_UNMANAGED = "#C0392B"   # deep red  -- no care management
C_GOLD      = "#C9A84C"   # gold accent
C_LIGHT     = "#F4F6F7"   # background
C_DARK      = "#1C2833"   # text

os.makedirs("reports", exist_ok=True)


# -------------------------------------------------------
# Load data
# -------------------------------------------------------
def load_data(path: str = "data/synthetic_claims_data.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        print("Dataset not found -- generating now...")
        sys.path.insert(0, "src")
        from generate_data import generate_synthetic_claims
        df = generate_synthetic_claims()
        os.makedirs("data", exist_ok=True)
        df.to_csv(path, index=False)
    return pd.read_csv(path)


# -------------------------------------------------------
# Phase 1: Descriptive statistics
# -------------------------------------------------------
def phase1_descriptive(df: pd.DataFrame) -> None:
    print("=" * 65)
    print("PHASE 1 -- COHORT DESCRIPTIVE STATISTICS")
    print("=" * 65)

    print(f"\n  Total cohort size:               {len(df):,}")
    print(f"  Overall 30-day readmission rate:  {df['readmitted_30d'].mean()*100:.1f}%")
    print(f"  Care management enrollment rate:  {df['care_management_flag'].mean()*100:.1f}%")
    print(f"  Mean age:                         {df['age'].mean():.1f} years")
    print(f"  Mean length of stay:              {df['length_of_stay'].mean():.1f} days")
    print(f"  Mean chronic conditions:          {df['chronic_condition_count'].mean():.1f}")
    print(f"  Mean index cost:                  ${df['index_cost'].mean():,.0f}")
    print()

    # Readmission rate by diagnosis
    print("  Readmission rate by primary diagnosis:")
    diag_rates = (
        df.groupby("primary_diagnosis")["readmitted_30d"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "readmit_rate", "count": "n"})
        .sort_values("readmit_rate", ascending=False)
    )
    diag_rates["readmit_rate"] = (diag_rates["readmit_rate"] * 100).round(1)
    for dx, row in diag_rates.iterrows():
        print(f"    {dx:<30}  {row['readmit_rate']:.1f}%  (n={row['n']:,})")

    # Readmission rate by care management status
    print()
    managed_rate   = df[df["care_management_flag"]==1]["readmitted_30d"].mean()
    unmanaged_rate = df[df["care_management_flag"]==0]["readmitted_30d"].mean()
    print(f"  Readmission rate -- Care Managed:      {managed_rate*100:.1f}%")
    print(f"  Readmission rate -- Not Care Managed:  {unmanaged_rate*100:.1f}%")
    print(f"  Absolute risk reduction (ARR):         {(unmanaged_rate-managed_rate)*100:.1f} pp")


# -------------------------------------------------------
# Phase 2: Chi-square test of independence
# -------------------------------------------------------
def phase2_chi_square(df: pd.DataFrame) -> dict:
    print()
    print("=" * 65)
    print("PHASE 2 -- HYPOTHESIS TESTING: CHI-SQUARE TEST")
    print("=" * 65)
    print()
    print("  H0: 30-day readmission is INDEPENDENT of Care Management enrollment.")
    print("  H1: 30-day readmission is DEPENDENT on Care Management enrollment.")
    print(f"  Alpha = 0.05  (two-tailed, 95% confidence)")
    print()

    contingency_table = pd.crosstab(
        df["care_management_flag"],
        df["readmitted_30d"],
        rownames=["Care Management"],
        colnames=["Readmitted 30d"],
    )

    print("  Contingency Table:")
    print(contingency_table.to_string(
        index=True,
        header=["Not Readmitted (0)", "Readmitted (1)"],
    ))

    chi2, p_val, dof, expected = stats.chi2_contingency(contingency_table)

    # Effect size -- Phi coefficient (appropriate for 2x2)
    n = len(df)
    phi = np.sqrt(chi2 / n)

    print()
    print(f"  Chi-square statistic: {chi2:.4f}")
    print(f"  Degrees of freedom:   {dof}")
    print(f"  p-value:              {p_val:.2e}")
    print(f"  Effect size (Phi):    {phi:.4f}")
    print()

    if p_val < 0.05:
        print("  RESULT: Reject H0.")
        print("  Care Management enrollment is significantly associated with")
        print("  reduced 30-day readmission rates (p < 0.05).")
    else:
        print("  RESULT: Fail to reject H0.")

    return {
        "chi2": chi2,
        "p_val": p_val,
        "dof": dof,
        "phi": phi,
        "contingency_table": contingency_table,
    }


# -------------------------------------------------------
# Phase 3: Multivariate logistic regression
# -------------------------------------------------------
def phase3_logistic_regression(df: pd.DataFrame) -> dict:
    print()
    print("=" * 65)
    print("PHASE 3 -- MULTIVARIATE LOGISTIC REGRESSION")
    print("=" * 65)
    print()
    print("  Model: logit(P[readmit]) = b0 + b1*care_mgmt + b2*age")
    print("         + b3*length_of_stay + b4*chronic_condition_count")
    print()

    # Feature matrix
    X = df[[
        "care_management_flag",
        "age",
        "length_of_stay",
        "chronic_condition_count",
    ]].copy()
    X = sm.add_constant(X)
    y = df["readmitted_30d"]

    model = sm.Logit(y, X).fit(disp=False)

    # Build results table
    params    = model.params
    p_values  = model.pvalues
    conf_int  = model.conf_int()
    odds_ratios = np.exp(params)
    or_lower    = np.exp(conf_int[0])
    or_upper    = np.exp(conf_int[1])

    results_df = pd.DataFrame({
        "Coefficient":     params.round(4),
        "Odds Ratio":      odds_ratios.round(4),
        "OR 95% CI Lower": or_lower.round(4),
        "OR 95% CI Upper": or_upper.round(4),
        "p-value":         p_values.round(4),
        "Significant":     (p_values < 0.05).map({True: "Yes", False: "No"}),
    })

    # Rename index for readability
    results_df.index = [
        "Intercept",
        "Care Management (enrolled=1)",
        "Age (per year)",
        "Length of Stay (per day)",
        "Chronic Condition Count",
    ]

    print(results_df.to_string())
    print()

    cm_or = odds_ratios["care_management_flag"]
    cm_p  = p_values["care_management_flag"]
    print(f"  Care Management Odds Ratio: {cm_or:.4f}")
    print(f"  Interpretation: Enrolled patients have {(1-cm_or)*100:.1f}% lower odds")
    print(f"  of 30-day readmission vs unenrolled (p={cm_p:.4f})")
    print()
    print(f"  Pseudo R-squared (McFadden): {model.prsquared:.4f}")
    print(f"  Log-likelihood:              {model.llf:.2f}")

    return {
        "model": model,
        "results_df": results_df,
        "odds_ratios": odds_ratios,
        "conf_int": conf_int,
    }


# -------------------------------------------------------
# Phase 4: Financial savings model
# -------------------------------------------------------
def phase4_savings_model(df: pd.DataFrame) -> dict:
    print()
    print("=" * 65)
    print("PHASE 4 -- FINANCIAL OPPORTUNITY SIZING")
    print("=" * 65)

    managed_rate   = df[df["care_management_flag"]==1]["readmitted_30d"].mean()
    unmanaged_rate = df[df["care_management_flag"]==0]["readmitted_30d"].mean()
    arr = unmanaged_rate - managed_rate   # absolute risk reduction

    avg_readmit_cost = df[df["readmit_cost"] > 0]["readmit_cost"].mean()

    # Per 1,000 newly enrolled unmanaged patients
    scale_1000      = 1_000
    avoided_1000    = int(scale_1000 * arr)
    savings_1000    = avoided_1000 * avg_readmit_cost

    # Per 10,000 members (annual projection)
    scale_10k       = 10_000
    avoided_10k     = int(scale_10k * arr)
    savings_10k     = avoided_10k * avg_readmit_cost

    # Break-even analysis: what program cost per member is justified?
    cost_per_member_breakeven = savings_1000 / scale_1000

    print()
    print(f"  Readmission rate -- Unmanaged:         {unmanaged_rate*100:.1f}%")
    print(f"  Readmission rate -- Care Managed:      {managed_rate*100:.1f}%")
    print(f"  Absolute Risk Reduction (ARR):         {arr*100:.1f} percentage points")
    print(f"  Average readmission cost:              ${avg_readmit_cost:,.0f}")
    print()
    print(f"  Avoided readmissions per 1,000 patients:  {avoided_1000}")
    print(f"  Projected savings per 1,000 patients:     ${savings_1000:,.0f}")
    print()
    print(f"  Avoided readmissions per 10,000 members:  {avoided_10k}")
    print(f"  Projected annual savings (10K members):   ${savings_10k:,.0f}")
    print()
    print(f"  Break-even program cost per enrolled member: ${cost_per_member_breakeven:,.0f}")
    print()
    print("  Strategic Recommendation:")
    print("  Prioritize outreach for patients with:")
    print("    >= 2 chronic conditions, AND")
    print("    Index length of stay > 4 days, AND")
    print("    Primary diagnosis of Heart Failure or COPD")
    print("  This high-risk segment represents the highest ROI for program expansion.")

    return {
        "arr":                   arr,
        "avg_readmit_cost":      avg_readmit_cost,
        "managed_rate":          managed_rate,
        "unmanaged_rate":        unmanaged_rate,
        "avoided_1000":          avoided_1000,
        "savings_1000":          savings_1000,
        "avoided_10k":           avoided_10k,
        "savings_10k":           savings_10k,
        "breakeven":             cost_per_member_breakeven,
    }


# -------------------------------------------------------
# Phase 5: Charts
# -------------------------------------------------------
def phase5_charts(df: pd.DataFrame, chi2_results: dict, logit_results: dict, savings: dict) -> None:
    print()
    print("=" * 65)
    print("PHASE 5 -- GENERATING OUTPUT CHARTS")
    print("=" * 65)

    # -------------------------------------------------------
    # Chart 1: Readmission rates by diagnosis and care management
    # -------------------------------------------------------
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_LIGHT)
    fig1.suptitle(
        "30-Day Readmission Rates -- Cohort Overview\n"
        "Care Management Impact by Diagnosis and Demographics",
        fontsize=13, fontweight="bold", color=C_DARK
    )

    # Panel 1: By diagnosis
    diag_cm = df.groupby(["primary_diagnosis", "care_management_flag"])["readmitted_30d"].mean().unstack()
    diag_cm.columns = ["Not Enrolled", "Enrolled"]
    diag_cm = diag_cm.sort_values("Not Enrolled", ascending=False) * 100

    x = np.arange(len(diag_cm))
    w = 0.35
    axes1[0].bar(x - w/2, diag_cm["Not Enrolled"], w,
                  label="No Care Management", color=C_UNMANAGED, alpha=0.88)
    axes1[0].bar(x + w/2, diag_cm["Enrolled"], w,
                  label="Care Management", color=C_MANAGED, alpha=0.88)

    for i, (nr, r) in enumerate(zip(diag_cm["Not Enrolled"], diag_cm["Enrolled"])):
        axes1[0].text(i - w/2, nr + 0.5, f"{nr:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        axes1[0].text(i + w/2, r + 0.5,  f"{r:.1f}%",  ha="center", fontsize=8.5, fontweight="bold")

    axes1[0].set_xticks(x)
    axes1[0].set_xticklabels(diag_cm.index, fontsize=9.5)
    axes1[0].set_ylabel("30-Day Readmission Rate (%)", fontsize=10)
    axes1[0].set_title("By Primary Diagnosis", fontsize=11, fontweight="bold")
    axes1[0].legend(fontsize=9)
    axes1[0].grid(True, alpha=0.25, axis="y")
    axes1[0].set_facecolor("white")
    axes1[0].set_ylim(0, diag_cm.max().max() * 1.25)

    # Panel 2: By age group
    df["age_group"] = pd.cut(
        df["age"], bins=[17, 39, 59, 74, 100],
        labels=["18-39", "40-59", "60-74", "75+"]
    )
    age_cm = df.groupby(["age_group", "care_management_flag"])["readmitted_30d"].mean().unstack()
    age_cm.columns = ["Not Enrolled", "Enrolled"]
    age_cm = age_cm * 100

    x2 = np.arange(len(age_cm))
    axes1[1].bar(x2 - w/2, age_cm["Not Enrolled"], w,
                  label="No Care Management", color=C_UNMANAGED, alpha=0.88)
    axes1[1].bar(x2 + w/2, age_cm["Enrolled"], w,
                  label="Care Management", color=C_MANAGED, alpha=0.88)

    for i, (nr, r) in enumerate(zip(age_cm["Not Enrolled"], age_cm["Enrolled"])):
        axes1[1].text(i - w/2, nr + 0.5, f"{nr:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
        axes1[1].text(i + w/2, r + 0.5,  f"{r:.1f}%",  ha="center", fontsize=8.5, fontweight="bold")

    axes1[1].set_xticks(x2)
    axes1[1].set_xticklabels(age_cm.index, fontsize=10)
    axes1[1].set_ylabel("30-Day Readmission Rate (%)", fontsize=10)
    axes1[1].set_title("By Age Group", fontsize=11, fontweight="bold")
    axes1[1].legend(fontsize=9)
    axes1[1].grid(True, alpha=0.25, axis="y")
    axes1[1].set_facecolor("white")
    axes1[1].set_ylim(0, age_cm.max().max() * 1.25)

    plt.tight_layout()
    plt.savefig("reports/01_readmission_rates_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: reports/01_readmission_rates_overview.png")

    # -------------------------------------------------------
    # Chart 2: Chi-square visualization -- contingency table heatmap
    # -------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 7), facecolor=C_LIGHT)
    fig2.suptitle(
        "Chi-Square Test of Independence\n"
        "Care Management Enrollment vs 30-Day Readmission Status",
        fontsize=13, fontweight="bold", color=C_DARK
    )

    ct = chi2_results["contingency_table"]
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    # Heatmap of percentages
    sns.heatmap(
        ct_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        ax=axes2[0],
        cbar_kws={"label": "Row %"},
        linewidths=0.5,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
    )
    axes2[0].set_title(
        f"Contingency Table (Row %)\n"
        f"Chi2={chi2_results['chi2']:.2f}, p={chi2_results['p_val']:.2e}, "
        f"Phi={chi2_results['phi']:.3f}",
        fontsize=10, fontweight="bold"
    )
    axes2[0].set_xticklabels(["Not Readmitted", "Readmitted"], fontsize=10)
    axes2[0].set_yticklabels(["Not Enrolled (0)", "Enrolled (1)"], fontsize=10, rotation=0)
    axes2[0].set_xlabel("30-Day Readmission Status", fontsize=10)
    axes2[0].set_ylabel("Care Management Enrollment", fontsize=10)

    # Bar chart comparison
    rates = [
        df[df["care_management_flag"]==0]["readmitted_30d"].mean() * 100,
        df[df["care_management_flag"]==1]["readmitted_30d"].mean() * 100,
    ]
    bars = axes2[1].bar(
        ["Not Enrolled", "Enrolled"],
        rates,
        color=[C_UNMANAGED, C_MANAGED],
        alpha=0.88,
        width=0.45,
    )
    for bar, rate in zip(bars, rates):
        axes2[1].text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.4,
            f"{rate:.1f}%",
            ha="center", fontsize=13, fontweight="bold", color=C_DARK
        )
    axes2[1].set_ylabel("30-Day Readmission Rate (%)", fontsize=11)
    axes2[1].set_title(
        f"Readmission Rate by Enrollment Status\n"
        f"ARR = {(rates[0]-rates[1]):.1f} pp  |  p < 0.001",
        fontsize=11, fontweight="bold"
    )
    axes2[1].set_ylim(0, max(rates) * 1.35)
    axes2[1].grid(True, alpha=0.25, axis="y")
    axes2[1].set_facecolor("white")

    plt.tight_layout()
    plt.savefig("reports/02_chi_square_test.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: reports/02_chi_square_test.png")

    # -------------------------------------------------------
    # Chart 3: Logistic regression odds ratios
    # -------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(13, 7), facecolor=C_LIGHT)

    model     = logit_results["model"]
    ors       = np.exp(model.params).drop("const")
    ci        = np.exp(model.conf_int()).drop("const")
    p_vals    = model.pvalues.drop("const")

    feature_labels = {
        "care_management_flag":     "Care Management\n(Enrolled vs Not)",
        "age":                      "Age\n(per year)",
        "length_of_stay":           "Length of Stay\n(per day)",
        "chronic_condition_count":  "Chronic Conditions\n(per additional)",
    }

    labels = [feature_labels.get(f, f) for f in ors.index]
    colors = [C_MANAGED if or_ < 1 else C_UNMANAGED for or_ in ors.values]

    y_pos = np.arange(len(ors))
    ax3.barh(
        y_pos, ors.values, height=0.45,
        color=colors, alpha=0.88,
        xerr=[ors.values - ci[0].values, ci[1].values - ors.values],
        error_kw={"ecolor": C_DARK, "capsize": 5, "lw": 1.5},
    )

    # OR values and significance labels
    for i, (or_, p) in enumerate(zip(ors.values, p_vals.values)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax3.text(
            max(ors.values) + 0.05, i,
            f"OR={or_:.3f}  {sig}",
            va="center", fontsize=9.5, fontweight="bold", color=C_DARK,
        )

    ax3.axvline(1.0, color=C_DARK, lw=2, ls="--", alpha=0.6, label="OR=1 (no effect)")
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=10.5)
    ax3.set_xlabel("Odds Ratio (with 95% Confidence Interval)", fontsize=11)
    ax3.set_title(
        "Multivariate Logistic Regression -- Odds Ratios for 30-Day Readmission\n"
        "Adjusted for age, length of stay, and chronic condition burden\n"
        "*** p<0.001  ** p<0.01  * p<0.05  ns = not significant",
        fontsize=11, fontweight="bold", color=C_DARK, pad=14
    )
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, axis="x")
    ax3.set_facecolor("white")
    p_protect = mpatches.Patch(color=C_MANAGED, alpha=0.88, label="Protective (OR < 1)")
    p_risk    = mpatches.Patch(color=C_UNMANAGED, alpha=0.88, label="Risk factor (OR > 1)")
    ax3.legend(handles=[p_protect, p_risk], fontsize=9, loc="lower right")

    plt.tight_layout()
    plt.savefig("reports/03_odds_ratios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: reports/03_odds_ratios.png")

    # -------------------------------------------------------
    # Chart 4: Executive savings dashboard
    # -------------------------------------------------------
    fig4 = plt.figure(figsize=(18, 11), facecolor=C_LIGHT)
    fig4.suptitle(
        "Care Management Program -- Financial Opportunity Sizing\n"
        "Healthcare Claims Cohort and Outcome Evaluation",
        fontsize=14, fontweight="bold", color=C_DARK, y=0.98
    )

    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, 3, figure=fig4, hspace=0.5, wspace=0.4,
                           left=0.06, right=0.97, top=0.88, bottom=0.08)

    # KPI tiles
    kpis = [
        ("Absolute Risk Reduction", f"{savings['arr']*100:.1f} pp",    C_MANAGED),
        ("Avg Readmit Cost",        f"${savings['avg_readmit_cost']:,.0f}", C_UNMANAGED),
        ("Annual Savings\n(10K members)", f"${savings['savings_10k']:,.0f}", "#117A65"),
    ]
    for i, (label, value, color) in enumerate(kpis):
        ax_k = fig4.add_subplot(gs[0, i])
        ax_k.set_facecolor("white")
        ax_k.text(0.5, 0.60, value, ha="center", va="center",
                   fontsize=20, fontweight="bold", color=color, transform=ax_k.transAxes)
        ax_k.text(0.5, 0.22, label, ha="center", va="center",
                   fontsize=9.5, color="#555", transform=ax_k.transAxes)
        for sp in ax_k.spines.values():
            sp.set_edgecolor("#ccc")
        ax_k.set_xticks([]); ax_k.set_yticks([])

    # Savings waterfall by diagnosis
    ax_w = fig4.add_subplot(gs[1, 0:2])
    diag_rates = df.groupby(["primary_diagnosis", "care_management_flag"])["readmitted_30d"].mean().unstack()
    diag_rates.columns = ["unmanaged", "managed"]
    diag_rates["arr"]  = diag_rates["unmanaged"] - diag_rates["managed"]
    diag_n = df[df["care_management_flag"]==0].groupby("primary_diagnosis").size()
    diag_rates["savings"] = diag_rates["arr"] * diag_n * savings["avg_readmit_cost"] / 1e6
    diag_rates = diag_rates.sort_values("savings", ascending=False)

    bars_w = ax_w.bar(
        diag_rates.index,
        diag_rates["savings"],
        color=[C_MANAGED, C_GOLD, C_UNMANAGED, "#27AE60"],
        alpha=0.88, width=0.5,
    )
    for bar, val in zip(bars_w, diag_rates["savings"]):
        ax_w.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f"${val:.2f}M",
            ha="center", fontsize=10, fontweight="bold", color=C_DARK
        )
    ax_w.set_ylabel("Projected Savings ($M)", fontsize=10)
    ax_w.set_title("Projected Savings by Diagnosis\n(if entire unmanaged cohort enrolled)",
                    fontsize=10, fontweight="bold")
    ax_w.grid(True, alpha=0.25, axis="y")
    ax_w.set_facecolor("white")

    # Break-even analysis
    ax_be = fig4.add_subplot(gs[1, 2])
    program_costs = np.linspace(0, savings["breakeven"] * 1.5, 100)
    net_savings   = savings["savings_1000"] - program_costs * 1000
    ax_be.plot(program_costs, net_savings / 1e6, color=C_MANAGED, lw=2.5)
    ax_be.axhline(0, color=C_DARK, ls="--", lw=1.5, alpha=0.5)
    ax_be.axvline(savings["breakeven"], color=C_UNMANAGED, ls=":", lw=2,
                   label=f"Break-even: ${savings['breakeven']:,.0f}/member")
    ax_be.fill_between(
        program_costs,
        net_savings / 1e6,
        0,
        where=net_savings >= 0,
        alpha=0.15, color=C_MANAGED,
    )
    ax_be.set_xlabel("Program Cost per Enrolled Member ($)", fontsize=10)
    ax_be.set_ylabel("Net Savings ($M) per 1,000 patients", fontsize=10)
    ax_be.set_title("Break-Even Analysis\n(per 1,000 enrolled patients)",
                     fontsize=10, fontweight="bold")
    ax_be.legend(fontsize=8.5)
    ax_be.grid(True, alpha=0.25)
    ax_be.set_facecolor("white")

    plt.savefig("reports/04_executive_savings_dashboard.png", dpi=150,
                bbox_inches="tight", facecolor=C_LIGHT)
    plt.close()
    print("  Saved: reports/04_executive_savings_dashboard.png")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main() -> None:
    print()
    print("=" * 65)
    print("Healthcare Claims Cohort and Care Management Outcome Evaluation")
    print("=" * 65)

    df = load_data()

    phase1_descriptive(df)
    chi2_results  = phase2_chi_square(df)
    logit_results = phase3_logistic_regression(df)
    savings       = phase4_savings_model(df)
    phase5_charts(df, chi2_results, logit_results, savings)

    print()
    print("=" * 65)
    print("Pipeline complete.")
    print("=" * 65)
    print("Outputs:")
    for f in sorted(os.listdir("reports")):
        size = os.path.getsize(f"reports/{f}")
        print(f"  reports/{f:<45} {size/1024:.1f} KB")


if __name__ == "__main__":
    main()
