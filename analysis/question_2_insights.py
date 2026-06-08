"""
QUESTION 2: What insights can you draw from the data?
This script extracts and presents key insights from data analysis with supporting evidence.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, print_section_header


def draw_insights(df):
    """
    Extract and present key insights from the data analysis.
    """
    print_section_header("QUESTION 2: KEY INSIGHTS FROM DATA ANALYSIS")

    # Insight 1: Class imbalance
    print("\n1. CLASS IMBALANCE & DATA QUALITY")
    default_rate = df['SeriousDlqin2yrs'].mean() * 100
    print(f"- Dataset is highly imbalanced: {default_rate:.2f}% default rate")
    print("- Traditional accuracy metrics will be misleading")
    print(f"- Missing data: {df['MonthlyIncome'].isnull().mean()*100:.1f}% income, {df['NumberOfDependents'].isnull().mean()*100:.1f}% dependents")

    # Count data quality issues
    age_zero = (df['age'] == 0).sum()
    age_over_100 = (df['age'] > 100).sum()
    debt_extreme = (df['DebtRatio'] > 100).sum()
    util_over_100 = (df['RevolvingUtilizationOfUnsecuredLines'] > 1).sum()

    print(f"- Data quality issues detected:")
    print(f"  - {age_zero} consumer(s) with age = 0")
    print(f"  - {age_over_100} consumers with age > 100")
    print(f"  - {debt_extreme:,} consumers ({debt_extreme/len(df)*100:.1f}%) with debt ratio > 100")
    print(f"  - {util_over_100:,} consumers ({util_over_100/len(df)*100:.1f}%) with utilization > 100%")
    print("  - Implication: Need careful data cleaning and class balancing strategies")

    # Insight 2: Delinquency variables - LEAKAGE CAUTION
    print("\n2. DELINQUENCY VARIABLES ARE HIGHLY ASSOCIATED WITH THE TARGET BUT MAY CAUSE LEAKAGE")
    no_delinq = ((df['NumberOfTimes90DaysLate'] == 0) & 
                 (df['NumberOfTime60-89DaysPastDueNotWorse'] == 0) & 
                 (df['NumberOfTime30-59DaysPastDueNotWorse'] == 0))
    no_delinq_default = df[no_delinq]['SeriousDlqin2yrs'].mean() * 100
    has_delinq_default = df[~no_delinq]['SeriousDlqin2yrs'].mean() * 100
    has_90day_default = df[df['NumberOfTimes90DaysLate'] > 0]['SeriousDlqin2yrs'].mean() * 100
  
    print(f"- No recorded delinquency = {no_delinq_default:.2f}% default rate")
    print(f"- Any recorded delinquency = {has_delinq_default:.2f}% default rate")
    print(f"- 90+ days late count > 0 = {has_90day_default:.2f}% default rate")

    # Quantify association strength
    delinq_risk_ratio = has_delinq_default / no_delinq_default if no_delinq_default > 0 else 0

    print(f"  - Insight: Delinquency variables show strong association with default outcomes ({delinq_risk_ratio:.1f}x higher risk)")
    print("  - Caution: These variables are measured in the same two-year period as the target")
    print("    and may overlap with the target definition (SeriousDlqin2yrs)")
    print("  - Implication: Exclude delinquency variables from a forward-looking predictive model")
    print("    unless timing confirms they are prior-period variables")

    # Insight 3: Credit utilization
    print("\n3. CREDIT UTILIZATION SIGNALS FINANCIAL DISTRESS")
    low_util_default = df[df['RevolvingUtilizationOfUnsecuredLines'] <= 0.3]['SeriousDlqin2yrs'].mean() * 100
    high_util_default = df[df['RevolvingUtilizationOfUnsecuredLines'] >= 0.8]['SeriousDlqin2yrs'].mean() * 100

    print(f"- Low utilization (<=30%) = {low_util_default:.2f}% default rate")
    print(f"- High utilization (>=80%) = {high_util_default:.2f}% default rate")
    print(f"- Insight: High utilization shows {high_util_default/low_util_default:.1f}x higher observed default rate")
    print("  - Implication: Maxed-out credit lines indicate financial distress")

    # Insight 4: Debt burden
    print("\n4. DEBT BURDEN MATTERS SIGNIFICANTLY")
    low_debt_default = df[df['DebtRatio'] <= 0.36]['SeriousDlqin2yrs'].mean() * 100
    high_debt_default = df[df['DebtRatio'] > 0.5]['SeriousDlqin2yrs'].mean() * 100

    print(f"- Healthy debt ratio (<=36%) = {low_debt_default:.2f}% default rate")
    print(f"- High debt ratio (>50%) = {high_debt_default:.2f}% default rate")
    print(f"- Insight: High debt burden shows {high_debt_default/low_debt_default:.1f}x higher observed default rate")
    print("  - Implication: Debt-to-income ratio is a critical underwriting metric")

    # Insight 5: Age patterns
    print("\n5. AGE SHOWS DECLINING RISK PATTERN")
    young_default = df[df['age'] < 30]['SeriousDlqin2yrs'].mean() * 100
    middle_default = df[(df['age'] >= 30) & (df['age'] < 60)]['SeriousDlqin2yrs'].mean() * 100
    senior_default = df[df['age'] >= 60]['SeriousDlqin2yrs'].mean() * 100

    print(f"- Younger consumers (<30) = {young_default:.2f}% default rate")
    print(f"- Middle-aged (30-60) = {middle_default:.2f}% default rate")
    print(f"- Older consumers (60+) = {senior_default:.2f}% default rate")
    print("- Insight: Risk decreases with age - seniors have the lowest default rate")
    print("- Implication: Younger borrowers (<30) require more scrutiny, seniors are lower risk")

    # Insight 6: Income level
    print("\n6. INCOME LEVEL MODERATES RISK")
    df_income = df[df['MonthlyIncome'].notna()]
    q1_threshold = df_income['MonthlyIncome'].quantile(0.25)
    q4_threshold = df_income['MonthlyIncome'].quantile(0.75)
    low_income_default = df_income[df_income['MonthlyIncome'] <= q1_threshold]['SeriousDlqin2yrs'].mean() * 100
    high_income_default = df_income[df_income['MonthlyIncome'] >= q4_threshold]['SeriousDlqin2yrs'].mean() * 100

    # Calculate reduction using unrounded values for accuracy
    income_reduction = (1 - high_income_default/low_income_default) * 100

    print(f"- Low income (Q1) = {low_income_default:.2f}% default rate")
    print(f"- High income (Q4) = {high_income_default:.2f}% default rate")
    print(f"- Insight: Higher income reduces risk by {income_reduction:.1f}%")
    print("- Implication: Income verification is important for risk assessment")

    # Insight 7: Real estate ownership
    print("\n7. REAL ESTATE OWNERSHIP INDICATES STABILITY")
    no_re_default = df[df['NumberRealEstateLoansOrLines'] == 0]['SeriousDlqin2yrs'].mean() * 100
    has_re_default = df[df['NumberRealEstateLoansOrLines'] > 0]['SeriousDlqin2yrs'].mean() * 100

    # Calculate reduction using unrounded values for accuracy
    re_reduction = (1 - has_re_default/no_re_default) * 100

    print(f"- No real estate = {no_re_default:.2f}% default rate")
    print(f"- Has real estate = {has_re_default:.2f}% default rate")
    print(f"- Insight: Real estate ownership associated with {re_reduction:.1f}% lower risk")
    print("- Implication: Homeownership is a positive stability indicator")

    # Insight 8: Correlations
    print("\n8. LINEAR CORRELATIONS (Pearson)")
    print("   Note: These are linear correlations only. Credit risk relationships are often nonlinear.")
    print("   Binned analysis (Insights 2-7) may show different patterns than correlation.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'SeriousDlqin2yrs' in numeric_cols:
        correlations = df[numeric_cols].corr()['SeriousDlqin2yrs'].drop('SeriousDlqin2yrs').sort_values(ascending=False)

        print("\n Strongest Positive Correlations:")
        positive_corr = correlations[correlations > 0]
        for feature, corr in positive_corr.head(3).items():
            print(f"  - {feature}: {corr:+.4f}")

        print("\n Strongest Negative Correlations:")
        negative_corr = correlations[correlations < 0]
        for feature, corr in negative_corr.tail(3).items():
            print(f"  - {feature}: {corr:+.4f}")

    # Get specific correlations for commentary
    debt_corr = correlations.get('DebtRatio', 0)
    util_corr = correlations.get('RevolvingUtilizationOfUnsecuredLines', 0)

    print("\n  ⚠️ Important:")
    print("  • Most correlations are very weak (< 0.15)")
    print(f"  • DebtRatio shows NEGATIVE correlation ({debt_corr:+.4f}) despite binned analysis showing risk")
    print("    -> This is due to outliers and nonlinear relationship")
    print(f"  • RevolvingUtilization correlation is near-zero ({util_corr:+.4f}) despite strong binned signal")
    print("  -> Linear correlation misses the nonlinear risk pattern")

    # Insight 9: Combined risk factors
    print("\n9. COMBINED RISK FACTORS")

    # High risk combo 1
    high_risk_1 = df[(df['RevolvingUtilizationOfUnsecuredLines'] >= 0.8) & (df['NumberOfTimes90DaysLate'] > 0)]
    high_risk_1_default = high_risk_1['SeriousDlqin2yrs'].mean() * 100 if len(high_risk_1) > 0 else 0

    print("\n🔥 High Utilization (>=80%) + Past 90-day Delinquency:")
    print(f"- Count: {len(high_risk_1):,} consumers")
    if len(high_risk_1) > 0:
        print(f"- Default rate: {high_risk_1_default:.2f}%")
        print(f"- {high_risk_1_default/default_rate:.1f}x higher than average")

    # High risk combo 2
    low_income_threshold = df['MonthlyIncome'].quantile(0.25)
    high_risk_2 = df[(df['DebtRatio'] > 0.5) & (df['MonthlyIncome'] < low_income_threshold)]
    high_risk_2_default = high_risk_2['SeriousDlqin2yrs'].mean() * 100 if len(high_risk_2) > 0 else 0

    print("\n📉 High Debt Ratio (>50%) + Low Income (Q1):")
    print(f"- Count: {len(high_risk_2):,} consumers")
    if len(high_risk_2) > 0:
        print(f"- Default rate: {high_risk_2_default:.2f}%")
        print(f"- {high_risk_2_default/default_rate:.1f}x higher than average")

    # Low risk combo
    low_risk = df[
        (df['RevolvingUtilizationOfUnsecuredLines'] <= 0.3) &
        (df['NumberOfTimes90DaysLate'] == 0) &
        (df['NumberOfTime30-59DaysPastDueNotWorse'] == 0) &
        (df['NumberRealEstateLoansOrLines'] > 0)
    ]
    low_risk_default = low_risk['SeriousDlqin2yrs'].mean() * 100 if len(low_risk) > 0 else 0

    print("\n🛡️ Low Risk Profile (Low Utilization + No Delinquency + Has Real Estate):")
    print(f"- Count: {len(low_risk):,} consumers")
    if len(low_risk) > 0:
        print(f"- Default rate: {low_risk_default:.2f}%")
        print(f"- ({low_risk_default/default_rate:.1f}x of average rate)")

    print("\n💡 KEY TAKEAWAY: Risk compounds when multiple negative factors are present")

    return df


def visualize_insights(df):
    """
    Create all visualizations for insights analysis.
    """
    print_section_header("QUESTION 2: VISUALIZATIONS")

    plot_correlation_analysis(df)
    plot_combined_risk_heatmap(df)

    return df


def plot_correlation_analysis(df):
    """
    Visualize correlations between features and target.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Select key features for visualization
    key_features = [
        'SeriousDlqin2yrs',
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]

    # Filter to available columns
    key_features = [f for f in key_features if f in df.columns]

    # Correlation heatmap
    plt.figure(figsize=(14, 10))
    correlation_matrix = df[key_features].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # Correlation with target
    correlations = df[key_features].corr()['SeriousDlqin2yrs'].drop('SeriousDlqin2yrs').sort_values()

    plt.figure(figsize=(10, 6))
    colors = ['red' if x < 0 else 'green' for x in correlations.values]
    plt.barh(range(len(correlations)), correlations.values, color=colors, alpha=0.7)
    plt.yticks(range(len(correlations)), correlations.index, fontsize=10)
    plt.xlabel('Correlation with Default', fontsize=12)
    plt.title('Feature Correlations with Serious Delinquency', fontsize=14, fontweight='bold')
    plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()


def plot_combined_risk_heatmap(df):
    """
    Create heatmap showing default rates for combined risk factors.
    """
    df_plot = df.copy()

    # Create utilization and debt categories
    df_plot['UtilCat'] = pd.cut(df_plot['RevolvingUtilizationOfUnsecuredLines'],
                                bins=[0, 0.3, 0.8, 1, 100],
                                labels=['Low<30%', 'Med30-80%', 'High80-100%', 'VeryHigh>100%'])

    df_plot['DebtCat'] = pd.cut(df_plot['DebtRatio'],
                                bins=[0, 0.36, 0.5, 1, 10000],
                                labels=['Low<36%', 'Med36-50%', 'High50-100%', 'VeryHigh>100%'])

    # Create pivot table
    pivot_data = df_plot.groupby(['UtilCat', 'DebtCat'])['SeriousDlqin2yrs'].mean() * 100
    pivot_table = pivot_data.unstack()

    # Calculate center dynamically as the overall default rate
    center_value = df['SeriousDlqin2yrs'].mean() * 100

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='RdYlGn_r', center=center_value,
                cbar_kws={'label': 'Default Rate (%)'}, linewidths=1)
    plt.title('Default Rate by Utilization and Debt Ratio\n(Combined Risk Factors)', fontsize=14, fontweight='bold')
    plt.xlabel('Debt Ratio Category', fontsize=12)
    plt.ylabel('Utilization Category', fontsize=12)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == '__main__':
    # Load data
    df = load_data('cs-training.csv')

    # Run analysis
    df = draw_insights(df)

    # Create visualizations
    df = visualize_insights(df)

    print("\n" + "="*80)
    print("="*80)
