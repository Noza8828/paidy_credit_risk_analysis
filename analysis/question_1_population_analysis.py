"""
QUESTION 1: What can you tell us about the population of consumers?
This script provides demographic and behavioral analysis of the consumer population.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, print_section_header


def analyze_consumer_population(df):
    """
    Analysis of the consumer population demographics and characteristics.
    """
    print_section_header("QUESTION 1: CONSUMER POPULATION ANALYSIS")
    
    # Overall statistics
    print("\n1. DATASET OVERVIEW")
    print(f"Total consumers: {len(df):,}")
    print(f"Features available: {df.shape[1]}")
    print(f"Default rate: {df['SeriousDlqin2yrs'].mean()*100:.2f}% ({df['SeriousDlqin2yrs'].sum():,} defaults)")
    
    # Demographic profile
    print("\n2. DEMOGRAPHIC PROFILE")
    print("\nAge Statistics:")
    print(f" - Mean age: {df['age'].mean():.1f} years")
    print(f" - Median age: {df['age'].median():.0f} years")
    print(f" - Age range: {df['age'].min()} - {df['age'].max()} years")
    print(f" - Standard deviation: {df['age'].std():.1f} years")
    
    # Age distribution buckets
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df['AgeGroup'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)
    age_dist = df['AgeGroup'].value_counts().sort_index()
    
    print("\nAge Distribution:")
    for age_group, count in age_dist.items():
        pct = (count / len(df)) * 100
        print(f" - {age_group}: {count:,} ({pct:.1f}%)")

    # Household composition
    print("\nHousehold Composition:")
    print(f"  - Mean dependents: {df['NumberOfDependents'].mean():.2f}")
    print(f"  - Median dependents: {df['NumberOfDependents'].median():.0f}")
    print(f"  - % with no dependents: {((df['NumberOfDependents'] == 0).mean()*100):.1f}%")
    print(f"  - % with 1-2 dependents: {(((df['NumberOfDependents'] >= 1) & (df['NumberOfDependents'] <= 2)).mean()*100):.1f}%")
    print(f"  - % with 3+ dependents: {((df['NumberOfDependents'] >= 3).mean()*100):.1f}%")

    # Income profile
    print("\n3. INCOME PROFILE")
    income_stats = df['MonthlyIncome'].describe()
    print(f"  - Missing income data: {df['MonthlyIncome'].isnull().sum():,} ({df['MonthlyIncome'].isnull().mean()*100:.1f}%)")
    print(f"  - Mean monthly income: ${income_stats['mean']:,.0f}")
    print(f"  - Median monthly income: ${income_stats['50%']:,.0f}")
    print(f"  - 25th percentile: ${income_stats['25%']:,.0f}")
    print(f"  - 75th percentile: ${income_stats['75%']:,.0f}")

    # Income quartiles
    df['IncomeQuartile'] = pd.qcut(df['MonthlyIncome'], q=4, labels=['Q1-Low', 'Q2-Medium', 'Q3-High', 'Q4-VeryHigh'], duplicates='drop')
    income_dist = df['IncomeQuartile'].value_counts().sort_index()
    print("\nIncome Distribution (Quartiles):")
    for quartile, count in income_dist.items():
        pct = (count / income_dist.sum()) * 100
        print(f"  - {quartile}: {count:,} ({pct:.1f}%)")

    # Credit behavior
    print("\n4. CREDIT BEHAVIOR")
    print(f"  - Mean credit lines/loans: {df['NumberOfOpenCreditLinesAndLoans'].mean():.1f}")
    print(f"  - Median credit lines/loans: {df['NumberOfOpenCreditLinesAndLoans'].median():.0f}")
    print(f"  - Mean real estate loans: {df['NumberRealEstateLoansOrLines'].mean():.2f}")
    print(f"  - % with no real estate loans: {((df['NumberRealEstateLoansOrLines'] == 0).mean()*100):.1f}%")
    print(f"  - % with mortgage: {((df['NumberRealEstateLoansOrLines'] >= 1).mean()*100):.1f}%")

    # Revolving credit utilization
    util_stats = df['RevolvingUtilizationOfUnsecuredLines'].describe()
    print("\nRevolving Credit Utilization:")
    print(f"  - Mean utilization: {util_stats['mean']*100:.1f}%")
    print(f"  - Median utilization: {util_stats['50%']*100:.1f}%")
    print(f"  - % with <30% utilization (healthy): {((df['RevolvingUtilizationOfUnsecuredLines'] < 0.3).mean()*100):.1f}%")
    print(f"  - % with 30-80% utilization (moderate): {(((df['RevolvingUtilizationOfUnsecuredLines'] >= 0.3) & (df['RevolvingUtilizationOfUnsecuredLines'] <= 0.8)).mean()*100):.1f}%")
    print(f"  - % with >80% utilization (high risk): {((df['RevolvingUtilizationOfUnsecuredLines'] > 0.8).mean()*100):.1f}%")
    
    # Debt ratio
    print("\nDebt-to-Income Ratio:")
    print(f"  - Mean debt ratio: {df['DebtRatio'].mean():.2f}")
    print(f"  - Median debt ratio: {df['DebtRatio'].median():.2f}")
    print(f"  - % with healthy debt ratio (<0.36): {((df['DebtRatio'] < 0.36).mean()*100):.1f}%")
    print(f"  - % with moderate debt ratio (0.36-0.50): {(((df['DebtRatio'] >= 0.36) & (df['DebtRatio'] <= 0.50)).mean()*100):.1f}%")
    print(f"  - % with high debt ratio (>=0.50): {((df['DebtRatio'] >= 0.50).mean()*100):.1f}%")

    # Delinquency history
    print("\n5. DELINQUENCY HISTORY")
    print(f"  - % with any past delinquency: {(((df['NumberOfTime30-59DaysPastDueNotWorse'] > 0) | (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0) | (df['NumberOfTimes90DaysLate'] > 0)).mean()*100):.1f}%")
    print(f"  - % with 30-59 days late: {((df['NumberOfTime30-59DaysPastDueNotWorse'] > 0).mean()*100):.1f}%")
    print(f"  - % with 60-89 days late: {((df['NumberOfTime60-89DaysPastDueNotWorse'] > 0).mean()*100):.1f}%")
    print(f"  - % with 90+ days late: {((df['NumberOfTimes90DaysLate'] > 0).mean()*100):.1f}%")

    return df

def visualize_consumer_population(df):
    """
    Visualizations for consumer population analysis.
    """
    print_section_header("QUESTION 1: VISUALIZATIONS")
    
    plot_population_overview(df)
    plot_default_by_segments(df)
    
    return df

def plot_population_overview(df):
    """
    Create visualizations for consumer population.
    """
    # Figure 1: Demographics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Age distribution
    axes[0, 0].hist(df['age'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f"Mean: {df['age'].mean():.1f}")
    axes[0, 0].axvline(df['age'].median(), color='green', linestyle='-', linewidth=2, label=f"Median: {df['age'].median():.1f}")
    axes[0, 0].set_title('Age Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Age (years)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Income distribution (log scale)
    income_clean = df['MonthlyIncome'].dropna()
    axes[0, 1].hist(np.log10(income_clean + 1), bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Monthly Income Distribution (log scale)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Log10(Monthly Income)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].grid(alpha=0.3)
    
    # Credit utilization
    util_clean = df['RevolvingUtilizationOfUnsecuredLines'].clip(upper=1.5)
    axes[1, 0].hist(util_clean, bins=50, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(0.3, color='green', linestyle='--', linewidth=2, label='30% (Healthy)')
    axes[1, 0].axvline(0.8, color='red', linestyle='--', linewidth=2, label='80% (High Risk)')
    axes[1, 0].set_title('Revolving Credit Utilization', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Utilization Rate', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Debt ratio
    debt_clean = df['DebtRatio'].clip(upper=2)
    axes[1, 1].hist(debt_clean, bins=50, color='mediumpurple', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0.36, color='orange', linestyle='--', linewidth=2, label='36% (Traditional threshold)')
    axes[1, 1].set_title('Debt-to-Income Ratio', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Debt Ratio', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_default_by_segments(df):
    """
    Plot default rates across different population segments.
    """
    # Figure 2: Default rate by segments
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Default by age group
    age_default = df.groupby('AgeGroup')['SeriousDlqin2yrs'].agg(['mean', 'count'])
    age_default['mean'] *= 100
    axes[0, 0].bar(range(len(age_default)), age_default['mean'], color='steelblue', alpha=0.7)
    axes[0, 0].set_xticks(range(len(age_default)))
    axes[0, 0].set_xticklabels(age_default.index, rotation=45)
    axes[0, 0].set_title('Default Rate by Age Group', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Default Rate (%)', fontsize=10)
    axes[0, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(age_default['mean']):
        axes[0, 0].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

    # Default by income quartile
    income_default = df.groupby('IncomeQuartile')['SeriousDlqin2yrs'].mean() * 100
    axes[0, 1].bar(range(len(income_default)), income_default.values, color='lightgreen', alpha=0.7)
    axes[0, 1].set_xticks(range(len(income_default)))
    axes[0, 1].set_xticklabels(income_default.index, rotation=45)
    axes[0, 1].set_title('Default Rate by Income Quartile', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Default Rate (%)', fontsize=10)
    axes[0, 1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(income_default.values):
        axes[0, 1].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

    # Default by utilization buckets
    df['UtilBucket'] = pd.cut(df['RevolvingUtilizationOfUnsecuredLines'],
                              bins=[0, 0.3, 0.8, 0.9, 100],
                              labels=['<30%', '30-80%', '80-90%', '>90%'])
    util_default = df.groupby('UtilBucket')['SeriousDlqin2yrs'].mean() * 100
    axes[0, 2].bar(range(len(util_default)), util_default.values, color='coral', alpha=0.7)
    axes[0, 2].set_xticks(range(len(util_default)))
    axes[0, 2].set_xticklabels(util_default.index, rotation=45)
    axes[0, 2].set_title('Default Rate by Credit Utilization', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Default Rate (%)', fontsize=10)
    axes[0, 2].grid(alpha=0.3, axis='y')
    for i, v in enumerate(util_default.values):
        axes[0, 2].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

    # Default by debt ratio buckets
    df['DebtBucket'] = pd.cut(df['DebtRatio'],
                              bins=[0, 0.36, 0.5, 1.0, 10000],
                              labels=['<36%', '36-50%', '50-100%', '>100%'])
    debt_default = df.groupby('DebtBucket')['SeriousDlqin2yrs'].mean() * 100
    axes[1, 0].bar(range(len(debt_default)), debt_default.values, color='mediumpurple', alpha=0.7)
    axes[1, 0].set_xticks(range(len(debt_default)))
    axes[1, 0].set_xticklabels(debt_default.index, rotation=45)
    axes[1, 0].set_title('Default Rate by Debt Ratio', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Default Rate (%)', fontsize=10)
    axes[1, 0].grid(alpha=0.3, axis='y')
    for i, v in enumerate(debt_default.values):
        axes[1, 0].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

    # Default by delinquency history
    df['HasDelinquency'] = ((df['NumberOfTime30-59DaysPastDueNotWorse'] > 0) |
                            (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0) |
                            (df['NumberOfTimes90DaysLate'] > 0)).map({True: 'Yes', False: 'No'})
    delinq_default = df.groupby('HasDelinquency')['SeriousDlqin2yrs'].mean() * 100
    axes[1, 1].bar(range(len(delinq_default)), delinq_default.values, color='salmon', alpha=0.7)
    axes[1, 1].set_xticks(range(len(delinq_default)))
    axes[1, 1].set_xticklabels(delinq_default.index)
    axes[1, 1].set_title('Default Rate by Delinquency History', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Default Rate (%)', fontsize=10)
    axes[1, 1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(delinq_default.values):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)

    # Default by real estate ownership
    df['HasRealEstate'] = (df['NumberRealEstateLoansOrLines'] > 0).map({True: 'Yes', False: 'No'})
    re_default = df.groupby('HasRealEstate')['SeriousDlqin2yrs'].mean() * 100
    axes[1, 2].bar(range(len(re_default)), re_default.values, color='khaki', alpha=0.7)
    axes[1, 2].set_xticks(range(len(re_default)))
    axes[1, 2].set_xticklabels(re_default.index)
    axes[1, 2].set_title('Default Rate by Real Estate Ownership', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Default Rate (%)', fontsize=10)
    axes[1, 2].grid(alpha=0.3, axis='y')
    for i, v in enumerate(re_default.values):
        axes[1, 2].text(i, v + 0.2, f'{v:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == '__main__':
    # Load data
    df = load_data('cs-training.csv')
    
    # Run data analysis
    df = analyze_consumer_population(df)
    
    print("\n" + "="*80)
    print("="*80)
