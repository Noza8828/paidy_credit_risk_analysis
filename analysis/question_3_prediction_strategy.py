"""
QUESTION 3: How can we use this data to predict that a consumer might not pay?

This script outlines comprehensive strategies for predicting consumer non-payment,
including rule-based scoring and machine learning approaches.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils import load_data, print_section_header
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score
)
from catboost import CatBoostClassifier
import sys
import warnings
warnings.filterwarnings('ignore')

def calculate_prediction_metrics(df):
    """
    Calculate prediction metrics and risk scores.
    """
    # Calculate risk scores
    df_score = calculate_risk_scores(df)

    # Calculate rule-based performance metrics
    rule_based_metrics = {}

    # Overall metrics
    rule_based_metrics['overall_default_rate'] = df_score['SeriousDlqin2yrs'].mean() * 100

    # Performance by risk category
    for category in ['Low Risk', 'Medium Risk', 'High Risk']:
        category_data = df_score[df_score['RiskCategory'] == category]
        if len(category_data) > 0:
            rule_based_metrics[f'{category.lower().replace(" ", "_")}_count'] = len(category_data)
            rule_based_metrics[f'{category.lower().replace(" ", "_")}_pct'] = len(category_data) / len(df_score) * 100
            rule_based_metrics[f'{category.lower().replace(" ", "_")}_defaults'] = category_data['SeriousDlqin2yrs'].sum()
            rule_based_metrics[f'{category.lower().replace(" ", "_")}_default_rate'] = category_data['SeriousDlqin2yrs'].mean() * 100
            rule_based_metrics[f'{category.lower().replace(" ", "_")}_avg_score'] = category_data['RiskScore'].mean()

    # Risk separation
    low_risk_default = rule_based_metrics.get('low_risk_default_rate', 0)
    high_risk_default = rule_based_metrics.get('high_risk_default_rate', 0)
    if low_risk_default > 0:
        rule_based_metrics['risk_separation'] = high_risk_default / low_risk_default

    # Calculate segmentation metrics
    segmentation_metrics = calculate_segmentation_metrics(df_score)

    return {
        'df_scored': df_score,
        'rule_based': rule_based_metrics,
        'segmentation': segmentation_metrics
    }


def calculate_risk_scores(df):
    """
    Calculate simple rule-based risk scores.

    IMPORTANT LIMITATIONS:
    1. Includes delinquency features that may cause target leakage (see Q2 analysis)
    2. This is for RETROSPECTIVE analysis only, not forward-looking predictions
    3. It's better to rebuild without delinquency variables or use confirmed prior-period data
    """

    df_score = df.copy()
    df_score['RiskScore'] = 0

    # Delinquency points
    # These are included for retrospective analysis to show maximum predictive potential
    df_score.loc[df_score['NumberOfTimes90DaysLate'] > 0, 'RiskScore'] += 40
    df_score.loc[df_score['NumberOfTime60-89DaysPastDueNotWorse'] > 0, 'RiskScore'] += 30
    df_score.loc[df_score['NumberOfTime30-59DaysPastDueNotWorse'] > 0, 'RiskScore'] += 20

    # Utilization points (aligned with Q2: >=80% = 21.08% default vs <30% = 2.22% default)
    # Note: Q2 heatmap uses different bins (0.3, 0.6, 1.0) for visualization
    df_score.loc[df_score['RevolvingUtilizationOfUnsecuredLines'] >= 0.8, 'RiskScore'] += 25
    df_score.loc[(df_score['RevolvingUtilizationOfUnsecuredLines'] >= 0.6) & 
                 (df_score['RevolvingUtilizationOfUnsecuredLines'] < 0.8), 'RiskScore'] += 15

    # Debt ratio points (aligned with Q2: >50% = 7.76% default vs <=36% = 5.83% default)
    df_score.loc[df_score['DebtRatio'] > 0.5, 'RiskScore'] += 20
    df_score.loc[(df_score['DebtRatio'] > 0.36) & (df_score['DebtRatio'] <= 0.5), 'RiskScore'] += 10

    # Age points (aligned with Q2: younger <30 have 11.73% default, seniors 60+ have 3.10%)
    # Only penalize very young (<25) who show elevated risk
    df_score.loc[df_score['age'] < 25, 'RiskScore'] += 10

    # Income points (using absolute thresholds; Q2 uses quartiles)
    df_score.loc[df_score['MonthlyIncome'].notna() & (df_score['MonthlyIncome'] < 2500), 'RiskScore'] += 10

    # No real estate
    df_score.loc[df_score['NumberRealEstateLoansOrLines'] == 0, 'RiskScore'] += 5

    # Protective factors (subtract points)
    no_delinq = ((df_score['NumberOfTimes90DaysLate'] == 0) & 
                 (df_score['NumberOfTime60-89DaysPastDueNotWorse'] == 0) & 
                 (df_score['NumberOfTime30-59DaysPastDueNotWorse'] == 0))
    df_score.loc[no_delinq, 'RiskScore'] -= 20

    df_score.loc[df_score['NumberRealEstateLoansOrLines'] > 0, 'RiskScore'] -= 15
    df_score.loc[df_score['MonthlyIncome'].notna() & (df_score['MonthlyIncome'] > 7500), 'RiskScore'] -= 10
    df_score.loc[df_score['RevolvingUtilizationOfUnsecuredLines'] < 0.3, 'RiskScore'] -= 10
    df_score.loc[df_score['DebtRatio'] < 0.36, 'RiskScore'] -= 10

    # Assign risk categories
    df_score['RiskCategory'] = pd.cut(
        df_score['RiskScore'],
        bins=[-np.inf, 25, 50, np.inf],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )

    return df_score


def calculate_segmentation_metrics(df):
    """
    Calculate segmentation metrics for different customer segments.
    """
  
    # Define segments
    df_seg = df.copy()
    df_seg['Segment'] = 'Standard'

    # Delinquency History segment (CAUTION: May cause target leakage!)
    df_seg.loc[(df_seg['NumberOfTime60-89DaysPastDueNotWorse'] > 0) | 
               (df_seg['NumberOfTimes90DaysLate'] > 0), 'Segment'] = 'Delinquency History'

    # Age-based segments (aligned with Q2 analysis)
    df_seg.loc[df_seg['age'] < 30, 'Segment'] = 'Young Borrowers' # Q2: 11.73% default
    df_seg.loc[df_seg['age'] >= 60, 'Segment'] = 'Senior Borrowers' # Q2: 3.10% default for 60+

    # Calculate metrics by segment
    segment_metrics = {}
    for segment in df_seg['Segment'].unique():
        segment_data = df_seg[df_seg['Segment'] == segment]
        segment_metrics[segment] = {
            'count': len(segment_data),
            'pct': len(segment_data) / len(df_seg) * 100,
            'default_rate': segment_data['SeriousDlqin2yrs'].mean() * 100
        }

    return segment_metrics


def plot_score_distributions(df_score):
    """Plot risk score distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Risk score distribution by actual default
    df_score[df_score['SeriousDlqin2yrs'] == 0]['RiskScore'].hist(
        bins=50, alpha=0.7, label='Non-Default', ax=axes[0], color='green'
    )
    df_score[df_score['SeriousDlqin2yrs'] == 1]['RiskScore'].hist(
        bins=50, alpha=0.7, label='Default', ax=axes[0], color='red'
    )
    axes[0].set_xlabel('Risk Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Risk Score Distribution by Actual Outcome', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axvline(25, color='orange', linestyle='--', linewidth=2, label='Low/Med threshold')
    axes[0].axvline(50, color='red', linestyle='--', linewidth=2, label='Med/High threshold')

    # Default rate by risk category
    risk_default = df_score.groupby('RiskCategory')['SeriousDlqin2yrs'].mean() * 100
    axes[1].bar(range(len(risk_default)), risk_default.values,
                color=['green', 'orange', 'red'], alpha=0.7)
    axes[1].set_xticks(range(len(risk_default)))
    axes[1].set_xticklabels(risk_default.index, rotation=45)
    axes[1].set_ylabel('Default Rate (%)', fontsize=12)
    axes[1].set_title('Default Rate by Risk Category', fontsize=14, fontweight='bold')
    axes[1].grid(alpha=0.3, axis='y')
    for i, v in enumerate(risk_default.values):
        axes[1].text(i, v + 0.5, f'{v:.2f}%', ha='center', fontsize=11)

    plt.tight_layout()
    plt.show()


def plot_score_performance(df_score):
    """Plot score performance metrics."""
    
    # Score vs default rate (binned)
    df_score['ScoreBin'] = pd.cut(df_score['RiskScore'], bins=20)
    score_perf = df_score.groupby('ScoreBin').agg({
        'SeriousDlqin2yrs': ['mean', 'count']
    })
    score_perf = score_perf[score_perf[('SeriousDlqin2yrs', 'count')] > 100] # Filter small bins

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Default rate by score bin
    x_vals = range(len(score_perf))
    default_rates = score_perf[('SeriousDlqin2yrs', 'mean')].values * 100

    axes[0].plot(x_vals, default_rates, marker='o', linewidth=2, markersize=6, color='darkred')
    axes[0].fill_between(x_vals, default_rates, alpha=0.3, color='red')
    axes[0].set_xlabel('Risk Score (Binned, Low to High)', fontsize=12)
    axes[0].set_ylabel('Default Rate (%)', fontsize=12)
    axes[0].set_title('Default Rate Increases with Risk Score', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)

    # Population distribution
    risk_counts = df_score['RiskCategory'].value_counts()
    axes[1].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
                colors=['lightgreen', 'orange', 'lightcoral'], startangle=90)
    axes[1].set_title('Population Distribution by Risk Category', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
  

def plot_segment_performance(df):
    """
    Plot default rates by customer segment.
    """
    # Define segments
    df_plot = df.copy()
    df_plot['Segment'] = 'Standard'

    # Delinquency History segment
    df_plot.loc[(df_plot['NumberOfTime60-89DaysPastDueNotWorse'] > 0) | 
                (df_plot['NumberOfTimes90DaysLate'] > 0), 'Segment'] = 'Delinquency History'

    # Age-based segments (aligned with Q2 analysis)
    df_plot.loc[df_plot['age'] < 30, 'Segment'] = 'Young Borrowers' # Q2: 11.73% default
    df_plot.loc[df_plot['age'] >= 60, 'Segment'] = 'Senior Borrowers' # Q2: 3.10% default for 60+

    # Analyze by segment
    segment_analysis = df_plot.groupby('Segment').agg({
        'SeriousDlqin2yrs': ['count', 'mean']
    })
    segment_analysis.columns = ['Count', 'Default_Rate']
    segment_analysis['Default_Rate'] *= 100
    segment_analysis = segment_analysis.sort_values('Default_Rate', ascending=False)

    # Visualization
    plt.figure(figsize=(10, 6))
    colors = ['red' if x > df_plot['SeriousDlqin2yrs'].mean()*100 else 'green' 
              for x in segment_analysis['Default_Rate'].values]
    plt.bar(range(len(segment_analysis)), segment_analysis['Default_Rate'].values,
            color=colors, alpha=0.7)
    plt.xticks(range(len(segment_analysis)), segment_analysis.index, fontsize=11)
    plt.ylabel('Default Rate (%)', fontsize=12)
    plt.title('Default Rates by Customer Segment', fontsize=14, fontweight='bold')
    plt.axhline(df_plot['SeriousDlqin2yrs'].mean()*100, color='black', linestyle='--',
                linewidth=2, label='Overall Average')
    plt.legend()
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()


# MACHINE LEARNING MODEL FUNCTIONS

def prepare_features(df, include_rule_based=True):
    """
    Prepare features for ML model training.
    Handle missing values, create engineered features, and prepare X, y.
    
    Missing income values are imputed with median BEFORE calculating RiskScore.
    """
    df_ml = df.copy()

    # Handle missing values
    # Income: impute with median
    df_ml['MonthlyIncome'] = df_ml['MonthlyIncome'].fillna(df_ml['MonthlyIncome'].median())

    # Dependents: impute with median (0)
    df_ml['NumberOfDependents'] = df_ml['NumberOfDependents'].fillna(0)

    # Cap outliers
    df_ml['DebtRatio'] = df_ml['DebtRatio'].clip(upper=df_ml['DebtRatio'].quantile(0.99))
    df_ml['RevolvingUtilizationOfUnsecuredLines'] = df_ml['RevolvingUtilizationOfUnsecuredLines'].clip(
        upper=df_ml['RevolvingUtilizationOfUnsecuredLines'].quantile(0.99)
    )

    # HYBRID APPROACH: Calculate rule-based risk score as a feature
    if include_rule_based:
      df_ml = calculate_risk_scores(df_ml)

    # Feature engineering
    # Total delinquency count
    df_ml['TotalDelinquencies'] = (
        df_ml['NumberOfTime30-59DaysPastDueNotWorse'] +
        df_ml['NumberOfTime60-89DaysPastDueNotWorse'] +
        df_ml['NumberOfTimes90DaysLate']
    )

    # Has any delinquency flag
    df_ml['HasDelinquency'] = (df_ml['TotalDelinquencies'] > 0).astype(int)

    # Has severe delinquency (60+ days)
    df_ml['HasSevereDelinquency'] = (
        (df_ml['NumberOfTime60-89DaysPastDueNotWorse'] > 0) |
        (df_ml['NumberOfTimes90DaysLate'] > 0)
    ).astype(int)

    # Utilization risk tiers (aligned with RiskScore thresholds: >=0.8 for high, 0.6-0.8 for medium)
    df_ml['UtilizationHigh'] = (df_ml['RevolvingUtilizationOfUnsecuredLines'] >= 0.8).astype(int)
    df_ml['UtilizationMedium'] = (
        (df_ml['RevolvingUtilizationOfUnsecuredLines'] >= 0.3) &
        (df_ml['RevolvingUtilizationOfUnsecuredLines'] < 0.8)
    ).astype(int)

    # Debt burden indicator
    df_ml['HighDebtBurden'] = (df_ml['DebtRatio'] > 0.5).astype(int)

    # Age groups (aligned with Q2 analysis and RiskScore thresholds)
    df_ml['AgeYoung'] = (df_ml['age'] < 25).astype(int)  # Matches RiskScore penalty at line 96
    df_ml['AgeSenior'] = (df_ml['age'] >= 60).astype(int)  # Q2: 3.10% default for 60+

    # Income tiers
    income_q1 = df_ml['MonthlyIncome'].quantile(0.25)
    income_q3 = df_ml['MonthlyIncome'].quantile(0.75)
    df_ml['IncomeLow'] = (df_ml['MonthlyIncome'] <= income_q1).astype(int)
    df_ml['IncomeHigh'] = (df_ml['MonthlyIncome'] >= income_q3).astype(int)

    # Credit lines per real estate
    df_ml['CreditLinesPerRealEstate'] = df_ml['NumberOfOpenCreditLinesAndLoans'] / (
        df_ml['NumberRealEstateLoansOrLines'] + 1
    )

    # Has real estate
    df_ml['HasRealEstate'] = (df_ml['NumberRealEstateLoansOrLines'] > 0).astype(int)

    # Risk category one-hot encoding
    if include_rule_based:
        df_ml['RiskCategory_Low'] = (df_ml['RiskCategory'] == 'Low Risk').astype(int)
        df_ml['RiskCategory_Medium'] = (df_ml['RiskCategory'] == 'Medium Risk').astype(int)
        df_ml['RiskCategory_High'] = (df_ml['RiskCategory'] == 'High Risk').astype(int)

    # Select features for modeling
    feature_cols = [
        # Original features
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'NumberOfDependents',
        # Engineered features
        'TotalDelinquencies',
        'HasDelinquency',
        'HasSevereDelinquency',
        'UtilizationHigh',
        'UtilizationMedium',
        'HighDebtBurden',
        'AgeYoung',
        'AgeSenior',
        'IncomeLow',
        'IncomeHigh',
        'CreditLinesPerRealEstate',
        'HasRealEstate'
    ]

    # Add rule-based features if included
    if include_rule_based:
        feature_cols.extend([
            'RiskScore',
            'RiskCategory_Low',
            'RiskCategory_Medium',
            'RiskCategory_High'
        ])

    X = df_ml[feature_cols]
    y = df_ml['SeriousDlqin2yrs']

    return X, y, feature_cols


def train_ml_model(df, test_size=0.3, random_state=42, include_rule_based=True):
    """
    Train a CatBoost model for default prediction.
    """
  
    # Prepare features
    X, y, feature_cols = prepare_features(df, include_rule_based=include_rule_based)

    print(f"Features created: {len(feature_cols)}")
    if include_rule_based:
        print(f"  - Includes RiskScore and RiskCategory from rule-based model")
    print(f"Total samples: {len(X):,}")
    print(f"Default rate: {y.mean()*100:.2f}%")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")

    # Train CatBoost model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        auto_class_weights='Balanced',
        random_state=random_state
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50,
    )

    return model, X_train, X_test, y_train, y_test, feature_cols

  
def evaluate_ml_model(model, X_train, X_test, y_train, y_test, feature_cols):
    """
    Evaluate ML model performance with comprehensive metrics.
    """
    print("\n" + "="*80)
    print("MODEL EVALUATION METRICS")
    print("="*80)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    # AUC-ROC
    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n1. AUC-ROC SCORE:")
    print(f"   Training AUC:   {train_auc:.4f}")
    print(f"   Test AUC:       {test_auc:.4f}")
    print(f"   Difference:     {abs(train_auc - test_auc):.4f}")

    if test_auc >= 0.85:
        print("   Status: ✅ EXCELLENT (>=0.85)")
    elif test_auc >= 0.80:
        print("   Status: ✅ VERY GOOD (>=0.80)")
    elif test_auc >= 0.75:
        print("   Status: ⚠️ GOOD (>=0.75)")
    else:
        print("   Status: ❌ NEEDS IMPROVEMENT (<0.75)")

    # Precision, Recall, F1
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"\n2. CLASSIFICATION METRICS (Test Set):")
    print(f"   Precision:      {test_precision:.4f} (of predicted defaults, what % are correct)")
    print(f"   Recall:         {test_recall:.4f} (of actual defaults, what % we catch)")
    print(f"   F1-Score:       {test_f1:.4f} (harmonic mean of precision & recall)")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n3. CONFUSION MATRIX (Test Set):")
    print(f"   True Negatives:  {tn:,} (correctly predicted non-defaults)")
    print(f"   False Positives: {fp:,} (incorrectly predicted as defaults)")
    print(f"   False Negatives: {fn:,} (missed defaults - COSTLY!)")
    print(f"   True Positives:  {tp:,} (correctly caught defaults)")

    # Business Metrics
    approval_rate = (tn + fn) / len(y_test) * 100
    actual_default_rate = y_test.mean() * 100
    predicted_default_rate = y_test_pred.mean() * 100

    # Calculate default rate if we approve predicted non-defaults
    approved_defaults = fn
    total_approved = tn + fn
    portfolio_default_rate = (approved_defaults / total_approved * 100) if total_approved > 0 else 0

    # Defaults prevented
    defaults_prevented = tp
    total_defaults = tp + fn
    prevention_rate = (defaults_prevented / total_defaults * 100) if total_defaults > 0 else 0

    print(f"\n4. BUSINESS IMPACT (Test Set):")
    print(f"   Approval Rate:         {approval_rate:.2f}% (predicted non-defaults)")
    print(f"   Actual Default Rate:   {actual_default_rate:.2f}%")
    print(f"   Predicted Default Rate:{predicted_default_rate:.2f}%")
    print(f"   Portfolio Default Rate:{portfolio_default_rate:.2f}% (if we approve predicted non-defaults)")
    print(f"   Defaults Prevented:    {defaults_prevented:,} out of {total_defaults:,} ({prevention_rate:.1f}%)")

    # Feature Importance
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(f"\n5. TOP 10 FEATURE IMPORTANCE:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"   {row['Feature']:45s} {row['Importance']:6.2f}")

    return {
        'train_auc': train_auc,
        'test_auc': test_auc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': cm,
        'feature_importance': feature_importance,
        'y_train_proba': y_train_proba,
        'y_test_proba': y_test_proba
    }

  
def plot_ml_performance(model, X_train, X_test, y_train, y_test, metrics):
    """
    Create comprehensive ML model performance visualizations.
    """
    fig = plt.figure(figsize=(18, 12))

    # 1. ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    fpr_train, tpr_train, _ = roc_curve(y_train, metrics['y_train_proba'])
    fpr_test, tpr_test, _ = roc_curve(y_test, metrics['y_test_proba'])

    ax1.plot(fpr_train, tpr_train, label=f"Train (AUC={metrics['train_auc']:.4f})",
             linewidth=2, alpha=0.7)
    ax1.plot(fpr_test, tpr_test, label=f"Test (AUC={metrics['test_auc']:.4f})",
             linewidth=2, alpha=0.7)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.set_xlabel('False Positive Rate', fontsize=11)
    ax1.set_ylabel('True Positive Rate', fontsize=11)
    ax1.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Precision-Recall Curve
    ax2 = plt.subplot(2, 3, 2)
    precision, recall, _ = precision_recall_curve(y_test, metrics['y_test_proba'])
    avg_precision = average_precision_score(y_test, metrics['y_test_proba'])

    ax2.plot(recall, precision, linewidth=2, label=f'AP={avg_precision:.4f}')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    cm = metrics['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt=',.0f', cmap='Blues', ax=ax3,
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    ax3.set_xlabel('Predicted', fontsize=11)
    ax3.set_ylabel('Actual', fontsize=11)
    ax3.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    # 4. Feature Importance
    ax4 = plt.subplot(2, 3, 4)
    top_features = metrics['feature_importance'].head(15).sort_values('Importance')
    ax4.barh(range(len(top_features)), top_features['Importance'], color='steelblue', alpha=0.7)
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'], fontsize=9)
    ax4.set_xlabel('Importance', fontsize=11)
    ax4.set_title('Top 15 Feature Importance', fontsize=13, fontweight='bold')
    ax4.grid(alpha=0.3, axis='x')

    # 5. Prediction Distribution
    ax5 = plt.subplot(2, 3, 5)
    ax5.hist(metrics['y_test_proba'][y_test == 0], bins=50, alpha=0.6,
             label='Non-Default', color='green', density=True)
    ax5.hist(metrics['y_test_proba'][y_test == 1], bins=50, alpha=0.6,
             label='Default', color='red', density=True)
    ax5.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold=0.5')
    ax5.set_xlabel('Predicted Probability', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Prediction Distribution', fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    MODEL PERFORMANCE SUMMARY

    AUC-ROC:      {metrics['test_auc']:.4f}
    Precision:    {metrics['precision']:.4f}
    Recall:       {metrics['recall']:.4f}
    F1-Score:     {metrics['f1']:.4f}

    CONFUSION MATRIX:
    True Negatives:   {metrics['confusion_matrix'][0,0]:,}
    False Positives:  {metrics['confusion_matrix'][0,1]:,}
    False Negatives:  {metrics['confusion_matrix'][1,0]:,}
    True Positives:   {metrics['confusion_matrix'][1,1]:,}

    MODEL STATUS: {'EXCELLENT' if metrics['test_auc'] >= 0.85 else 'VERY GOOD' if metrics['test_auc'] >= 0.80 else 'GOOD'}
    """

    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center')

    plt.suptitle('CatBoost Model Performance Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()


def plot_shap_importance(model, X_train, X_test, feature_cols, max_display=10):
    """
    Plot SHAP feature importance for model interpretability.
    """
    print("\n" + "="*80)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*80)

    # Create SHAP explainer
    sample_size = min(1000, len(X_train))
    X_sample = X_train.sample(n=sample_size, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # If binary classification, take positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Plot 1: SHAP Bar Chart (Feature Importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type='bar',
        max_display=max_display,
        show=False
    )
    plt.title('Top 10 Features by SHAP Importance', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Mean |SHAP value| (Average Impact on Prediction)', fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=11)
    plt.tight_layout()
    plt.show()

    # Plot 2: SHAP Beeswarm
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_sample,
        max_display=max_display,
        show=False
    )
    plt.title('Top 10 Features - Detailed SHAP Impact Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12)
    plt.tick_params(axis='y', labelsize=12)
    plt.tick_params(axis='x', labelsize=11)

    # Add colorbar label with better positioning
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel('Feature Value', fontsize=11, rotation=270, labelpad=20)

    plt.tight_layout()
    plt.show()

    # Print interpretation
    print("\n" + "="*80)
    print("SHAP INTERPRETATION:")
    print("="*80)
    print("""
    The SHAP plots show:

    1. BEESWARM PLOT (Left):
       - Each dot is a data point
       - X-axis: SHAP value (impact on prediction)
       - Color: Feature value (red=high, blue=low)
       - Shows which features have the biggest impact and how

    2. BAR PLOT (Right):
       - Shows average absolute SHAP value per feature
       - Higher bar = more important feature overall
       - Ranks features by global importance

    KEY INSIGHTS:
       - Features are ranked by their impact on model predictions (see bar plot)
       - Red dots pushing right = high feature value -> higher default risk
       - Blue dots pushing left = low feature value -> lower default risk
       - The top-ranked features show the strongest influence on default predictions
    """)


def visualize_prediction_strategy(df):
    """
    Create all visualizations for prediction strategy analysis.
    """
    print_section_header("QUESTION 3: VISUALIZATIONS")

    if 'RiskScore' not in df.columns:
        df = calculate_risk_scores(df)

    plot_score_distributions(df)
    plot_score_performance(df)
    plot_segment_performance(df)

    return df


# Main execution
if __name__ == '__main__':
    # Load data
    df = load_data('cs-training.csv')

    # Calculate prediction metrics
    prediction_results = calculate_prediction_metrics(df)

    print(f"  • df_scored: DataFrame with RiskScore and RiskCategory")
    print(f"  • rule_based: {len(prediction_results['rule_based'])} metrics")
    print(f"  • segmentation: {len(prediction_results['segmentation'])} segments")

    print(f"  Overall default rate: {prediction_results['rule_based']['overall_default_rate']:.2f}%")
    print(f"  High risk default rate: {prediction_results['rule_based']['high_risk_default_rate']:.2f}%")
    print(f"  Risk separation: {prediction_results['rule_based']['risk_separation']:.1f}x")
