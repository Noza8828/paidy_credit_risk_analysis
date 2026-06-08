"""
Utility functions for credit risk analysis
Common functions used across all analysis questions
"""

import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

def load_data(train_path='cs-training.csv'):
    """Load and prepare the training dataset."""
    import os
    
    # If relative path is provided, prepend the data folder
    if not os.path.isabs(train_path) and not os.path.exists(train_path):
        # Get the parent directory (paidy root)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        data_dir = os.path.join(parent_dir, 'data')
        train_path = os.path.join(data_dir, train_path)
        
    df = pd.read_csv(train_path)
    
    # Drop the unnamed index column if it exists
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        
    print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df

def get_feature_descriptions():
    """Return dictionary of feature descriptions."""
    return {
        'SeriousDlqin2yrs': 'Person experienced 90 days past due delinquency or worse (TARGET)',
        'RevolvingUtilizationOfUnsecuredLines': 'Total balance on credit cards and personal lines of credit / credit limits',
        'age': 'Age of borrower in years',
        'NumberOfTime30-59DaysPastDueNotWorse': 'Number of times borrower has been 30-59 days past due (last 2 years)',
        'DebtRatio': 'Monthly debt payments, alimony, living costs / monthly gross income',
        'MonthlyIncome': 'Monthly income',
        'NumberOfOpenCreditLinesAndLoans': 'Number of open loans and lines of credit',
        'NumberOfTimes90DaysLate': 'Number of times borrower has been 90 days or more past due',
        'NumberRealEstateLoansOrLines': 'Number of mortgage and real estate loans',
        'NumberOfTime60-89DaysPastDueNotWorse': 'Number of times borrower has been 60-89 days past due (last 2 years)',
        'NumberOfDependents': 'Number of dependents in family excluding themselves'
    }

def print_section_header(title):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def print_subsection_header(title):
    """Print formatted subsection header."""
    print("\n" + "-"*80)
    print(title)
    print("-" * 80)
