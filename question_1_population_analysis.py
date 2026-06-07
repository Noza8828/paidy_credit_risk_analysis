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
    print(f" - Mean dependents: {df['NumberOfDependents'].mean():.2f}")
    print(f" - Median dependents: {df['NumberOfDependents'].median():.0f}")
    print(f" - % with no dependents: {(df['NumberOfDependents'] == 0).mean()*100:.1f}%")
