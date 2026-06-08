# PAIDY Credit Risk Analysis

It's an analysis of consumer credit data to predict delinquency and provide risk management recommendations

# Paidy Credit Risk Analysis

Comprehensive analysis of consumer credit data to predict delinquency and provide risk management recommendations.

## Project Overview

This analysis addresses the Paidy Risk Exercise requirements by answering four key questions:

1. What can you tell us about the population of consumers?
2. What insights can you draw from the data?
3. How can we use this data to predict that a consumer might not pay?
4. What are some recommendations you can make to manage the risk of these consumers?

## Files Structure
```
paidy/
├── README.md                          # This file
├── Risk_Candidate_Exercise_-_202605.pdf # Exercise requirements
├── Data Dictionary.xls                # Feature descriptions
├── cs-training.csv                    # Training dataset (150K records)
├── cs-test.csv                        # Test dataset (101K records)
├── sampleEntry.csv                    # Sample submission format
├── credit_risk_analysis_notebook.ipynb # Jupyter notebook (recommended)
├── utils.py                           # Common utility functions
├── question_1_population_analysis.py  # Question 1 analysis
├── question_2_insights.py             # Question 2 analysis
├── question_3_prediction_strategy.py  # Question 3 analysis
└── question_4_risk_recommendations.py # Question 4 analysis
````

## Dataset Details

### Target Variable 
- SeriousDlqin2yrs: Binary (0/1), 1 = experienced 90+ days delinquency
  
### Features (10 predictors)
1. RevolvingUtilizationOfUnsecuredLines: Credit utilization ratio
2. Age: Borrower age in years
3. NumberOfTime30-59DaysPastDueNotWorse: Count of 30-59 day lates
4. DebtRatio: Monthly debt / monthly income
5. MonthlyIncome: Monthly income ($)
6. NumberOfOpenCreditLinesAndLoans: Count of credit lines
7. NumberOfTimes90DaysLate: Count of 90+ day lates
8. NumberRealEstateLoansOrLines: Count of real estate loans
9. NumberOfTime60-89DaysPastDueNotWorse: Count of 60-89 day lates
10. NumberOfDependents: Number of dependents
