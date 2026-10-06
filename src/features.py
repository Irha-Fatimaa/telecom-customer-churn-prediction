import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    # Basic cleaning
    x['TotalCharges'] = pd.to_numeric(x['TotalCharges'], errors='coerce')
    x['TotalCharges'] = x['TotalCharges'].fillna(x['MonthlyCharges'] * x['tenure'].clip(lower=1))

    # Business-motivated engineered features
    service_cols = ['PhoneService','MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    x['NumServices'] = sum((x[c] == 'Yes').astype(int) for c in service_cols)
    x['HasInternet'] = (x['InternetService'] != 'No').astype(int)
    x['HasSupportBundle'] = ((x['OnlineSecurity']=='Yes') & (x['TechSupport']=='Yes')).astype(int)
    x['AutoPay'] = x['PaymentMethod'].isin(['Bank transfer (automatic)','Credit card (automatic)']).astype(int)
    x['FamilyAccount'] = ((x['Partner']=='Yes') | (x['Dependents']=='Yes')).astype(int)
    x['AvgMonthlySpend'] = x['TotalCharges'] / x['tenure'].clip(lower=1)
    x['ChargeIncreaseRatio'] = x['MonthlyCharges'] / x['AvgMonthlySpend'].replace(0, np.nan)
    x['ChargeIncreaseRatio'] = x['ChargeIncreaseRatio'].replace([np.inf,-np.inf],np.nan).fillna(1.0)
    x['TenureBand'] = pd.cut(x['tenure'], bins=[-1,6,12,24,48,72], labels=['0-6m','7-12m','13-24m','25-48m','49-72m'])
    x['HighMonthlyCharge'] = (x['MonthlyCharges'] >= 90).astype(int)
    x['MonthToMonth'] = (x['Contract']=='Month-to-month').astype(int)
    x['SupportGap'] = (((x['InternetService']!='No')) & ((x['TechSupport']=='No') | (x['OnlineSecurity']=='No'))).astype(int)
    return x
