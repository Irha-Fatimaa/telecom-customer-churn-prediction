from pathlib import Path
import json, sys
import joblib
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from features import add_features
ROOT=Path(__file__).resolve().parent
model=joblib.load(ROOT/'models'/'churn_model.joblib')
meta=json.loads((ROOT/'models'/'metadata.json').read_text())

sample={
 'customerID':'DEMO-001','gender':'Female','SeniorCitizen':0,'Partner':'No','Dependents':'No','tenure':5,
 'PhoneService':'Yes','MultipleLines':'No','InternetService':'Fiber optic','OnlineSecurity':'No','OnlineBackup':'No',
 'DeviceProtection':'No','TechSupport':'No','StreamingTV':'Yes','StreamingMovies':'Yes','Contract':'Month-to-month',
 'PaperlessBilling':'Yes','PaymentMethod':'Electronic check','MonthlyCharges':95.0,'TotalCharges':475.0
}
df=add_features(pd.DataFrame([sample])).drop(columns=['customerID'])
p=float(model.predict_proba(df)[:,1][0]); label='Churn' if p>=meta['threshold'] else 'Stay'
print(f"Prediction: {label}\nChurn probability: {p:.3%}\nDecision threshold: {meta['threshold']:.2f}")
