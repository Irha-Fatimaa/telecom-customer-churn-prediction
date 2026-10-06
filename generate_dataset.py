import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N = 6000
rng = np.random.default_rng(SEED)

customer_id = [f"CUST-{i:05d}" for i in range(1, N+1)]
gender = rng.choice(["Female","Male"], N)
senior = rng.binomial(1, 0.17, N)
partner = rng.choice(["Yes","No"], N, p=[0.49,0.51])
dependents = np.where(partner=="Yes", rng.choice(["Yes","No"],N,p=[0.38,0.62]), rng.choice(["Yes","No"],N,p=[0.16,0.84]))
tenure = np.clip(np.round(rng.gamma(2.0, 16.0, N)),0,72).astype(int)
phone = rng.choice(["Yes","No"],N,p=[0.90,0.10])
multiple = np.where(phone=="No","No phone service",rng.choice(["Yes","No"],N,p=[0.46,0.54]))
internet = rng.choice(["Fiber optic","DSL","No"],N,p=[0.45,0.40,0.15])

def service(prob_yes=0.45):
    vals = rng.choice(["Yes","No"], N, p=[prob_yes,1-prob_yes])
    return np.where(internet=="No","No internet service",vals)

online_security=service(.36)
online_backup=service(.43)
device_protection=service(.44)
tech_support=service(.34)
streaming_tv=service(.48)
streaming_movies=service(.48)
contract=rng.choice(["Month-to-month","One year","Two year"],N,p=[.56,.24,.20])
paperless=rng.choice(["Yes","No"],N,p=[.61,.39])
payment=rng.choice(["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"],N,p=[.33,.22,.23,.22])

base=(18 + (phone=="Yes")*15 + (multiple=="Yes")*6).astype(float)
base += (internet=="DSL")*26 + (internet=="Fiber optic")*46
for s in [online_security,online_backup,device_protection,tech_support,streaming_tv,streaming_movies]:
    base += (s=="Yes")*5.5
monthly=np.clip(base+rng.normal(0,4.5,N),18,125).round(2)
total=(monthly*np.maximum(tenure,1)*(0.94+rng.normal(0,.035,N))).clip(18).round(2)

# Realistic churn mechanism with controlled noise.
logit = -1.55
logit += 1.15*(contract=="Month-to-month") - .50*(contract=="One year") - 1.00*(contract=="Two year")
logit += .55*(internet=="Fiber optic") + .22*senior
logit += .50*(payment=="Electronic check") + .20*(paperless=="Yes")
logit += .48*(tech_support=="No") + .40*(online_security=="No")
logit += .42*(tenure<12) + .18*((tenure>=12)&(tenure<24)) - .60*(tenure>=48)
logit += .25*(monthly>90) - .22*(partner=="Yes") - .18*(dependents=="Yes")
logit += rng.normal(0,.42,N)
prob=1/(1+np.exp(-logit))
churn=np.where(rng.random(N)<prob,"Yes","No")

df=pd.DataFrame({
    'customerID':customer_id,'gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':dependents,
    'tenure':tenure,'PhoneService':phone,'MultipleLines':multiple,'InternetService':internet,
    'OnlineSecurity':online_security,'OnlineBackup':online_backup,'DeviceProtection':device_protection,
    'TechSupport':tech_support,'StreamingTV':streaming_tv,'StreamingMovies':streaming_movies,
    'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,
    'MonthlyCharges':monthly,'TotalCharges':total,'Churn':churn
})
# Introduce a small number of blank TotalCharges values to demonstrate cleaning.
blank_idx=rng.choice(df.index,size=35,replace=False)
df.loc[blank_idx,'TotalCharges']=np.nan
out=Path(__file__).resolve().parent/'data'/'telecom_customer_churn.csv'
df.to_csv(out,index=False)
print(f"Saved {len(df):,} rows to {out}")
print(f"Churn rate: {(df.Churn=='Yes').mean():.3%}")
