import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from features import add_features

ROOT=Path(__file__).resolve().parent
model=joblib.load(ROOT/'models'/'churn_model.joblib')
meta=json.loads((ROOT/'models'/'metadata.json').read_text())

st.set_page_config(page_title='Telecom Churn Predictor',page_icon='📉',layout='wide')
st.title('Telecom Customer Churn Predictor')
st.caption('Enter customer/account information to estimate churn risk and support retention decisions.')

c1,c2,c3=st.columns(3)
with c1:
    gender=st.selectbox('Gender',['Female','Male']); senior=st.selectbox('Senior Citizen',[0,1]); partner=st.selectbox('Partner',['No','Yes']); dependents=st.selectbox('Dependents',['No','Yes']); tenure=st.slider('Tenure (months)',0,72,12)
    phone=st.selectbox('Phone Service',['Yes','No']); multiple=st.selectbox('Multiple Lines',['No','Yes','No phone service'])
with c2:
    internet=st.selectbox('Internet Service',['Fiber optic','DSL','No']); security=st.selectbox('Online Security',['No','Yes','No internet service']); backup=st.selectbox('Online Backup',['No','Yes','No internet service']); protection=st.selectbox('Device Protection',['No','Yes','No internet service']); support=st.selectbox('Tech Support',['No','Yes','No internet service']); tv=st.selectbox('Streaming TV',['No','Yes','No internet service']); movies=st.selectbox('Streaming Movies',['No','Yes','No internet service'])
with c3:
    contract=st.selectbox('Contract',['Month-to-month','One year','Two year']); paperless=st.selectbox('Paperless Billing',['Yes','No']); payment=st.selectbox('Payment Method',['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)']); monthly=st.number_input('Monthly Charges',18.0,125.0,70.0,1.0); total=st.number_input('Total Charges',18.0,10000.0,float(max(18,monthly*max(tenure,1))),10.0)

if st.button('Predict Churn Risk',type='primary'):
    row=pd.DataFrame([{'customerID':'APP','gender':gender,'SeniorCitizen':senior,'Partner':partner,'Dependents':dependents,'tenure':tenure,'PhoneService':phone,'MultipleLines':multiple,'InternetService':internet,'OnlineSecurity':security,'OnlineBackup':backup,'DeviceProtection':protection,'TechSupport':support,'StreamingTV':tv,'StreamingMovies':movies,'Contract':contract,'PaperlessBilling':paperless,'PaymentMethod':payment,'MonthlyCharges':monthly,'TotalCharges':total}])
    x=add_features(row).drop(columns=['customerID'])
    p=float(model.predict_proba(x)[:,1][0]); churn=p>=meta['threshold']
    st.metric('Churn probability',f'{p:.1%}')
    if churn: st.error('High churn risk - prioritize for a retention action.')
    else: st.success('Lower churn risk - continue normal engagement.')
    st.progress(min(max(p,0.0),1.0))
    st.caption(f"Model: {meta['selected_model']} | decision threshold: {meta['threshold']:.2f}")
