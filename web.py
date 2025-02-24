import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

with open('./xgboost.pkl', 'rb') as c:
    model = pickle.load(c)


st.title('Risk Calculator for Moderate to Severe Granulocytopenia Induced by Antithyroid Drugs')
st.subheader('This risk score assesses the likelihood of moderate-to-severe granulocytopenia in hyperthyroid patients treated with antithyroid drugs, defined as a granulocyte count <1 × 10^9/L following treatment.')
st.divider()

# 创建一个数字输入框
gender_input = st.selectbox('Sex',('man', 'woman'))
ne_input = st.number_input('NE (×10^9/L)', value=0.0, step=1.0)
wbc_input = st.number_input('WBC (×10^9/L)', value=0.0, step=1.0)
hgb_input = st.number_input('HGB (g/L)', value=0.0, step=1.0)
plt_input = st.number_input('PLT (×10^9/L)', value=0.0, step=1.0)
tpoab_input = st.number_input('TPOAb (IU/mL)', value=0.0, step=1.0)

sex = np.where(gender_input=='man',1,0)
NE = (ne_input - 3.32766399)/1.6190891
WBC = (wbc_input - 5.96424336)/2.00761288
HGB = (hgb_input - 133.33905933)/18.53388135
PLT = (plt_input - 240.80848453)/77.45826165
TPOAb = (tpoab_input - 388.90537128)/479.11668147

features = np.array([NE,WBC,HGB,PLT,TPOAb,sex]).reshape(1,-1)

if st.button('Predict'):
    col1, col2 = st.columns(2)
    p = model.predict_proba(features)[:,1]*100
    col1.metric("Score", int(p), )