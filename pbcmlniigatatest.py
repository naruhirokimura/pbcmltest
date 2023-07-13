import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt

st.title('The prediction model for treatrment response in PBC patients')
st.write('This app aims to predict treatment response for Primary Biliary Cholangitis patients base on Machine learning')
st.write('Please enter pre-treatment data by moving the slide bar.')

modelst2 = xgb.Booster()
modelst2.load_model("modelst.json")


TP = st.sidebar.slider(label='Total protein (g/dL)', min_value=5.5, max_value=9.3,value=8.0, step=0.1)
ALT = st.sidebar.slider(label='ALT (IU/L)', min_value=8, max_value=1058,value=80)
Tbil = st.sidebar.slider(label='T-Bil (mg/dL)', min_value=0.2, max_value=4.3,value=1.0, step=0.1)

sample = np.array([['TP','ALT','Tbil'],[TP, ALT, Tbil]])
dfsample = pd.DataFrame(data=[[TP, ALT, Tbil]], columns=['TP','ALT','Tbil'])
st.write(dfsample)    

predst = modelst2.predict(xgb.DMatrix(dfsample))


if predst <0.841:
  st.write("This patient may not archieve Paris II criteria, please consider additional treatment.")
else:
  st.write('This patient will archieve Paris II criteria')
