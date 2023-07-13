import pickle
import numpy as np
import pandas as pd
import streamlit as st

with open('model.pickle', 'wb') as f:
    model = pickle.load(f)


TP = st.sidebar.slider(label='Total protein (g/dL)', min_value=5.5, max_value=9.3,value=8.0, step=0.1)
ALT = st.sidebar.slider(label='ALT (IU/L)', min_value=8, max_value=1058,value=80)
Tbil = st.sidebar.slider(label='T-Bil (mg/dL)', min_value=0.2, max_value=4.3,value=1.0, step=0.1)

sample = np.array([['TP','ALT','Tbil'],[TP, ALT, Tbil]])
dfsample = pd.DataFrame(data=[[TP, ALT, Tbil]], columns=['TP','ALT','Tbil'])
st.write(dfsample)    

pd1=model.predict_proba(dfsample)
st.write(pd1)   
