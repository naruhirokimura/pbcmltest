import pickle
import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn import metrics
import matplotlib.pyplot as plt

st.title('The prediction model for treatrment response in patients with PBC.')
st.write('This app aims to predict treatment response for patients with Primary Biliary Cholangitis base on Machine learning')
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
st.write(predst)
st.write("the patient with data over 0.841 would arcieve Paris II criteria.")

if predst <0.841:
  st.write("This patient may not archieve Paris II criteria, please consider additional treatment.")
else:
  st.write('This patient will archieve Paris II criteria')

fig = plt.figure()
fig.set_size_inches(7, 7)
ax1 = fig.add_subplot(111, projection='3d')
df3Dnon = pd.read_csv('pcbmlvalidation3Dtarget-XGB-.csv')
df3Dres = pd.read_csv('pcbmlvalidation3Dtarget+XGB+.csv')
df3Dresnon = pd.read_csv('pcbmlvalidation3Dtarget+XGB-.csv')
df3Dnonres = pd.read_csv('pcbmlvalidation3Dtarget-XGB+.csv')
# X,Y,Z軸にラベルを設定
ax1.set_xlabel("Total Protein")
ax1.set_ylabel("ALT")
ax1.set_zlabel("T-Bil")
sc = ax1.scatter(df3Dnon.TP, df3Dnon.ALT, df3Dnon.Tbil, s=100,color="Black")
sc = ax1.scatter(df3Dres.TP, df3Dres.ALT, df3Dres.Tbil, s=100, color="purple")
sc = ax1.scatter(df3Dresnon.TP, df3Dresnon.ALT, df3Dresnon.Tbil, s=100, color="red")
sc = ax1.scatter(df3Dnonres.TP, df3Dnonres.ALT, df3Dnonres.Tbil, s=100, color="blue")
sc = ax1.scatter(dfsample.TP, dfsample.ALT, dfsample.Tbil, s=100, color="green")
st.pyplot(fig)
st.write("Black: Non responder and Machine learning predicted")
st.write("Blue: non responder but Machine learning mis predicted")
st.write("Red: responder but Machine learning mis predicted")
st.write("Purple: responder and Machine learning predicted")
st.write("Green: sample data")
