import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

model=tf.keras.models.load_model('model.h5')

with open('ScaledData.pkl','rb') as f:
    scaler=pickle.load(f)

with open('LabelEncoder_Gender.pkl','rb') as f:
    le_gender=pickle.load(f)
with open('OneHotEncoder_Geography.pkl','rb') as f:
    ohe_geography=pickle.load(f)

st.title("Customer Churn Prediction")
geography = st.selectbox('Geography', ohe_geography.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [le_gender.transform([gender ]) [0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'EstimatedSalary': [estimated_salary]
})
input_df=pd.DataFrame(input_data)
geo_encoded=ohe_geography.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=ohe_geography.get_feature_names_out(['Geography']))

input_data=pd.concat([input_df,geo_encoded_df],axis=1)

input_scaled=scaler.transform(input_data)

prediction=model.predict(input_scaled)
prediction_probability=prediction[0][0]
st.write(f"Prediction Probability: {prediction_probability:.2f}")
if prediction_probability>0.5:
    st.write("The customer is likely to leave the bank.")
else:
    st.write("The customer is likely to stay with the bank.")