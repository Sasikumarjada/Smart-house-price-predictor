#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Load Dataset
@st.cache
def load_data(file_path):
    return pd.read_csv(file_path)

# Load data
data = load_data(r"C:\Users\sasik\My_Project\Housing.csv")

# Preprocessing
def preprocess_data(data):
    # Encode categorical variables
    label_encoders = {}
    for col in ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 
                'airconditioning', 'prefarea', 'furnishingstatus']:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    # Split features and target
    X = data.drop('price', axis=1)
    y = data['price']
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders

X, y, scaler, label_encoders = preprocess_data(data)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building
model = XGBRegressor(random_state=42, objective='reg:squarederror')
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Save Model and Preprocessing Artifacts
joblib.dump(model, "house_price_xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

# Streamlit Interface
st.title("House Price Prediction App")

# Dropdown for selecting area name in Hyderabad
area_names = ['Banjara Hills', 'Jubilee Hills', 'Gachibowli', 'Kukatpally', 'Madhapur', 'Hitech City', 
              'Begumpet', 'Secunderabad', 'Ameerpet', 'Old City', 'Hi-Tech City', 'Shamshabad', 'A S Rao Nagar', 
              'Kondapur', 'Chanda Nagar', 'Malkajgiri', 'LB Nagar', 'Nagole', 'Goshamahal', 'Musheerabad']  # Example list of areas in Hyderabad
selected_area = st.selectbox("Select Area", area_names)

# Area name encoding logic (if necessary)
# Here you can add logic to encode the selected area into numerical value if you need to use it for model prediction
# For simplicity, we'll skip encoding area in this example, as it is not in the dataset

# Input fields for user input
area_sqft = st.number_input("Area (in sqft)", min_value=500, max_value=13300000, value=7420)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=4)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
stories = st.number_input("Number of Stories", min_value=1, max_value=4, value=2)
mainroad = st.selectbox("Mainroad (yes=1, no=0)", [1, 0])
guestroom = st.selectbox("Guestroom (yes=1, no=0)", [1, 0])
basement = st.selectbox("Basement (yes=1, no=0)", [1, 0])
hotwaterheating = st.selectbox("Hotwaterheating (yes=1, no=0)", [1, 0])
airconditioning = st.selectbox("Airconditioning (yes=1, no=0)", [1, 0])
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=2)
prefarea = st.selectbox("Preferred Area (yes=1, no=0)", [1, 0])
furnishingstatus = st.selectbox(
    "Furnishing Status (furnished=0, semi-furnished=1, unfurnished=2)", [0, 1, 2]
)

# Predict
if st.button("Predict Price"):
    # Assume the selected area does not directly influence prediction in the model (you can modify it as needed)
    input_data = np.array([area_sqft, bedrooms, bathrooms, stories, mainroad, guestroom,
                           basement, hotwaterheating, airconditioning, parking,
                           prefarea, furnishingstatus]).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    st.write(f"Predicted House Price: {prediction[0]:,.2f}")


# In[ ]:




