import streamlit as st
import pandas as pd
import numpy as np
import joblib


kmeans = joblib.load('E:\ML_project\Customer_Segementation_clustering\kmeans_model.pkl')  # Load the KMeans model
scaler = joblib.load('E:\ML_project\Customer_Segementation_clustering\scaler.pkl')  # Load the scaler

st.title("Customer Segmentation Prediction")
st.write("Enter customer details to predict the segment ")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
income=st.number_input("Income", min_value=0, max_value=200000, value=50000)
total_spending=st.number_input("Total Spending (sum of purchases)",min_value=0, max_value=5000, value=1000)
num_web_purchases=st.number_input("Number of Web Purchases", min_value=0, max_value=100, value=10)

num_of_store_purchases=st.number_input("Number of Store Purchases", min_value=0, max_value=100, value=10)
num_of_web_visits=st.number_input("Number of Web Visits per Month", min_value=0, max_value=50, value=3)
recency=st.number_input("Recency (days since last purchase)", min_value=0, max_value=365, value=30)


input_data = pd.DataFrame({
    'Age': [age],
    'Income': [income],
    'Total_Spending': [total_spending],
    'NumWebPurchases': [num_web_purchases],
    'NumStorePurchases': [num_of_store_purchases],
    'NumWebVisitsMonth': [num_of_web_visits],
    'Recency': [recency]

})

input_scaled = scaler.transform(input_data)  # Scale the input data

if st.button("Predict Segment"):
    Cluster = kmeans.predict(input_scaled)[0]

    st.success(f"Predicted Segment: Cluster {Cluster}")

    # st.write("""
    #             Cluster 0: High income, High Spending -> Premium customer
    #             Cluster 1: Moderate income, Moderate Spending -> Regular customer   
             
    #             Cluster 2: Low income, Low Spending -> Budget customer
    #             Cluster 3: High web purchases, Low store purchases -> Digital buyers
    #             Cluster 4: Low web purchases, High store purchases -> In-store shoppers
    #             Cluster 5: High web visits, Low purchases -> Browsers

    # """

    # )