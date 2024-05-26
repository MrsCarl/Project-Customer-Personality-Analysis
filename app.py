{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46d3d97e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# app.py\n",
    "import streamlit as st\n",
    "import joblib\n",
    "import numpy as np\n",
    "\n",
    "# Load the trained model\n",
    "model = joblib.load('gradient_boosting_model.pkl')\n",
    "\n",
    "# Title of the Streamlit app\n",
    "st.title('Customer Prediction Model')\n",
    "\n",
    "# Input features\n",
    "st.header('Enter customer data:')\n",
    "education = st.selectbox('Education', ['Basic', 'Graduation', 'Master', 'PhD'])\n",
    "income = st.number_input('Income', value=0.0)\n",
    "kidhome = st.number_input('Kidhome', value=0)\n",
    "teenhome = st.number_input('Teenhome', value=0)\n",
    "recency = st.number_input('Recency', value=0)\n",
    "wines = st.number_input('Wines', value=0)\n",
    "fruits = st.number_input('Fruits', value=0)\n",
    "meat = st.number_input('Meat', value=0)\n",
    "fish = st.number_input('Fish', value=0)\n",
    "sweets = st.number_input('Sweets', value=0)\n",
    "gold = st.number_input('Gold', value=0)\n",
    "num_deals_purchases = st.number_input('NumDealsPurchases', value=0)\n",
    "num_web_purchases = st.number_input('NumWebPurchases', value=0)\n",
    "num_catalog_purchases = st.number_input('NumCatalogPurchases', value=0)\n",
    "num_store_purchases = st.number_input('NumStorePurchases', value=0)\n",
    "num_web_visits_month = st.number_input('NumWebVisitsMonth', value=0)\n",
    "time_enrolled_days = st.number_input('Time_Enrolled_Days', value=0)\n",
    "age = st.number_input('Age', value=0)\n",
    "spent = st.number_input('Spent', value=0)\n",
    "living_with = st.selectbox('Living_With', ['Alone', 'Partner', 'Parents', 'Others'])\n",
    "children = st.number_input('Children', value=0)\n",
    "family_size = st.number_input('Family_Size', value=0)\n",
    "is_parent = st.selectbox('Is_Parent', ['Yes', 'No'])\n",
    "total_promos = st.number_input('Total_Promos', value=0)\n",
    "agglomerative_cluster = st.number_input('Agglomerative_Cluster', value=0)\n",
    "\n",
    "# Map categorical features to numeric values\n",
    "education_mapping = {'Basic': 0, 'Graduation': 1, 'Master': 2, 'PhD': 3}\n",
    "living_with_mapping = {'Alone': 0, 'Partner': 1, 'Parents': 2, 'Others': 3}\n",
    "is_parent_mapping = {'Yes': 1, 'No': 0}\n",
    "\n",
    "education = education_mapping[education]\n",
    "living_with = living_with_mapping[living_with]\n",
    "is_parent = is_parent_mapping[is_parent]\n",
    "\n",
    "# Make a prediction\n",
    "if st.button('Predict'):\n",
    "    features = np.array([[education, income, kidhome, teenhome, recency, wines, fruits, meat, fish, sweets, gold,\n",
    "                          num_deals_purchases, num_web_purchases, num_catalog_purchases, num_store_purchases,\n",
    "                          num_web_visits_month, time_enrolled_days, age, spent, living_with, children, family_size,\n",
    "                          is_parent, total_promos, agglomerative_cluster]])\n",
    "    prediction = model.predict(features)\n",
    "    prediction_proba = model.predict_proba(features)\n",
    "    \n",
    "    st.write(f'Prediction: {\"Positive\" if prediction[0] else \"Negative\"}')\n",
    "    st.write(f'Prediction Probability: {prediction_proba[0]}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}