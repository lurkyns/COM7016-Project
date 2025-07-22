import streamlit as st
import pandas as pd
import joblib

# Load data and model
df = pd.read_csv("data/cleaned/food_labelled.csv")
model = joblib.load("models/decision_tree_model.pkl")

def search_food():
    food_name = st.text_input("Enter food name:")

    if st.button("Check Suitability"):
        st.session_state.food_name = food_name

        # Find exact match first
        exact_match = df[df['Description'].str.lower() == food_name.lower()]
        partial_matches = df[df['Description'].str.contains(food_name, case=False, na=False) & ~df['Description'].str.lower().eq(food_name.lower())]

        # Combine results: exact match first
        matches = pd.concat([exact_match, partial_matches])

        if matches.empty:
            st.session_state.page = "form"
        else:
            X = matches.drop(columns=['Description', 'Diabetic_Suitability'])
            prediction = model.predict(X)

            # Add predictions and convert 0/1 to Yes/No
            matches['Predicted_Suitability'] = prediction
            matches['Predicted_Suitability'] = matches['Predicted_Suitability'].map({1: "Yes", 0: "No"})

            st.write(matches[['Description', 'Predicted_Suitability']])
