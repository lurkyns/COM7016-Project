import streamlit as st
import pandas as pd
import joblib
import os

st.write("Streamlit app script started successfully!")

# Check working directory
st.write("Current working directory:", os.getcwd())

# Corrected CSV path
csv_path = "data/cleaned/food_labelled.csv"
st.write("Checking CSV path:", csv_path)
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.write(" CSV loaded successfully!")
    st.write("Columns in DataFrame:", df.columns.tolist())
else:
    st.error(" CSV file NOT FOUND!")
    st.stop()

# Corrected model path
model_path = "models/decision_tree_model.pkl"
st.write("Checking model path:", model_path)
if os.path.exists(model_path):
    model = joblib.load(model_path)
    st.write(" Model loaded successfully!")
else:
    st.error("Model file NOT FOUND!")
    st.stop()

# Streamlit UI
st.title("Can I Eat This? - Diabetes Food Suitability Checker")

st.write("""Enter a food name or ingredient (e.g. "tomato", "pasta", "butter").
The app will check the nutrition data and tell you if it's suitable for diabetics.""")

# User input
food_name = st.text_input("Enter food name:")

if st.button("Check Suitability"):
    st.write("User typed:", food_name)
    matches = df[df['Description'].str.contains(food_name, case=False, na=False)]
    
    st.write(f"Number of matches found: {len(matches)}")
    
    if len(matches) == 0:
        st.warning("Oh no! Food not found in the database.")
    else:
        st.success(f"Found {len(matches)} matching food(s):")
        
        for idx, row in matches.iterrows():
            st.write(f"### {row['Description']}")
            st.write(f"- Sugar: {row['Data.Sugar Total']} g/100g")
            st.write(f"- Carbs: {row['Data.Carbohydrate']} g/100g")
            st.write(f"- Fibre: {row['Data.Fiber']} g/100g")
            st.write(f"- Energy: {row['Data.Kilocalories']} kcal/100g")
            st.write(f"- Saturated Fat: {row['Data.Fat.Saturated Fat']} g/100g")
            
            # Prepare input for model
            input_data = row[
                [
                    'Data.Sugar Total',
                    'Data.Carbohydrate',
                    'Data.Fiber',
                    'Data.Kilocalories',
                    'Data.Fat.Saturated Fat'
                ]
            ].values.reshape(1, -1)
            
            prediction = model.predict(input_data)[0]
            
            if prediction == 1:
                st.success(" Suitable for diabetics!")
            else:
                st.error(" Not suitable for diabetics.")
