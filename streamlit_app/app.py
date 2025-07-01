import streamlit as st
import pandas as pd
import joblib
from utils import save_unknown_food

# Load Data 
df = pd.read_csv("data/cleaned/food_labelled.csv")

# Load model
model = joblib.load("models/decision_tree_model.pkl")

# App title
st.title("Can I Eat This? - Diabetes Food Suitability Checker")

st.write("""
Enter a food name or ingredient (e.g. "tomato", "pasta", "butter").
The app will check the nutrition data and tell you if it's suitable for diabetics.
""")

# User input
food_name = st.text_input("Enter food name:")

if st.button("Check Suitability"):
    st.write("User typed:", food_name)
    
    # Search for matches
    matches = df[df['Description'].str.contains(food_name, case=False, na=False)]
    
    st.write(f"Number of matches found: {len(matches)}")
    
    # If no matches found
    if len(matches) == 0:
        st.warning("Oh no! Food not found in the database.")
        
        st.write("**Help us improve!** Please enter your details so we can let you know if we add this food.")
        
        with st.form("user_info_form"):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            email = st.text_input("Email Address")
            
            submitted = st.form_submit_button("Submit")
            
            if submitted:
                if email.strip() == "" or food_name.strip() == "":
                    st.error("Please enter both a food name and your email so we can contact you.")
        else:
            save_unknown_food(food_name, first_name, last_name, email)
            st.success(f"Saved '{food_name}' with email {email}.")
    
    # If matches found 
    else:
        for idx, row in matches.iterrows():
            st.markdown(f"### {row['Description'].title()}")
            
            st.markdown(
                f"""
                - **Sugar:** {round(row['Data.Sugar Total'], 1)} g/100g
                - **Carbs:** {round(row['Data.Carbohydrate'], 1)} g/100g
                - **Fibre:** {round(row['Data.Fiber'], 1)} g/100g
                - **Energy:** {round(row['Data.Kilocalories'], 0)} kcal/100g
                - **Saturated Fat:** {round(row['Data.Fat.Saturated Fat'], 1)} g/100g
                """
            )
            
            # Prepare the input for the model
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
                st.success("Suitable for diabetics!")
            else:
                st.error("Not suitable for diabetics.")
