import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data and Model
df = pd.read_csv("data/cleaned/food_labelled.csv")
model = joblib.load("models/decision_tree_model.pkl")

# Fit TF-IDF vectoriser for suggestions
vectoriser = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectoriser.fit_transform(df["Description"])

# Helper: AI-style nutrition explanation
def get_nutrition_explanation(row):
    reasons = []
    if row["Data.Sugar Total"] > 10:
        reasons.append(f"high sugar ({row['Data.Sugar Total']} g)")
    if row["Data.Carbohydrate"] > 50:
        reasons.append(f"high carbs ({row['Data.Carbohydrate']} g)")
    if row["Data.Fat.Saturated Fat"] > 5:
        reasons.append(f"high saturated fat ({row['Data.Fat.Saturated Fat']} g)")
    if row["Data.Fiber"] < 2:
        reasons.append(f"low fibre ({row['Data.Fiber']} g)")
    return "This food has " + ", ".join(reasons) + "." if reasons else \
           "This food has a balanced nutrition profile."

# Helper: suggest alternatives if prediction is "No"
def suggest_alternatives(bad_row, max_suggestions=3):
    query_vec = vectoriser.transform([bad_row["Description"]])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    temp = df.copy()
    temp["sim"] = sims
    temp = temp[temp["Description"] != bad_row["Description"]]
    
    X_full = temp[[
        "Data.Sugar Total", "Data.Carbohydrate",
        "Data.Fiber", "Data.Kilocalories",
        "Data.Fat.Saturated Fat"
    ]]
    preds = model.predict(X_full.values)
    temp["Predicted"] = preds
    temp = temp[temp["Predicted"] == 1]  # only “Yes”
    
    temp = temp.sort_values(["sim", "Data.Sugar Total"], ascending=[False, True])
    return temp.head(max_suggestions)["Description"].tolist()

# STREAMLIT UI
st.title("Can I Eat This? – Diabetes Food Suitability Checker")

st.write("Enter a food name (e.g. *tomato*, *pasta*, *butter*) to see if it’s suitable for diabetics.")

food_name = st.text_input("Enter food name:")

#  Optional smart filters
st.markdown("### Optional filters")
low_sugar = st.checkbox("Low Sugar (≤ 5g)")
low_carbs = st.checkbox("Low Carbohydrate (≤ 30g)")
high_fibre = st.checkbox("High Fibre (≥ 3g)")
low_kcal = st.checkbox("Low Calories (≤ 100 kcal)")

if st.button("Check Suitability"):
    st.write("User typed:", food_name)
    matches = df[df["Description"].str.contains(food_name, case=False, na=False)]

    # Apply filters
    if low_sugar:
        matches = matches[matches["Data.Sugar Total"] <= 5]
    if low_carbs:
        matches = matches[matches["Data.Carbohydrate"] <= 30]
    if high_fibre:
        matches = matches[matches["Data.Fiber"] >= 3]
    if low_kcal:
        matches = matches[matches["Data.Kilocalories"] <= 100]

    st.write(f"Matches found: **{len(matches)}**")

    if matches.empty:
        st.warning("Food not found in the database.")
        st.write("**Help us improve!** Enter your details so we can add this food later.")
        with st.form("user_info"):
            first = st.text_input("First Name")
            last = st.text_input("Last Name")
            mail = st.text_input("Email Address")
            if st.form_submit_button("Submit"):
                if mail.strip() and food_name.strip():
                    save_unknown_food(food_name, first, last, mail)
                    st.success("Thanks! We’ve saved your suggestion.")
                else:
                    st.error("Email and food name are required.")
        st.stop()

    for _, row in matches.iterrows():
        st.markdown(f"### {row['Description'].title()}")

        # nutrition panel
        st.markdown(
            f"""
            - **Sugar:** {row['Data.Sugar Total']:.1f} g  
            - **Carbs:** {row['Data.Carbohydrate']:.1f} g  
            - **Fibre:** {row['Data.Fiber']:.1f} g  
            - **Energy:** {row['Data.Kilocalories']:.0f} kcal  
            - **Sat. fat:** {row['Data.Fat.Saturated Fat']:.1f} g  
            """
        )

        # prediction
        features = row[[
            "Data.Sugar Total",
            "Data.Carbohydrate",
            "Data.Fiber",
            "Data.Kilocalories",
            "Data.Fat.Saturated Fat"
        ]].values.reshape(1, -1)

        pred = model.predict(features)[0]
        explanation = get_nutrition_explanation(row)

        if pred == 1:
            st.success(" Suitable for diabetics.")
            st.info(explanation)
        else:
            st.error(" Not suitable for diabetics.")
            st.info(explanation)

            alts = suggest_alternatives(row)
            if alts:
                st.warning("**Try these healthier alternatives:**")
                for alt in alts:
                    st.markdown(f"- {alt.title()}")
            else:
                st.warning("No close healthier alternatives found.")
