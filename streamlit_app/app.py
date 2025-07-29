import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv("data/cleaned/food_labelled.csv")
model = joblib.load("models/decision_tree_model.pkl")

# TF-IDF vectoriser for alternative suggestions
vectoriser = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectoriser.fit_transform(df["Description"])

#  CATEGORY CLASSIFIER (rule‑based keywords)

category_keywords = {
    "Vegetable": ["lettuce", "spinach", "kale", "carrot", "broccoli", "pepper",
                  "tomato", "onion", "cabbage", "bean"],
    "Meat": ["chicken", "beef", "pork", "bacon", "sausage", "lamb", "turkey",
             "ham", "meat"],
    "Grain": ["rice", "pasta", "bread", "cereal", "quinoa", "oat", "flour",
              "noodle", "grain"],
    "Fruit": ["apple", "banana", "grape", "orange", "berry", "mango", "melon",
              "fruit", "peach", "pear"],
    "Dairy": ["milk", "cheese", "yogurt", "butter", "cream", "dairy"],
    "Snack": ["chocolate", "biscuit", "cookie", "crisp", "cake", "candy",
              "snack", "chips", "cracker"],
}

def classify_food_category(description: str) -> str:
    """Return a simple food category based on keyword matching."""
    text = description.lower()
    for cat, words in category_keywords.items():
        if any(w in text for w in words):
            return cat
    return "Other"

# USER PROFILE FORM
def profile_form():
    if "profile" not in st.session_state:
        st.session_state.profile = None

    if st.session_state.profile is None:
        st.markdown("## Set Your Dietary Profile")
        with st.form("profile_form"):
            diab_type = st.selectbox(
                "Which best describes you?",
                ["Type 1", "Type 2", "Prediabetes", "Not sure"]
            )
            goals = st.multiselect(
                "Select your dietary goals:",
                ["Low sugar", "Low carbs", "Low calories", "High fibre"]
            )
            allergies_raw = st.text_input(
                "Any allergies or ingredients to avoid? (comma-separated)",
                placeholder="e.g. nuts, lactose"
            )
            submitted = st.form_submit_button("Save profile")
            if submitted:
                st.session_state.profile = {
                    "diabetes_type": diab_type,
                    "goals": goals,
                    "allergies": [
                        a.strip().lower()
                        for a in allergies_raw.split(",") if a.strip()
                    ]
                }
                st.success("Profile saved!")
                st.experimental_rerun()
    else:
        p = st.session_state.profile
        st.markdown(
            f"""
            <div style='background:#F1F8E9;padding:10px;border-radius:8px'>
            <b>Active profile</b><br>
            • Diabetes type: <b>{p['diabetes_type']}</b><br>
            • Goals: <b>{', '.join(p['goals']) or 'None'}</b><br>
            • Allergies: <b>{', '.join(p['allergies']) or 'None'}</b>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Update profile"):
            st.session_state.profile = None
            st.experimental_rerun()

# EXPLANATION GENERATOR 

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

    user_goals = st.session_state.get("profile", {}).get("goals", [])
    extra_context = ""
    if "Low sugar" in user_goals and row["Data.Sugar Total"] > 5:
        extra_context += " You selected low sugar as a goal."
    if "Low carbs" in user_goals and row["Data.Carbohydrate"] > 30:
        extra_context += " You selected low carbs as a goal."
    if "Low calories" in user_goals and row["Data.Kilocalories"] > 120:
        extra_context += " You selected low calorie foods."
    if "High fibre" in user_goals and row["Data.Fiber"] < 3:
        extra_context += " You wanted high fibre."

    return ("This food has " + ", ".join(reasons) + "." if reasons else "This food has a balanced nutrition profile.") + extra_context

# AI FEATURE: SMART FEEDBACK SUMMARY
def generate_dietary_feedback(row, prediction):
    if prediction == 1:
        return (
            f"The food '{row['Description']}' is suitable for diabetics. "
            f"It contains {round(row['Data.Sugar Total'], 1)}g of sugar and {round(row['Data.Carbohydrate'], 1)}g of carbohydrates, "
            f"which are within manageable levels. Its {round(row['Data.Fiber'], 1)}g of fibre supports better blood sugar control."
        )
    else:
        issues = []
        if row["Data.Sugar Total"] > 10:
            issues.append("excess sugar")
        if row["Data.Carbohydrate"] > 50:
            issues.append("high carbohydrate content")
        if row["Data.Fat.Saturated Fat"] > 5:
            issues.append("high saturated fat")
        if row["Data.Fiber"] < 2:
            issues.append("low fibre")

        details = ", ".join(issues)
        return (
            f"'{row['Description']}' is not suitable for diabetics due to {details}. "
            f"Try choosing foods with better fibre, sugar, and carb balance."
        )

# AI FEATURE 3: Custom warning based on diabetes type

def generate_diabetes_warning(row, diabetes_type):
    sugar = row["Data.Sugar Total"]
    carbs = row["Data.Carbohydrate"]
    warning = ""

    if diabetes_type == "Type 1" and sugar > 10:
        warning = " This food may cause a blood sugar spike and may require insulin adjustment."
    elif diabetes_type == "Type 2" and (sugar > 10 or carbs > 50):
        warning = " This food has high sugar and carbohydrate content. Monitor portion size carefully."
    elif diabetes_type == "Prediabetes" and (sugar > 8 or carbs > 40):
        warning = " This food could raise your blood sugar levels significantly."

    return warning

# AI FEATURE 4: CONTEXTUAL RECOMMENDATIONS BASED ON FOOD TEXT
def get_contextual_recommendations(description):
    suggestions = []

    if "fried" in description.lower():
        suggestions.append("Consider grilled or baked options instead of fried.")
    if "syrup" in description.lower() or "sugar" in description.lower():
        suggestions.append("Try unsweetened or low-sugar alternatives.")
    if "white bread" in description.lower() or "refined" in description.lower():
        suggestions.append("Wholegrain options might be better for blood sugar control.")
    if "juice" in description.lower():
        suggestions.append("Whole fruits are generally better than fruit juices.")
    if "pastry" in description.lower() or "cake" in description.lower():
        suggestions.append("Limit baked sweets and consider lower-carb snacks.")

    return suggestions

# ALTERNATIVE SUGGESTIONS

def suggest_alternatives(bad_row, max_suggestions=3):
    query_vec = vectoriser.transform([bad_row["Description"]])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    temp = df.copy()
    temp["sim"] = sims
    temp = temp[temp["Description"] != bad_row["Description"]]

    X_full = temp[[
        "Data.Sugar Total", "Data.Carbohydrate",
        "Data.Fiber", "Data.Kilocalories", "Data.Fat.Saturated Fat"
    ]]
    preds = model.predict(X_full.values)
    temp["Predicted"] = preds
    temp = temp[temp["Predicted"] == 1]
    temp = temp.sort_values(["sim", "Data.Sugar Total"], ascending=[False, True])
    return temp.head(max_suggestions)["Description"].tolist()

# STREAMLIT MAIN APP (UI)
st.title("Can I Eat This? – Diabetes Food Suitability Checker")
profile_form()

st.warning("⚠ This tool is for educational purposes only and should not replace medical advice. Always consult a healthcare provider for personalised dietary guidance.")

st.write("Enter a food name (e.g. *tomato*, *pasta*, *butter*) to see if it’s suitable for diabetics.")
food_name = st.text_input("Enter food name:")

st.markdown("### Optional filters")
low_sugar = st.checkbox("Low Sugar (≤ 5g)")
low_carbs = st.checkbox("Low Carbohydrate (≤ 30g)")
high_fibre = st.checkbox("High Fibre (≥ 3g)")
low_kcal = st.checkbox("Low Calories (≤ 100 kcal)")

if st.button("Check Suitability"):
    st.write("User typed:", food_name)
    matches = df[df["Description"].str.contains(food_name, case=False, na=False)]

# Allergy filtering

    if "profile" in st.session_state and st.session_state.profile:
        for allergen in st.session_state.profile["allergies"]:
            matches = matches[~matches["Description"].str.contains(allergen, case=False, na=False)]

  # Checkbox filters
  
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
        st.warning("Food not found or excluded due to your profile.")
        st.stop()
        
        # Nutrient Comparison Chart
for _, row in matches.iterrows():
    st.markdown(f"### {row['Description'].title()}")
    category = classify_food_category(row["Description"])
    st.markdown(f"**Category:** {category}")

    # Nutrition panel
    st.markdown(
        f"""
        - **Sugar:** {row['Data.Sugar Total']:.1f} g  
        - **Carbs:** {row['Data.Carbohydrate']:.1f} g  
        - **Fibre:** {row['Data.Fiber']:.1f} g  
        - **Energy:** {row['Data.Kilocalories']:.0f} kcal  
        - **Sat. fat:** {row['Data.Fat.Saturated Fat']:.1f} g  
        """
    )

    # Prediction and explanations
    features = row[[
        "Data.Sugar Total", "Data.Carbohydrate",
        "Data.Fiber", "Data.Kilocalories",
        "Data.Fat.Saturated Fat"
    ]].values.reshape(1, -1)
    pred = model.predict(features)[0]
    explanation = get_nutrition_explanation(row)
    feedback = generate_dietary_feedback(row, pred)

    # Prediction display
    if pred == 1:
        st.success(" Suitable for diabetics.")
    else:
        st.error(" Not suitable for diabetics.")
        alts = suggest_alternatives(row)
        if alts:
            st.warning("Try these healthier alternatives:")
            for alt in alts:
                st.markdown(f"- {alt.title()}")
        else:
            st.info("No close healthier options found.")

    st.info(explanation)
    st.markdown(f"**Smart Summary:** {feedback}")

    # Visual: Nutrient comparison
    st.markdown("#### Nutrient Breakdown vs Thresholds")
    nutrients = {
        "Sugar": row["Data.Sugar Total"],
        "Carbs": row["Data.Carbohydrate"],
        "Fibre": row["Data.Fiber"],
        "Kcal": row["Data.Kilocalories"],
        "Sat Fat": row["Data.Fat.Saturated Fat"]
    }
    thresholds = {
        "Sugar": 10, "Carbs": 50, "Fibre": 3, "Kcal": 120, "Sat Fat": 5
    }

    labels = list(nutrients.keys())
    values = list(nutrients.values())
    max_vals = [thresholds[k] for k in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, max_vals, color='lightgrey', label="Recommended Max")
    ax.bar(labels, values, color='skyblue', label="This Food")
    ax.set_ylabel("g or kcal per 100g")
    ax.set_title("Nutrition vs Health Guidelines")
    ax.legend()
    st.pyplot(fig)

    # Get contextual suggestions based on food text
    extra_notes = get_contextual_recommendations(row["Description"])
    if extra_notes:
        st.markdown("**Contextual Recommendations:**")
        for note in extra_notes:
            st.write(f"- {note}")

    # Add warning if relevant
    diab_type = st.session_state.get("profile", {}).get("diabetes_type", "")
    warning = generate_diabetes_warning(row, diab_type)
    if warning:
        st.warning(warning)

    