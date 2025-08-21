import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# -------------------- DATA & MODEL --------------------
df = pd.read_csv("data/cleaned/food_labelled.csv")
model = joblib.load("models/decision_tree_model.pkl")

vectoriser = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectoriser.fit_transform(df["Description"])

# -------------------- CATEGORY HELPER -----------------
category_keywords = {
    "Vegetable": ["lettuce", "spinach", "kale", "carrot", "broccoli", "pepper", "tomato", "onion", "cabbage", "bean"],
    "Meat": ["chicken", "beef", "pork", "bacon", "sausage", "lamb", "turkey", "ham", "meat"],
    "Grain": ["rice", "pasta", "bread", "cereal", "quinoa", "oat", "flour", "noodle", "grain"],
    "Fruit": ["apple", "banana", "grape", "orange", "berry", "mango", "melon", "fruit", "peach", "pear"],
    "Dairy": ["milk", "cheese", "yogurt", "butter", "cream", "dairy"],
    "Snack": ["chocolate", "biscuit", "cookie", "crisp", "cake", "candy", "snack", "chips", "cracker"],
}

def classify_food_category(desc: str) -> str:
    text = desc.lower()
    for cat, words in category_keywords.items():
        if any(w in text for w in words):
            return cat
    return "Other"

# -------------------- PROFILE HELPERS -----------------
def get_profile() -> dict:
    """Safe accessor for the profile object."""
    prof = st.session_state.get("profile", None)
    return prof or {}   # if None, return {}

def profile_form():
    if "profile" not in st.session_state:
        st.session_state.profile = None

    if st.session_state.profile is None:
        st.markdown("## Set Your Dietary Profile")
        with st.form("profile_form"):
            diab_type = st.selectbox("Which best describes you?", ["Type 1", "Type 2", "Prediabetes", "Not sure"])
            goals = st.multiselect("Select your dietary goals:", ["Low sugar", "Low carbs", "Low calories", "High fibre"])
            allergies_raw = st.text_input("Any allergies? (comma‑separated)", placeholder="e.g. nuts, lactose")
            if st.form_submit_button("Save profile"):
                st.session_state.profile = {
                    "diabetes_type": diab_type,
                    "goals": goals,
                    "allergies": [a.strip().lower() for a in allergies_raw.split(",") if a.strip()],
                }
                st.success("Profile saved!")
                st.experimental_rerun()
    else:
        p = get_profile()
        st.markdown(
            f"""
            <div style='background:#F1F8E9;padding:8px;border-radius:6px'>
              <b>Active profile</b><br>
              • Type: <b>{p.get('diabetes_type','')}</b><br>
              • Goals: <b>{', '.join(p.get('goals',[])) or 'None'}</b><br>
              • Allergies: <b>{', '.join(p.get('allergies',[])) or 'None'}</b>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Update profile"):
            st.session_state.profile = None
            st.experimental_rerun()

# -------------------- EXPLANATION & UTILS --------------
def get_nutrition_explanation(row):
    reasons = []
    if row["Data.Sugar Total"] > 10:
        reasons.append(f"high sugar ({row['Data.Sugar Total']:.1f} g)")
    if row["Data.Carbohydrate"] > 50:
        reasons.append(f"high carbs ({row['Data.Carbohydrate']:.1f} g)")
    if row["Data.Fat.Saturated Fat"] > 5:
        reasons.append(f"high saturated fat ({row['Data.Fat.Saturated Fat']:.1f} g)")
    if row["Data.Fiber"] < 2:
        reasons.append(f"low fibre ({row['Data.Fiber']:.1f} g)")

    goals = get_profile().get("goals", [])
    if "Low sugar" in goals and row["Data.Sugar Total"] > 5:
        reasons.append("exceeds your low‑sugar goal")
    if "Low carbs" in goals and row["Data.Carbohydrate"] > 30:
        reasons.append("exceeds your low‑carb goal")
    if "Low calories" in goals and row["Data.Kilocalories"] > 120:
        reasons.append("exceeds your low‑calorie goal")
    if "High fibre" in goals and row["Data.Fiber"] < 3:
        reasons.append("below your fibre target")

    return "This food has " + ", ".join(reasons) + "." if reasons else "This food has a balanced nutrition profile."

def generate_dietary_feedback(row, pred):
    if pred == 1:
        return (f"✅ '{row['Description']}' looks okay: "
                f"{row['Data.Sugar Total']:.1f} g sugar • "
                f"{row['Data.Carbohydrate']:.1f} g carbs • "
                f"{row['Data.Fiber']:.1f} g fibre.")
    issues = []
    if row["Data.Sugar Total"] > 10: issues.append("high sugar")
    if row["Data.Carbohydrate"] > 50: issues.append("high carbs")
    if row["Data.Fiber"] < 2: issues.append("low fibre")
    if row["Data.Fat.Saturated Fat"] > 5: issues.append("high saturated fat")
    return "⚠️ Not ideal because of " + ", ".join(issues) + "."

def generate_diabetes_warning(row, dtype):
    if dtype == "Type 1" and row["Data.Sugar Total"] > 10:
        return "May require insulin adjustment."
    if dtype == "Type 2" and (row["Data.Sugar Total"] > 10 or row["Data.Carbohydrate"] > 50):
        return "Could spike blood sugar—watch your portion."
    if dtype == "Prediabetes" and (row["Data.Sugar Total"] > 8 or row["Data.Carbohydrate"] > 40):
        return "High sugar/carbs for pre‑diabetes."
    return ""

def get_contextual_recommendations(desc):
    notes, text = [], desc.lower()
    if "fried" in text: notes.append("Consider grilled/baked instead of fried.")
    if "syrup" in text or "sugar" in text: notes.append("Look for unsweetened versions.")
    if "white bread" in text or "refined" in text: notes.append("Whole‑grain can keep glucose steadier.")
    if "juice" in text: notes.append("Whole fruit is usually better than juice.")
    if "pastry" in text or "cake" in text: notes.append("Limit baked sweets; consider lower‑carb snacks.")
    return notes

def suggest_alternatives(bad_row, k=3):
    query_vec = vectoriser.transform([bad_row["Description"]])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    temp = df.copy()
    temp["sim"] = sims
    temp = temp[temp["Description"] != bad_row["Description"]]

    feats = temp[["Data.Sugar Total", "Data.Carbohydrate", "Data.Fiber", "Data.Kilocalories", "Data.Fat.Saturated Fat"]]
    temp["Predicted"] = model.predict(feats.values)
    temp = temp[temp["Predicted"] == 1]
    temp = temp.sort_values(["sim", "Data.Sugar Total"], ascending=[False, True])
    return temp.head(k)["Description"].tolist()

def meal_scores_for(row):
    scores = {"Breakfast": 0, "Lunch": 0, "Dinner": 0}
    sugar, carbs, fibre, kcal, sat = row["Data.Sugar Total"], row["Data.Carbohydrate"], row["Data.Fiber"], row["Data.Kilocalories"], row["Data.Fat.Saturated Fat"]
    # Breakfast
    if sugar <= 6: scores["Breakfast"] += 1
    if kcal <= 200: scores["Breakfast"] += 1
    if fibre >= 2: scores["Breakfast"] += 1
    if sat <= 3: scores["Breakfast"] += 1
    if carbs <= 30: scores["Breakfast"] += 1
    # Lunch
    if 20 <= carbs <= 50: scores["Lunch"] += 1
    if 250 <= kcal <= 500: scores["Lunch"] += 1
    if fibre >= 3: scores["Lunch"] += 1
    if sat <= 5: scores["Lunch"] += 1
    if sugar <= 10: scores["Lunch"] += 1
    # Dinner
    if carbs <= 40: scores["Dinner"] += 1
    if kcal <= 350: scores["Dinner"] += 1
    if fibre >= 3: scores["Dinner"] += 1
    if sugar <= 8: scores["Dinner"] += 1
    if sat <= 4: scores["Dinner"] += 1
    return scores

# -------------------- UI -------------------------------
st.title("Can I Eat This? – Diabetes Food Suitability Checker")
st.warning("⚠️ Educational tool only — not medical advice. Always consult a healthcare professional.")
profile_form()

st.write("Enter a food (e.g. **tomato**, **pasta**) to check suitability.")
food_name = st.text_input("Enter food name:")

st.markdown("### Optional filters")
low_sugar = st.checkbox("Low Sugar (≤ 5 g)")
low_carbs = st.checkbox("Low Carbs (≤ 30 g)")
high_fibre = st.checkbox("High Fibre (≥ 3 g)")
low_kcal = st.checkbox("Low Calories (≤ 100 kcal)")

if st.button("Check Suitability"):
    matches = df[df["Description"].str.contains(food_name, case=False, na=False)]

    # Allergy filter
    prof = get_profile()
    for allerg in prof.get("allergies", []):
        matches = matches[~matches["Description"].str.contains(allerg, case=False, na=False)]

    # Checkbox filters
    if low_sugar: matches = matches[matches["Data.Sugar Total"] <= 5]
    if low_carbs: matches = matches[matches["Data.Carbohydrate"] <= 30]
    if high_fibre: matches = matches[matches["Data.Fiber"] >= 3]
    if low_kcal: matches = matches[matches["Data.Kilocalories"] <= 100]

    st.write(f"Matches found: **{len(matches)}**")
    if matches.empty:
        st.warning("Food not found or excluded due to your profile. Try another item.")
        st.stop()

    for _, row in matches.iterrows():
        st.markdown(f"### {row['Description'].title()}")
        st.markdown(f"**Category:** {classify_food_category(row['Description'])}")

        # Nutrition facts
        st.markdown(
            f"""
            - **Sugar:** {row['Data.Sugar Total']:.1f} g  
            - **Carbs:** {row['Data.Carbohydrate']:.1f} g  
            - **Fibre:** {row['Data.Fiber']:.1f} g  
            - **Energy:** {row['Data.Kilocalories']:.0f} kcal  
            - **Sat. fat:** {row['Data.Fat.Saturated Fat']:.1f} g
            """
        )

        # Prediction
        x = row[["Data.Sugar Total", "Data.Carbohydrate", "Data.Fiber", "Data.Kilocalories", "Data.Fat.Saturated Fat"]].values.reshape(1, -1)
        pred = model.predict(x)[0]

        if pred == 1:
            st.success("Suitable for diabetics.")
        else:
            st.error("Not suitable for diabetics.")
            alts = suggest_alternatives(row)
            if alts:
                st.write("**Try these alternatives:**")
                for a in alts:
                    st.markdown(f"- {a.title()}")

        st.info(get_nutrition_explanation(row))
        st.markdown(f"**Smart Summary:** {generate_dietary_feedback(row, pred)}")

        # Meal-based scores
        st.markdown("#### 🍽️ Meal Suitability Scores")
        scores = meal_scores_for(row)
        for meal, score in scores.items():
            stars = "⭐" * score + "☆" * (5 - score)
            st.markdown(f"- **{meal}:** {stars} ({score}/5)")

        # Nutrient bar chart
        st.markdown("#### Nutrient Breakdown vs Thresholds")
        nutrients = {
            "Sugar": row["Data.Sugar Total"],
            "Carbs": row["Data.Carbohydrate"],
            "Fibre": row["Data.Fiber"],
            "Kcal": row["Data.Kilocalories"],
            "Sat Fat": row["Data.Fat.Saturated Fat"],
        }
        limits = {"Sugar": 10, "Carbs": 50, "Fibre": 3, "Kcal": 120, "Sat Fat": 5}

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(list(limits.keys()), list(limits.values()), color="lightgrey", label="Recommended max")
        ax.bar(list(nutrients.keys()), list(nutrients.values()), color="skyblue", label="This food")
        ax.set_ylabel("g or kcal per 100 g")
        ax.set_title("Nutrition vs Guidelines")
        ax.legend()
        st.pyplot(fig)

        # Contextual notes
        notes = get_contextual_recommendations(row["Description"])
        if notes:
            st.markdown("**Contextual Recommendations:**")
            for n in notes:
                st.write(f"- {n}")

        # Diabetes-type warning
        warn = generate_diabetes_warning(row, get_profile().get("diabetes_type", ""))
        if warn:
            st.warning(warn)
