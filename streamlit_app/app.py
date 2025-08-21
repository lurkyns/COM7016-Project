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

def classify_food_category(desc: str) -> str:
    desc = desc.lower()
    for cat, words in category_keywords.items():
        if any(w in desc for w in words):
            return cat
    return "Other"

# -------------------- PROFILE FORM --------------------
def profile_form():
    # Use an empty dict (not None) so .get() is always safe everywhere
    if "profile" not in st.session_state:
        st.session_state.profile = {}

    if not st.session_state.profile:
        st.markdown("## Set Your Dietary Profile")
        with st.form("profile_form"):
            diab_type = st.selectbox("Which best describes you?",
                                     ["Type 1", "Type 2", "Prediabetes", "Not sure"])
            goals = st.multiselect("Select your dietary goals:",
                                   ["Low sugar", "Low carbs", "Low calories", "High fibre"])
            allergies_raw = st.text_input("Any allergies? (comma‑separated)",
                                          placeholder="e.g. nuts, lactose")
            if st.form_submit_button("Save profile"):
                st.session_state.profile = {
                    "diabetes_type": diab_type,
                    "goals": goals,
                    "allergies": [a.strip().lower()
                                  for a in allergies_raw.split(",") if a.strip()]
                }
                st.success("Profile saved!")
                st.experimental_rerun()
    else:
        p = st.session_state.profile
        st.markdown(
            f"""
            <div style='background:#F1F8E9;padding:8px;border-radius:6px'>
              <b>Active profile</b><br>
              • Type : <b>{p.get('diabetes_type','')}</b><br>
              • Goals: <b>{', '.join(p.get('goals', [])) or 'None'}</b><br>
              • Allergies: <b>{', '.join(p.get('allergies', [])) or 'None'}</b>
            </div>""",
            unsafe_allow_html=True,
        )
        if st.button("Update profile"):
            st.session_state.profile = {}
            st.experimental_rerun()

# -------------------- EXPLANATION & UTILS --------------
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

    profile = st.session_state.get("profile") or {}
    goals = profile.get("goals", [])
    if "Low sugar" in goals and row["Data.Sugar Total"] > 5:
        reasons.append("exceeds your low‑sugar goal")
    if "Low carbs" in goals and row["Data.Carbohydrate"] > 30:
        reasons.append("exceeds your low‑carb goal")
    if "Low calories" in goals and row["Data.Kilocalories"] > 120:
        reasons.append("exceeds your low‑calorie goal")
    if "High fibre" in goals and row["Data.Fiber"] < 3:
        reasons.append("below your fibre target")

    return "This food has " + ", ".join(reasons) + "." if reasons else \
           "This food has a balanced nutrition profile."

def generate_dietary_feedback(row, pred):
    if pred == 1:
        return (f"✅ '{row['Description']}' looks okay: only "
                f"{row['Data.Sugar Total']:.1f} g sugar & "
                f"{row['Data.Carbohydrate']:.1f} g carbs.")
    issues = []
    if row["Data.Sugar Total"] > 10:
        issues.append("lots of sugar")
    if row["Data.Carbohydrate"] > 50:
        issues.append("too many carbs")
    if row["Data.Fiber"] < 2:
        issues.append("very little fibre")
    return f"⚠️ Not ideal because of {', '.join(issues)}."

def generate_diabetes_warning(row, dtype):
    if dtype == "Type 1" and row["Data.Sugar Total"] > 10:
        return "May require insulin adjustment."
    if dtype == "Type 2" and (row["Data.Sugar Total"] > 10 or row["Data.Carbohydrate"] > 50):
        return "Could spike blood sugar—watch your portion."
    if dtype == "Prediabetes" and (row["Data.Sugar Total"] > 8 or row["Data.Carbohydrate"] > 40):
        return "High sugar/carbs for pre‑diabetes."
    return ""

def get_contextual_recommendations(desc):
    notes = []
    text = desc.lower()
    if "fried" in text:
        notes.append("Consider grilled/baked instead of fried.")
    if "syrup" in text or "sugar" in text:
        notes.append("Look for unsweetened versions.")
    if "white bread" in text or "refined" in text:
        notes.append("Whole‑grain might keep glucose steadier.")
    return notes

def suggest_alternatives(bad_row, k=3):
    query_vec = vectoriser.transform([bad_row["Description"]])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    temp = df.copy()
    temp["sim"] = sims
    temp = temp[temp["Description"] != bad_row["Description"]]

    feats = temp[[
        "Data.Sugar Total", "Data.Carbohydrate",
        "Data.Fiber", "Data.Kilocalories", "Data.Fat.Saturated Fat"
    ]]
    temp["Predicted"] = model.predict(feats)
    temp = temp[temp["Predicted"] == 1]
    temp = temp.sort_values(["sim", "Data.Sugar Total"], ascending=[False, True])
    return temp.head(k)["Description"].tolist()

# -------------------- UI -------------------------------
st.title("Can I Eat This? – Diabetes Food Suitability Checker")
profile_form()

st.write("Enter a food (e.g. **tomato**, **pasta**) to check suitability.")
food_name = st.text_input("Enter food name:")

st.markdown("### Optional filters")
low_sugar = st.checkbox("Low Sugar (≤ 5 g)")
low_carbs = st.checkbox("Low Carbs (≤ 30 g)")
high_fibre = st.checkbox("High Fibre (≥ 3 g)")
low_kcal = st.checkbox("Low Calories (≤ 100 kcal)")

if st.button("Check Suitability"):
    # Initial match
    matches = df[df["Description"].str.contains(food_name, case=False, na=False)]

    # Allergy filter
    profile = st.session_state.get("profile") or {}
    for allerg in profile.get("allergies", []):
        matches = matches[~matches["Description"].str.contains(allerg, case=False, na=False)]

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
        st.warning("Food not found or excluded by filters. Try another item.")
        st.stop()

    # ---------- Results loop ----------
    for _, row in matches.iterrows():
        st.markdown(f"### {row['Description'].title()}")
        st.markdown(f"**Category:** {classify_food_category(row['Description'])}")

        # Nutrition table
        st.markdown(
            f"""
            - **Sugar:** {row['Data.Sugar Total']:.1f} g  
            - **Carbs:** {row['Data.Carbohydrate']:.1f} g  
            - **Fibre:** {row['Data.Fiber']:.1f} g  
            - **Energy:** {row['Data.Kilocalories']:.0f} kcal  
            - **Sat. fat:** {row['Data.Fat.Saturated Fat']:.1f} g"""
        )

        # Prediction
        x = row[[
            "Data.Sugar Total", "Data.Carbohydrate", "Data.Fiber",
            "Data.Kilocalories", "Data.Fat.Saturated Fat"
        ]].values.reshape(1, -1)
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

        # Chart
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
        ax.bar(nutrients.keys(), limits.values(), color="lightgrey", label="Recommended max")
        ax.bar(nutrients.keys(), nutrients.values(), color="skyblue", label="This food")
        ax.set_ylabel("g or kcal /100 g")
        ax.set_title("Nutrition vs Guidelines")
        ax.legend()
        st.pyplot(fig)

        # Contextual notes
        notes = get_contextual_recommendations(row["Description"])
        if notes:
            st.markdown("**Contextual Recommendations:**")
            for n in notes:
                st.write(f"- {n}")

        # Warning
        dtype = profile.get("diabetes_type", "")
        warn = generate_diabetes_warning(row, dtype)
        if warn:
            st.warning(warn)