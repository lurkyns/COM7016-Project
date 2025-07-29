import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load cleaned and labelled dataset
df = pd.read_csv("data/cleaned/food_labelled.csv")

# Features and target
X = df[[
    "Data.Sugar Total", "Data.Carbohydrate",
    "Data.Fiber", "Data.Kilocalories",
    "Data.Fat.Saturated Fat"
]]
y = df["Diabetic_Suitability"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load models with correct paths
models = {
    "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_model.pkl")
}

# Evaluate each model
for name, model in models.items():
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n {name} Model Evaluation:")
    print(f" - Accuracy:  {acc:.3f}")
    print(f" - Precision: {prec:.3f}")
    print(f" - Recall:    {rec:.3f}")
    print(f" - F1 Score:  {f1:.3f}")
