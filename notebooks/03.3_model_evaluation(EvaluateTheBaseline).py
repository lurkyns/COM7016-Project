import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier

# Load dataset
df = pd.read_csv("data/cleaned/food_labelled.csv")

# Features and target
X = df[[
    "Data.Sugar Total", "Data.Carbohydrate",
    "Data.Fiber", "Data.Kilocalories", "Data.Fat.Saturated Fat"
]]
y = df["Diabetic_Suitability"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create baseline model that always predicts the majority class
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
y_baseline = baseline.predict(X_test)

# Evaluate the baseline model
acc = accuracy_score(y_test, y_baseline)
prec = precision_score(y_test, y_baseline, zero_division=0)
rec = recall_score(y_test, y_baseline, zero_division=0)
f1 = f1_score(y_test, y_baseline, zero_division=0)

print("\nBaseline (Most Frequent Class) Model Evaluation:")
print(f"• Accuracy : {acc:.3f}")
print(f"• Precision: {prec:.3f}")
print(f"• Recall   : {rec:.3f}")
print(f"• F1 Score : {f1:.3f}")