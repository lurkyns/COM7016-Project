import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/cleaned/food_labelled.csv")

# Features and label
X = df[[
    "Data.Sugar Total", "Data.Carbohydrate",
    "Data.Fiber", "Data.Kilocalories",
    "Data.Fat.Saturated Fat"
]]
y = df["Diabetic_Suitability"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load model
model = joblib.load("models/decision_tree_model.pkl")

# Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Not Suitable', 'Suitable']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
disp.plot(ax=ax, cmap='Blues')
plt.title("Confusion Matrix - Decision Tree")
plt.show()
