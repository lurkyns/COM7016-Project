
# COM7016 MSc Project: Can I Eat This? – Diabetes Food Suitability Checker

This health-focused web application uses machine learning to assess whether food items are suitable for individuals with diabetes. The project blends applied AI with public health insight to support better dietary decisions. The app is interactive, user-personalised, and built using Python and Streamlit.

## Project Summary

**Can I Eat This?** is a health informatics tool that predicts the diabetic suitability of food based on nutritional data. It uses supervised machine learning and a custom user profile (diabetes type, dietary goals, allergies) to give tailored results, dietary feedback, meal-time scoring, and suggestions.

Diabetes affects millions worldwide, and dietary choices play a crucial role in managing the condition. This web app leverages machine learning to help users understand if a food item is suitable for them based on nutritional data.

**Disclaimer:** This tool is for educational purposes only and should not replace medical advice. Always consult a healthcare professional for dietary decisions.

## Aims and Objectives

### Aim:
To develop a user-friendly machine learning-powered application that helps diabetic individuals make informed dietary decisions.

### Objectives:
- Curate and label a food nutrition dataset for diabetic suitability.
- Explore and compare machine learning models (Decision Tree, Logistic Regression, Naive Bayes).
- Evaluate models using accuracy, precision, recall, and F1 score.
- Deploy the best-performing model in a Streamlit application.
- Implement user profiling (diabetes type, dietary goals, allergies).
- Provide real-time food suitability feedback and healthy alternatives.
- Visualise nutritional breakdown and meal suitability scores.
- Ensure usability and transparency for non-technical users.

## Known Limitations

- Dataset is limited to 100g servings; portion-based variation is not considered.
- Model relies on pre-labelled data — unseen food types may be misclassified.
- No image or barcode support — future versions could explore these.


## Model Evaluation Summary

Models were trained using a subset of nutritional features (`Sugar`, `Carbohydrate`, `Fibre`, `Calories`, `Saturated Fat`) and evaluated via an 80/20 train-test split.

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Decision Tree        | 1.000    | 1.000     | 1.000  | 1.000    |
| Logistic Regression  | 0.929    | 0.925     | 0.960  | 0.942    |
| Naive Bayes          | 0.954    | 0.963     | 0.927  | 0.945    |

 The Decision Tree was chosen for deployment due to its simplicity and perfect fit for the dataset, but the app can support other models with minimal adjustments. Because non-technical users, including dietitians or diabetic patients, can easily understand the model's judgements, its usage in healthcare-related systems encourages openness. This is in line with human-centered computing concepts, which emphasise the importance of trust and clarity. Despite producing comparably high accuracy, Logistic Regression is less suited to capturing the intricate, frequently nonlinear patterns present in food classification due to its linear character.

## How to Run the App

### 1. Clone this repository or download the files

https://github.com/lurkyns/COM7016-Project.git

### 2. Install the required libraries

- streamlit
- pandas
- scikit-learn
- matplotlib
- joblib
- seaborn

Install them using the following command 

- pip install streamlit pandas scikit-learn matplotlib joblib seaborn

## Project Structure
Below is the main project structure 

├── app.py
├── data/
│   └── cleaned/
│       └── food_labelled.csv
├── models/
│   └── decision_tree_model.pkl
├── README.md

## Run the App
To launch the Streamlit app in your browser:

- streamlit run app.py

The app will automatically open in your default browser at http://localhost:8501.

## Author
Osazeme Orobosa Lurkyns Arthur