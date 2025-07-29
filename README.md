# COM7016 MSc Project - Can I Eat This? Machine Learning for Diabetic Diets 

This project uses machine learning to classify food items as suitable or not suitable for people with diabetes based on their nutritional data.

## Goals
- Use supervised ML models: Decision Tree, Logistic Regression, and Naive Bayes
- Build a user-friendly Streamlit app
- Evaluate with standard metrics (accuracy, precision, recall, F1-score)
- Focus on interpretability, ethics, and real-world usability

## Author
Osazeme Orobosa Lurkyns Arthur

## Model Evaluation Summary

Three machine learning models were evaluated using a train-test split and the following metrics:
- **Accuracy**: How often the model is correct
- **Precision**: How often it predicts a "suitable" food correctly
- **Recall**: How well it detects all suitable foods
- **F1 Score**: Balance between precision and recall

| Model                | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Decision Tree       | 1.000    | 1.000     | 1.000  | 1.000    |
| Logistic Regression | 0.929    | 0.925     | 0.960  | 0.942    |
| Naive Bayes         | 0.954    | 0.963     | 0.927  | 0.945    |

The Decision Tree is probably overfitting the data even though it received perfect ratings.  However, because non-technical users, including dietitians or diabetic patients, can easily understand the model's judgements, its usage in healthcare-related systems encourages openness.  This is in line with human-centered computing concepts, which emphasise the importance of trust and clarity.  Despite producing comparably high accuracy, Logistic Regression is less suited to capturing the intricate, frequently nonlinear patterns present in food classification due to its linear character.

Model Evaluation & Baseline Comparison
To assess the performance of the machine learning models predicting food suitability for diabetics, a baseline model and multiple classifiers were evaluated using a train/test split and key classification metrics.

Baseline Model
A Dummy Classifier was used as a baseline, always predicting the most frequent class in the training set. This helps determine whether the trained models provide value beyond naive guessing.

Metric	Baseline (Most Frequent)
Accuracy	0.604
Precision	0.604
Recall	1.000
F1 Score	0.753

The baseline performs poorly in terms of understanding the actual patterns in the data. It correctly guesses the majority class but fails to capture the minority class (unsuitable foods), which is critical in a health-related context.

Decision Tree Model (Selected)
The Decision Tree was chosen for its strong interpretability and superior performance across all metrics:

Metric	Decision Tree
Accuracy	1.000
Precision	1.000
Recall	1.000
F1 Score	1.000

It significantly outperforms the baseline, making it the most suitable model for predicting diabetic food suitability in this project.

Feature Importance
To support explainability, feature importance from the decision tree was also explored to highlight the most influential nutritional factors in the prediction process (e.g. sugar, carbohydrates, fibre, etc.).

