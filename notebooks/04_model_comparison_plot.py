import matplotlib.pyplot as plt

# Evaluation results
results = {
    "Decision Tree":    {"Accuracy": 1.000, "Precision": 1.000, "Recall": 1.000, "F1": 1.000},
    "Logistic Regression": {"Accuracy": 0.929, "Precision": 0.925, "Recall": 0.960, "F1": 0.942},
    "Naive Bayes":      {"Accuracy": 0.954, "Precision": 0.963, "Recall": 0.927, "F1": 0.945},
}

metrics = ["Accuracy", "Precision", "Recall", "F1"]
models = list(results.keys())

# Create grouped bar chart
x = range(len(metrics))
width = 0.25

fig, ax = plt.subplots()
for i, model in enumerate(models):
    scores = [results[model][metric] for metric in metrics]
    ax.bar([m + i * width for m in x], scores, width=width, label=model)

ax.set_xticks([m + width for m in x])
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Model Comparison by Metric")
ax.legend()
plt.tight_layout()
plt.show()