# ============================================================
# LOAN APPROVAL PREDICTION
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score

# ------------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------------
data = pd.read_csv("loan_prediction.csv")
print(data.head())
print(data.info())

print("\nMissing values in each column:")
print(data.isnull().sum())

# ------------------------------------------------------------
# EXPLORE CATEGORICAL VARIABLES
# ------------------------------------------------------------
for col in [
    "Gender", "Married", "Dependents", "Self_Employed",
    "Loan_Amount_Term", "Credit_History"
]:
    print(f"\nApplicants by {col}:")
    print(data[col].value_counts())

# ------------------------------------------------------------
# CLEAN MISSING VALUES
# ------------------------------------------------------------
loan_data = data.copy()

for col in ["Gender", "Married", "Dependents", "Self_Employed",
            "Loan_Amount_Term", "Credit_History"]:
    loan_data[col] = loan_data[col].fillna(
        loan_data[col].value_counts().idxmax()
    )

loan_data["LoanAmount"] = loan_data["LoanAmount"].fillna(
    loan_data["LoanAmount"].mean()
)

print("\nMissing values after cleaning:")
print(loan_data.isnull().sum())

# ------------------------------------------------------------
# ENCODE CATEGORICAL VALUES
# ------------------------------------------------------------
mappings = {
    "Gender": {"Female": 0, "Male": 1},
    "Married": {"No": 0, "Yes": 1},
    "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3},
    "Education": {"Not Graduate": 0, "Graduate": 1},
    "Self_Employed": {"No": 0, "Yes": 1},
    "Property_Area": {"Semiurban": 0, "Urban": 1, "Rural": 2},
}

for col, mapping in mappings.items():
    loan_data[col] = loan_data[col].replace(mapping)

# ------------------------------------------------------------
# FEATURES AND TARGET
# ------------------------------------------------------------
X = loan_data.iloc[:, 1:12]
Y = loan_data.iloc[:, 12]

# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------
model_names = [
    "Gradient Boosting",
    "Random Forest",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Linear SVM"
]

models = [
    GradientBoostingClassifier(),
    RandomForestClassifier(n_estimators=10, random_state=42),
    DecisionTreeClassifier(random_state=42),
    KNeighborsClassifier(),
    svm.LinearSVC(max_iter=5000)
]

model_scores = []

for name, model in zip(model_names, models):
    scores = cross_val_score(model, X, Y, cv=5)
    accuracy = scores.mean()
    model_scores.append(accuracy)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

print("\nModel Performance:")
print("--------------------------------")
for name, score in zip(model_names, model_scores):
    print(f"{name}: {score * 100:.2f}%")

# ------------------------------------------------------------
# MODEL PERFORMANCE CHART
# ------------------------------------------------------------
positions = np.arange(len(model_names))

plt.figure(figsize=(9, 5))
plt.barh(positions, model_scores, alpha=0.7)
plt.yticks(positions, model_names)
plt.xlabel("Mean 5-Fold Cross-Validation Accuracy")
plt.title("Loan Approval Classification Performance")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200, bbox_inches="tight")
plt.show()
