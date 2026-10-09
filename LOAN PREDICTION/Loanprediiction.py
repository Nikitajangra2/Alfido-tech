# ============================================================
# LOAN APPROVAL PREDICTION
# ============================================================

# ------------------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

# Used for 5-fold cross-validation
from sklearn.model_selection import cross_val_score


# ------------------------------------------------------------
# 2. LOAD THE DATASET
# ------------------------------------------------------------

# Read the loan dataset
data = pd.read_csv("loan_prediction.csv")

# Display the first 5 rows
print(data.head())


# ------------------------------------------------------------
# 3. UNDERSTAND THE DATA
# ------------------------------------------------------------

# Display information about columns, data types
# and number of non-empty values
print(data.info())

# Count missing values in every column
print("\nMissing values in each column:")
print(data.isnull().sum())


# ------------------------------------------------------------
# 4. CHECK MISSING VALUES
# ------------------------------------------------------------

# Display the number of applicants according to gender
print("\nApplicants by Gender:")
print(data["Gender"].value_counts())

# Display the number of applicants according to marital status
print("\nApplicants by Marital Status:")
print(data["Married"].value_counts())

# Display the number of applicants according to dependents
print("\nApplicants by Dependents:")
print(data["Dependents"].value_counts())

# Display the number of applicants according to employment status
print("\nApplicants by Self Employment:")
print(data["Self_Employed"].value_counts())

# Display the number of applicants according to loan term
print("\nApplicants by Loan Amount Term:")
print(data["Loan_Amount_Term"].value_counts())

# Display the number of applicants according to credit history
print("\nApplicants by Credit History:")
print(data["Credit_History"].value_counts())


# ------------------------------------------------------------
# 5. FILL MISSING VALUES
# ------------------------------------------------------------

# Make a copy of the original dataset
loan_data = data.copy()

# For categorical columns, use the most common value
loan_data["Gender"] = loan_data["Gender"].fillna(
    loan_data["Gender"].value_counts().idxmax()
)

loan_data["Married"] = loan_data["Married"].fillna(
    loan_data["Married"].value_counts().idxmax()
)

loan_data["Dependents"] = loan_data["Dependents"].fillna(
    loan_data["Dependents"].value_counts().idxmax()
)

loan_data["Self_Employed"] = loan_data["Self_Employed"].fillna(
    loan_data["Self_Employed"].value_counts().idxmax()
)

# For Loan Amount, use the average loan amount
loan_data["LoanAmount"] = loan_data["LoanAmount"].fillna(
    loan_data["LoanAmount"].mean()
)

# For Loan Amount Term, use the most common value
loan_data["Loan_Amount_Term"] = loan_data["Loan_Amount_Term"].fillna(
    loan_data["Loan_Amount_Term"].value_counts().idxmax()
)

# For Credit History, use the most common value
loan_data["Credit_History"] = loan_data["Credit_History"].fillna(
    loan_data["Credit_History"].value_counts().idxmax()
)

# Check whether missing values are still present
print("\nMissing values after cleaning:")
print(loan_data.isnull().sum())


# ------------------------------------------------------------
# 6. CONVERT TEXT VALUES INTO NUMBERS
# ------------------------------------------------------------

# Gender:
# Female = 0
# Male   = 1
loan_data["Gender"] = loan_data["Gender"].replace({
    "Female": 0,
    "Male": 1
})

# Married:
# No  = 0
# Yes = 1
loan_data["Married"] = loan_data["Married"].replace({
    "No": 0,
    "Yes": 1
})

# Dependents:
# 0  = 0
# 1  = 1
# 2  = 2
# 3+ = 3
loan_data["Dependents"] = loan_data["Dependents"].replace({
    "0": 0,
    "1": 1,
    "2": 2,
    "3+": 3
})

# Education:
# Not Graduate = 0
# Graduate     = 1
loan_data["Education"] = loan_data["Education"].replace({
    "Not Graduate": 0,
    "Graduate": 1
})

# Self Employed:
# No  = 0
# Yes = 1
loan_data["Self_Employed"] = loan_data["Self_Employed"].replace({
    "No": 0,
    "Yes": 1
})

# Property Area:
# Semiurban = 0
# Urban     = 1
# Rural     = 2
loan_data["Property_Area"] = loan_data["Property_Area"].replace({
    "Semiurban": 0,
    "Urban": 1,
    "Rural": 2
})


# ------------------------------------------------------------
# 7. SEPARATE INPUT FEATURES AND TARGET
# ------------------------------------------------------------

# X contains the information used to make predictions.
# Loan_ID is removed because it is only an identification number.
X = loan_data.iloc[:, 1:12]

# Y contains the final answer:
# Loan_Status
Y = loan_data.iloc[:, 12]


# ------------------------------------------------------------
# 8. CREATE A LIST TO STORE MODEL RESULTS
# ------------------------------------------------------------

model_names = [
    "Gradient Boosting",
    "Random Forest",
    "Decision Tree",
    "K-Nearest Neighbor",
    "Linear SVM"
]

model_scores = []


# ------------------------------------------------------------
# 9. GRADIENT BOOSTING CLASSIFIER
# ------------------------------------------------------------

gradient_boosting = GradientBoostingClassifier()

# Perform 5-fold cross-validation
scores = cross_val_score(
    gradient_boosting,
    X,
    Y,
    cv=5
)

# Calculate average accuracy
gradient_boosting_accuracy = scores.mean()

# Store the result
model_scores.append(gradient_boosting_accuracy)

print(
    "Gradient Boosting Accuracy: %.2f%%"
    % (gradient_boosting_accuracy * 100)
)


# ------------------------------------------------------------
# 10. RANDOM FOREST CLASSIFIER
# ------------------------------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=10
)

scores = cross_val_score(
    random_forest,
    X,
    Y,
    cv=5
)

random_forest_accuracy = scores.mean()

model_scores.append(random_forest_accuracy)

print(
    "Random Forest Accuracy: %.2f%%"
    % (random_forest_accuracy * 100)
)


# ------------------------------------------------------------
# 11. DECISION TREE CLASSIFIER
# ------------------------------------------------------------

decision_tree = DecisionTreeClassifier()

scores = cross_val_score(
    decision_tree,
    X,
    Y,
    cv=5
)

decision_tree_accuracy = scores.mean()

model_scores.append(decision_tree_accuracy)

print(
    "Decision Tree Accuracy: %.2f%%"
    % (decision_tree_accuracy * 100)
)


# ------------------------------------------------------------
# 12. K-NEAREST NEIGHBOR CLASSIFIER
# ------------------------------------------------------------

knn = KNeighborsClassifier()

scores = cross_val_score(
    knn,
    X,
    Y,
    cv=5
)

knn_accuracy = scores.mean()

model_scores.append(knn_accuracy)

print(
    "K-Nearest Neighbor Accuracy: %.2f%%"
    % (knn_accuracy * 100)
)


# ------------------------------------------------------------
# 13. LINEAR SUPPORT VECTOR MACHINE
# ------------------------------------------------------------

linear_svm = svm.LinearSVC(max_iter=5000)

scores = cross_val_score(
    linear_svm,
    X,
    Y,
    cv=5
)

linear_svm_accuracy = scores.mean()

model_scores.append(linear_svm_accuracy)

print(
    "Linear SVM Accuracy: %.2f%%"
    % (linear_svm_accuracy * 100)
)


# ------------------------------------------------------------
# 14. DISPLAY ALL MODEL RESULTS
# ------------------------------------------------------------

print("\nModel Performance:")
print("--------------------------------")

for name, score in zip(model_names, model_scores):
    print(f"{name}: {score * 100:.2f}%")


# ------------------------------------------------------------
# 15. DRAW A BAR CHART
# ------------------------------------------------------------

positions = np.arange(len(model_names))

plt.barh(
    positions,
    model_scores,
    align="center",
    alpha=0.5
)

plt.yticks(positions, model_names)

plt.xlabel("Score")
plt.title("Classification Performance")

plt.show()