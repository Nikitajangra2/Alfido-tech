import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


# 1. Read the data

FILE = "zomato.csv"
df = pd.read_csv(FILE)
print(df.head())

# 2. Cleaning

def yes_no_to_number(x):
    if x == "Yes":
        return 1
    if x == "No":
        return 0
    return np.nan


for field in ("online_order", "book_table"):
    df[field] = df[field].map(yes_no_to_number)

df["rate"] = (
    df["rate"]
    .astype(str)
    .str.replace("/5", "", regex=False)
)
df["rate"] = pd.to_numeric(df["rate"], errors="coerce")

df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

cost_col = "approx_cost(for two people)"
df[cost_col] = (
    df[cost_col]
    .astype(str)
    .str.replace(",", "", regex=False)
)
df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")


# Keep only the ten most frequent values in the four categorical fields.
category_fields = [
    "location",
    "rest_type",
    "cuisines",
    "listed_in(type)"
]

for field in category_fields:
    frequent = df[field].value_counts().head(10).index
    df[field] = df[field].where(df[field].isin(frequent), "Other")


# Missing-value treatment
for field in ("online_order", "book_table"):
    df[field] = df[field].fillna(df[field].mode().iloc[0])

for field in ("rate", "votes", cost_col):
    df[field] = df[field].fillna(df[field].median())

for field in category_fields:
    df[field] = df[field].fillna("Unknown")


# Remove columns that are not used in the modelling.
remove_fields = ["phone", "dish_liked", "address", "name"]
df.drop(columns=remove_fields, inplace=True)

print("\nMissing values after cleaning:")
print(df.isna().sum())


# ------------------------------------------------------------
# 3. Basic EDA
# ------------------------------------------------------------

print("\nDataset summary:")
print(df.describe(include="all"))

print("\nCategorical values:")
for field in category_fields:
    print(f"\n{field}")
    print(df[field].unique())


numeric_fields = [
    "online_order",
    "book_table",
    "rate",
    "votes",
    cost_col
]

for field in numeric_fields:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[field], bins=30, kde=True,color="skyblue")
    plt.title("Distribution of " + field)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


for field in category_fields:
    plt.figure(figsize=(10, 6))
    order = df[field].value_counts().index
    sns.countplot(data=df, y=field, order=order)
    plt.title("Count of " + field)
    plt.show()


sns.pairplot(df[numeric_fields])
plt.show()


for field in ["rate", "votes", cost_col]:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="online_order", y=field)
    plt.title(field + " by Online Order")
    plt.show()


# ------------------------------------------------------------
# 4. Feature engineering
# ------------------------------------------------------------

median_cost = df[cost_col].median()

# 1 = above the median cost, 0 = median or below.
df["expensive"] = (df[cost_col] > median_cost).astype(int)

df.drop(columns=cost_col, inplace=True)

print("\nData after feature engineering:")
print(df.head())


# ------------------------------------------------------------
# 5. Prepare predictors
# ------------------------------------------------------------

target = df["expensive"]
features = df.drop(columns="expensive")

# The original workflow encoded the categorical variables before
# making the train/test split, so that order is retained here.
encoder = OneHotEncoder(drop="first")
encoded_features = encoder.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    encoded_features,
    target,
    test_size=0.20,
    random_state=42
)

print("\nTrain/test shapes:")
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


scaler = MaxAbsScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 6. Classification: expensive vs. not expensive
# ------------------------------------------------------------

classifiers = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000, random_state=42),

    "Decision Tree":
        DecisionTreeClassifier(max_depth=10, random_state=42),

    "Random Forest":
        RandomForestClassifier(random_state=42),

    "Gradient Boosting":
        GradientBoostingClassifier(random_state=42)
}

classification_results = []

for label, estimator in classifiers.items():
    estimator.fit(X_train_scaled, y_train)
    prediction = estimator.predict(X_test_scaled)

    classification_results.append({
        "Model": label,
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction),
        "Recall": recall_score(y_test, prediction),
        "F1 Score": f1_score(y_test, prediction)
    })

classification_table = pd.DataFrame(classification_results)
print("\nClassification results:")
print(classification_table)


# ------------------------------------------------------------
# 7. Regression section
# ------------------------------------------------------------
# This follows the target used by the supplied notebook. At this
# point y is still the "expensive" target, even though the section
# is described there as rating regression.

regressors = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression":
        RidgeCV(alphas=[0.1, 1.0, 10.0], cv=5),

    "Lasso Regression":
        LassoCV(alphas=[0.1, 1.0, 10.0], cv=5),

    "ElasticNet Regression":
        ElasticNetCV(alphas=[0.1, 1.0, 10.0], cv=5),

    "Random Forest":
        RandomForestRegressor(random_state=42),

    "Gradient Boosting":
        GradientBoostingRegressor(random_state=42)
}

regression_results = []

for label, estimator in regressors.items():
    estimator.fit(X_train_scaled, y_train)

    cv_scores = cross_val_score(
        estimator,
        X_test_scaled,
        y_test,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    rmse = -cv_scores.mean()

    regression_results.append({
        "Model": label,
        "RMSE": rmse
    })

regression_table = pd.DataFrame(regression_results)
print("\nRegression results:")
print(regression_table)


# ------------------------------------------------------------
# 8. K-Means clustering
# ------------------------------------------------------------

inertia_values = []
cluster_range = range(1, 11)

for number in cluster_range:
    km = KMeans(n_clusters=number, random_state=42)
    km.fit(X_train_scaled)
    inertia_values.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(cluster_range), inertia_values, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Distortion")
plt.title("Elbow Method For Optimal k")
plt.grid(True)
plt.show()


# The supplied analysis used four clusters.
km = KMeans(n_clusters=4, random_state=42)
km.fit(X_train_scaled)

train_clusters = km.labels_
test_clusters = km.predict(X_test_scaled)

all_clusters = np.concatenate([train_clusters, test_clusters])
df["cluster"] = all_clusters

print("\nClustered data:")
print(df.head())


# ------------------------------------------------------------
# 9. Describe each cluster
# ------------------------------------------------------------

cluster_numeric = [
    "online_order",
    "book_table",
    "rate",
    "votes",
    "expensive"
]

cluster_categorical = [
    "location",
    "rest_type",
    "cuisines",
    "listed_in(type)"
]

cluster_average = df.groupby("cluster")[cluster_numeric].mean()

cluster_common = df.groupby("cluster")[cluster_categorical].agg(
    lambda series: series.mode().iloc[0] if not series.mode().empty else np.nan
)

cluster_summary = pd.concat(
    [cluster_average, cluster_common],
    axis=1
)

print("\nCluster characteristics:")
print(cluster_summary)


# ------------------------------------------------------------
# 10. t-SNE visualization
# ------------------------------------------------------------

random.seed(42)

sample_size = min(2000, X_train_scaled.shape[0])
chosen_rows = random.sample(
    range(X_train_scaled.shape[0]),
    sample_size
)

sample_matrix = X_train_scaled[chosen_rows]
sample_labels = train_clusters[chosen_rows]

tsne_model = TSNE(
    n_components=2,
    init="random",
    random_state=42
)

embedded = tsne_model.fit_transform(sample_matrix.toarray())

tsne_data = pd.DataFrame(
    embedded,
    columns=["Dimension 1", "Dimension 2"]
)
tsne_data["Cluster"] = sample_labels

plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=tsne_data,
    x="Dimension 1",
    y="Dimension 2",
    hue="Cluster",
    palette="viridis"
)
plt.title("t-SNE visualization of clusters")
plt.show()


# ------------------------------------------------------------
# 11. Quick conclusions
# ------------------------------------------------------------

best_classifier = classification_table.loc[
    classification_table["Accuracy"].idxmax()
]

best_regressor = regression_table.loc[
    regression_table["RMSE"].idxmin()
]

print("\nBest classification model:")
print(best_classifier)

print("\nBest regression model:")
print(best_regressor)

print(
    "\nThe cluster summary can be used to compare the four groups. "
    "The supplied analysis found that the groups were quite similar "
    "and overlapped substantially in the t-SNE visualization."
)
