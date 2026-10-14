
# SMART CUSTOMER CLASSIFICATION PROJECT

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# 1. CREATE CUSTOMER DATASET

np.random.seed(42)

n = 2000

tenure = np.random.randint(1, 73, n)
monthly_charges = np.random.uniform(300, 2000, n)
support_calls = np.random.poisson(2, n)

contract_type = np.random.choice(
    ["Monthly", "Yearly"], n, p=[0.7, 0.3]
)

internet_service = np.random.choice(
    ["Fiber", "DSL"], n
)

late_payments = np.random.poisson(1, n)

risk_score = (
    1.2 * (contract_type == "Monthly")
    + 0.15 * support_calls
    + 0.35 * late_payments
    - 0.035 * tenure
    + 0.0004 * monthly_charges
    + np.random.normal(0, 0.8, n)
)

probability = 1 / (1 + np.exp(-risk_score + 1.5))

churn = np.random.binomial(1, probability)

df = pd.DataFrame({
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "internet_service": internet_service,
    "late_payments": late_payments,
    "churn": churn
})

print("\nCUSTOMER DATASET")
print(df.head())

print("\nChurn distribution:")
print(df["churn"].value_counts())


# 2. SEPARATE FEATURES AND TARGET

X = df.drop("churn", axis=1)
y = df["churn"]


# 3. SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. PREPROCESS DATA

numeric_features = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "late_payments"
]

categorical_features = [
    "contract_type",
    "internet_service"
]

preprocessor = ColumnTransformer([
    (
        "numeric",
        StandardScaler(),
        numeric_features
    ),
    (
        "categorical",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ),
        categorical_features
    )
])


# 5. CREATE FIVE CLASSIFICATION MODELS

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    ),

    "Naive Bayes": GaussianNB(),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    )
}


# 6. TRAIN AND EVALUATE MODELS

trained_models = {}
results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    trained_models[name] = pipeline

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    print(f"\n{'=' * 45}")
    print("MODEL:", name)
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")


# 7. COMPARE ALL MODELS

results_df = pd.DataFrame(results).T

print("\nMODEL COMPARISON")
print(results_df.round(3))

results_df.to_csv("model_comparison.csv")


# 8. PLOT MODEL PERFORMANCE

results_df["F1-Score"].plot(
    kind="bar",
    figsize=(10, 5),
    title="Classification Model Comparison"
)

plt.ylabel("F1-Score")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()


# 9. SELECT BEST MODEL BY F1-SCORE

best_name = results_df["F1-Score"].idxmax()
best_model = trained_models[best_name]

print("\nBEST MODEL:", best_name)


# 10. CONFUSION MATRIX

y_pred = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stays", "Leaves"],
    yticklabels=["Stays", "Leaves"]
)

plt.title(f"Confusion Matrix: {best_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))


# 11. PREDICT A NEW CUSTOMER

new_customer = pd.DataFrame([{
    "tenure": 4,
    "monthly_charges": 1800,
    "support_calls": 5,
    "contract_type": "Monthly",
    "internet_service": "Fiber",
    "late_payments": 3
}])

prediction = best_model.predict(new_customer)[0]

churn_probability = best_model.predict_proba(
    new_customer
)[0, 1]

print("\nNEW CUSTOMER PREDICTION")
print(f"Churn probability: {churn_probability:.2%}")

if prediction == 1:
    print("Customer is predicted to leave.")
else:
    print("Customer is predicted to stay.")


# 12. SAVE BEST MODEL

joblib.dump(
    best_model,
    "customer_churn_model.pkl"
)

print("\nProject completed successfully!")
print("Model saved as customer_churn_model.pkl")