
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Customer Churn AI",
    page_icon="📊",
    layout="wide"
)

# -----------------------------
# LOAD PROJECT FILES
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    return joblib.load(
        os.path.join(BASE_DIR, "customer_churn_model.pkl")
    )

@st.cache_data
def load_results():
    path = os.path.join(BASE_DIR, "model_comparison.csv")
    return pd.read_csv(path, index_col=0)

model = load_model()
results = load_results()

# Recreate the same synthetic dataset for dashboard statistics
@st.cache_data
def create_demo_data():
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

    return pd.DataFrame({
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "support_calls": support_calls,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "late_payments": late_payments,
        "churn": churn
    })

df = create_demo_data()

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📊 Customer Churn AI")
st.sidebar.caption("Machine Learning Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Customer",
        "Model Comparison",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Demo project using synthetic customer data."
)

# -----------------------------
# DASHBOARD HOME
# -----------------------------
if page == "Dashboard":

    st.title("Customer Churn Prediction")
    st.write(
        "Analyze customer data and estimate which "
        "customers may leave a subscription service."
    )

    total = len(df)
    churned = int(df["churn"].sum())
    retained = total - churned
    churn_rate = churned / total * 100

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Customers Churned", f"{churned:,}")
    c3.metric("Customers Retained", f"{retained:,}")
    c4.metric("Demo Churn Rate", f"{churn_rate:.1f}%")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Distribution")
        st.bar_chart(
            pd.DataFrame({
                "Customers": [retained, churned]
            }, index=["Stayed", "Left"])
        )

    with right:
        st.subheader("Contract Distribution")
        st.bar_chart(df["contract_type"].value_counts())

    st.subheader("Sample Customer Records")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "All dashboard statistics above use recreated "
        "synthetic data for demonstration."
    )

# -----------------------------
# PREDICT CUSTOMER
# -----------------------------
elif page == "Predict Customer":

    st.title("🔍 Predict Customer Churn")
    st.write("Enter customer details below.")

    with st.form("customer_form"):

        c1, c2 = st.columns(2)

        with c1:
            tenure = st.slider(
                "Tenure (months)", 1, 72, 12
            )
            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=300.0,
                max_value=2000.0,
                value=1000.0,
                step=50.0
            )
            support_calls = st.number_input(
                "Support Calls",
                min_value=0,
                max_value=30,
                value=2
            )

        with c2:
            contract_type = st.selectbox(
                "Contract Type",
                ["Monthly", "Yearly"]
            )
            internet_service = st.selectbox(
                "Internet Service",
                ["Fiber", "DSL"]
            )
            late_payments = st.number_input(
                "Late Payments",
                min_value=0,
                max_value=30,
                value=1
            )

        submitted = st.form_submit_button(
            "Predict Churn",
            type="primary",
            use_container_width=True
        )

    if submitted:

        customer = pd.DataFrame([{
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "support_calls": support_calls,
            "contract_type": contract_type,
            "internet_service": internet_service,
            "late_payments": late_payments
        }])

        prediction = int(model.predict(customer)[0])
        probability = float(
            model.predict_proba(customer)[0, 1]
        )

        st.divider()
        st.subheader("Prediction Result")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Estimated Churn Probability",
                f"{probability:.2%}"
            )
            st.progress(probability)

        with c2:
            if prediction == 1:
                st.error(
                    "Customer predicted to leave"
                )
                st.write(
                    "Consider reviewing support issues "
                    "and possible retention options."
                )
            else:
                st.success(
                    "Customer predicted to stay"
                )
                st.write(
                    "Continue providing good customer service."
                )

        st.caption(
            "This is a model estimate, not a guarantee. "
            "The model was trained on synthetic data."
        )

# -----------------------------
# MODEL COMPARISON
# -----------------------------
elif page == "Model Comparison":

    st.title("📈 Classification Model Comparison")

    st.write(
        "Compare the five trained algorithms "
        "using the test-set evaluation results."
    )

    st.dataframe(
        results.style.format("{:.3f}"),
        use_container_width=True
    )

    st.subheader("F1-Score Comparison")
    st.bar_chart(results["F1-Score"])

    st.subheader("Accuracy Comparison")
    st.bar_chart(results["Accuracy"])

    st.subheader("Saved Confusion Matrix")

    matrix_path = os.path.join(
        BASE_DIR, "confusion_matrix.png"
    )

    if os.path.exists(matrix_path):
        st.image(
            matrix_path,
            use_container_width=True
        )
    else:
        st.warning("Confusion matrix image not found.")

    st.subheader("Saved Model Comparison Graph")

    graph_path = os.path.join(
        BASE_DIR, "model_comparison.png"
    )

    if os.path.exists(graph_path):
        st.image(
            graph_path,
            use_container_width=True
        )

    best_name = results["F1-Score"].idxmax()

    st.info(
        f"Highest test F1-score: {best_name}"
    )

# -----------------------------
# ABOUT PROJECT
# -----------------------------
elif page == "About Project":

    st.title("About This Project")

    st.write("""
    This project demonstrates a customer churn
    classification system using Python and Machine Learning.
    """)

    st.subheader("Algorithms Used")

    st.markdown("""
    1. Logistic Regression
    2. K-Nearest Neighbors (KNN)
    3. Naive Bayes
    4. Decision Tree
    5. Support Vector Machine (SVM)
    """)

    st.subheader("Evaluation Metrics")

    st.markdown("""
    - Accuracy
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
    """)

    st.subheader("Technologies")

    st.write(
        "Python, Pandas, NumPy, Scikit-learn, "
        "Matplotlib, Seaborn and Streamlit"
    )

    st.warning(
        "This is an educational demonstration. "
        "The customer dataset is synthetic and should "
        "not be used for real business decisions."
    )
