import streamlit as st
import pandas as pd
from utils.data_loader import load_nsl_dataset
from utils.model_utils import load_models
from utils.visualization import plot_distribution, plot_confusion_matrix

st.set_page_config(page_title="IDS Demo", layout="wide")

st.title("🚨 Intrusion Detection System (IDS) using ML")

# Load dataset & models
df = load_nsl_dataset()
rf, xgb, svm, voting, scaler = load_models()

# Sidebar
st.sidebar.title("Navigation")
choice = st.sidebar.radio("Go to", ["📊 Dataset Overview", "🤖 Model Evaluation", "⚡ Live Prediction"])

# -------------------- DATASET --------------------
if choice == "📊 Dataset Overview":
    st.header("Dataset Overview")
    st.dataframe(df.head(10))

    st.subheader("Class Distribution")
    st.pyplot(plot_distribution(df))

# -------------------- EVALUATION --------------------
elif choice == "🤖 Model Evaluation":
    st.header("Model Performance")

    X = df.drop("labels", axis=1)
    y = df["labels"]

    # Convert categorical
    X = pd.get_dummies(X, columns=["protocol_type", "service", "flag"])
    X = scaler.transform(X)

    y_pred = voting.predict(X)
    st.subheader("Confusion Matrix (Voting)")
    st.pyplot(plot_confusion_matrix(y, y_pred))

# -------------------- PREDICTION --------------------
elif choice == "⚡ Live Prediction":
    st.header("Live Prediction")

    duration = st.number_input("Duration", min_value=0, max_value=1000, value=0)
    src_bytes = st.number_input("Source Bytes", min_value=0, max_value=100000, value=491)
    dst_bytes = st.number_input("Destination Bytes", min_value=0, max_value=100000, value=0)

    protocol_type = st.selectbox("Protocol Type", ["tcp", "udp", "icmp"])
    service = st.selectbox("Service", ["http", "smtp", "ftp", "other"])
    flag = st.selectbox("Flag", ["SF", "S0", "REJ"])

    if st.button("Predict"):
        sample = pd.DataFrame([{
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "protocol_type": protocol_type,
            "service": service,
            "flag": flag
        }])

        # One-hot encoding
        sample = pd.get_dummies(sample)
        train_X = pd.get_dummies(df.drop("labels", axis=1), columns=["protocol_type", "service", "flag"])
        sample = sample.reindex(columns=train_X.columns, fill_value=0)

        sample = scaler.transform(sample)

        preds = {
            "RandomForest": rf.predict(sample)[0],
            "XGBoost": xgb.predict(sample)[0],
            "SVM": svm.predict(sample)[0],
            "Voting": voting.predict(sample)[0]
        }

        st.subheader("🔮 Predictions")
     
     