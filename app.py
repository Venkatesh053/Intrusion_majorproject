import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests
import numpy as np
import os

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Network Intrusion Detection System",
                   page_icon="🛡️",
                   layout="wide")

# -------------------- LOTTIE HELPERS --------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None

LOTTIE_SAFE = "https://assets7.lottiefiles.com/packages/lf20_jtbfg2nb.json"  # shield
LOTTIE_ALERT = "https://assets7.lottiefiles.com/packages/lf20_jmgekfqg.json"  # alert
LOTTIE_DASH = "https://assets1.lottiefiles.com/packages/lf20_vf9g5v5p.json"   # dashboard spark

lottie_shield = load_lottie_url(LOTTIE_SAFE)
lottie_alert = load_lottie_url(LOTTIE_ALERT)
lottie_dash = load_lottie_url(LOTTIE_DASH)

# -------------------- STYLES --------------------
st.markdown("""
    <style>
    .main {background-color:#0f1116;}
    h1, h2, h3, h4, h5, h6 {color: #00c6ff;}
    .stButton>button {background-color:#00c6ff;color:white;border:none;padding:8px 20px;border-radius:8px;font-weight:bold}
    .stButton>button:hover {background-color:#0073e6}
    .sidebar .sidebar-content {background-color:#1a1c25}
    .card {background: linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:10px;}
    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODELS --------------------
@st.cache_resource
def load_models():
    try:
        models = {}
        base = "./models"
        if os.path.exists(os.path.join(base, "voting_model.pkl")):
            models["Voting"] = joblib.load(os.path.join(base, "voting_model.pkl"))
        if os.path.exists(os.path.join(base, "randomforest.pkl")):
            models["RandomForest"] = joblib.load(os.path.join(base, "randomforest.pkl"))
        if os.path.exists(os.path.join(base, "xgboost.pkl")):
            models["XGBoost"] = joblib.load(os.path.join(base, "xgboost.pkl"))
        if os.path.exists(os.path.join(base, "svm.pkl")):
            models["SVM"] = joblib.load(os.path.join(base, "svm.pkl"))

        scaler = None
        features = None
        if os.path.exists(os.path.join(base, "scaler.pkl")):
            scaler = joblib.load(os.path.join(base, "scaler.pkl"))
        if os.path.exists(os.path.join(base, "model_features.pkl")):
            features = joblib.load(os.path.join(base, "model_features.pkl"))

        return models, scaler, features
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return {}, None, None

models, scaler, model_features = load_models()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dataset Overview", "⚡ Live Attack Prediction"]) 
st.sidebar.markdown("---")
st.sidebar.info("Project: Intrusion Detection using NSL-KDD Developed for Major Project Demo")

# -------------------- ATTACK MAPPINGS & LABELS --------------------
nsl_attack_to_category = {
    "neptune":"DoS", "smurf":"DoS", "back":"DoS", "teardrop":"DoS", "pod":"DoS", "land":"DoS", "apache2":"DoS", "mailbomb":"DoS",
    "satan":"Probe", "ipsweep":"Probe", "nmap":"Probe", "portsweep":"Probe", "mscan":"Probe", "saint":"Probe",
    "guess_passwd":"R2L", "ftp_write":"R2L", "imap":"R2L", "phf":"R2L", "spy":"R2L", "warezclient":"R2L", "warezmaster":"R2L",
    "xlock":"R2L", "xsnoop":"R2L", "sendmail":"R2L", "named":"R2L",
    "buffer_overflow":"U2R", "loadmodule":"U2R", "perl":"U2R", "rootkit":"U2R", "httptunnel":"U2R", "ps":"U2R",
}

BINARY_LABEL_MAP = {0: "Normal Traffic 🟢", 1: "Attack Detected 🔴", '0': "Normal Traffic 🟢", '1': "Attack Detected 🔴"}
BOOL_TEXT_MAP = {0: "No", 1: "Yes"}
LOGGED_IN_MAP = {0: "Not Logged In 🔒", 1: "Logged In 🔓"}
LAND_MAP = {0: "Different Host 🌐", 1: "Same Host 🖥️"}
GUEST_MAP = {0: "Regular User 👤", 1: "Guest User 👥"}

# Helper to map model numeric predictions to text
def preds_to_text(preds):
    return [BINARY_LABEL_MAP.get(int(p), str(p)) if not pd.isna(p) else 'Unknown' for p in preds]

# -------------------- SAFE CSV READER --------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read CSV {path}: {e}")
        return pd.DataFrame()

# -------------------- DATASET OVERVIEW --------------------
if page == "📊 Dataset Overview":
    st.title("📊 NSL-KDD Dataset Overview")

    default_path = "./datasets/nslkdd/nslkdd_processed.csv"
    df = safe_read_csv(default_path)
    if df.empty:
        st.warning(f"Default dataset not found at {default_path}. Upload your dataset below.")
        uploaded = st.file_uploader("Upload NSL-KDD processed CSV", type=["csv"]) 
        if uploaded:
            df = pd.read_csv(uploaded)

    if df.empty:
        st.stop()

    # Detect label column
    label_col = None
    for cand in ["labels", "label", "attack", "class", "target", "attack_type"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        st.error("No label column found (expected one of labels,label,attack,class,target,attack_type).")
        st.stop()

    # Create readable category and attack name columns
    def map_label_to_category(v):
        try:
            if pd.isna(v):
                return "Unknown"
            if isinstance(v, (int, np.integer)):
                return "Normal" if int(v) == 0 else "Attack"
            s = str(v).lower()
            if s == 'normal' or s == 'normal.':
                return "Normal"
            return nsl_attack_to_category.get(s, "Attack")
        except Exception:
            return "Attack"

    def map_label_to_attackname(v):
        try:
            if pd.isna(v):
                return "Unknown"
            if isinstance(v, (int, np.integer)):
                return "Normal" if int(v) == 0 else "Attack"
            s = str(v).lower()
            return s
        except Exception:
            return str(v)

    df['category'] = df[label_col].apply(map_label_to_category)
    df['attack_name'] = df[label_col].apply(map_label_to_attackname)

    total = len(df)
    normal_count = (df['category'] == 'Normal').sum()
    attack_count = total - normal_count

    # Top header with Lottie
    header_col1, header_col2 = st.columns([3,1])
    with header_col1:
        st.metric("Total records", total)
    with header_col2:
        if lottie_dash:
            st_lottie(lottie_dash, height=120)

    # Plotly Donut for Normal vs Attack
    fig_pie = go.Figure(go.Pie(labels=['Normal','Attack'], values=[normal_count, attack_count], hole=0.45))
    fig_pie.update_traces(textinfo='percent+label')
    fig_pie.update_layout(title='Normal vs Attack Distribution', transition={'duration':500})
    st.plotly_chart(fig_pie, use_container_width=True)

    # Animated-like bar (attack category counts) — using frame trick of cumulative top N (simple animation loop not necessary)
    attack_types = df[df['category'] != 'Normal']['attack_name'].value_counts().reset_index()
    attack_types.columns = ['attack','count']
    if not attack_types.empty:
        fig_bar = px.bar(attack_types.head(30), x='attack', y='count', title='Top Attack Types', labels={'count':'Count','attack':'Attack Name'})
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    # Numeric scatter with options
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.markdown("### Numeric feature scatter (sample)")
        x_col = st.selectbox("X feature", numeric_cols, index=0)
        y_col = st.selectbox("Y feature", numeric_cols, index=1)
        sample_df = df.sample(min(3000, len(df)))
        fig_sc = px.scatter(sample_df, x=x_col, y=y_col, color='category', hover_data=['attack_name'], title=f"{x_col} vs {y_col} (sample)")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top numeric feature summaries")
    st.dataframe(df[numeric_cols].describe().T)

# -------------------- LIVE PREDICTION --------------------
elif page == "⚡ Live Attack Prediction":
    st.title("⚡ Live Attack Prediction Dashboard")

    st.markdown("""
        Upload a single **NSL-KDD formatted record** (CSV or multiple)  
        and get predictions from all trained models in real time.
    """)

    uploaded_file = st.file_uploader("📂 Upload a CSV file (with or without 'labels' column)", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("### 🧾 Uploaded Data Preview")
        st.dataframe(df_input.head(), use_container_width=True)

        # --- Attack mapping dictionary ---
        nsl_attack_to_category = {
            "neptune":"DoS", "smurf":"DoS", "back":"DoS", "teardrop":"DoS", "pod":"DoS", "land":"DoS", "apache2":"DoS", "mailbomb":"DoS",
            "satan":"Probe", "ipsweep":"Probe", "nmap":"Probe", "portsweep":"Probe", "mscan":"Probe", "saint":"Probe",
            "guess_passwd":"R2L", "ftp_write":"R2L", "imap":"R2L", "phf":"R2L", "spy":"R2L", "warezclient":"R2L", "warezmaster":"R2L",
            "xlock":"R2L", "xsnoop":"R2L", "sendmail":"R2L", "named":"R2L",
            "buffer_overflow":"U2R", "loadmodule":"U2R", "perl":"U2R", "rootkit":"U2R", "httptunnel":"U2R", "ps":"U2R",
        }

        # --- Handle categorical encoding ---
        cat_cols = ["protocol_type", "service", "flag"]
        if any(col in df_input.columns for col in cat_cols):
            df_input = pd.get_dummies(df_input, columns=[c for c in cat_cols if c in df_input.columns])

        # --- Align features ---
        for col in model_features:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[model_features]

        # --- Scale data ---
        X_scaled = scaler.transform(df_input)

        # --- Predictions ---
        preds = {name: model.predict(X_scaled) for name, model in models.items()}

        # --- Combine results ---
        results = pd.DataFrame(preds)
        results['Final'] = results['Voting']

        # --- Add readable text columns ---
        label_map = {0: "Normal Traffic 🟢", 1: "Attack Detected 🔴"}
        for col in results.columns:
            results[f"{col}_text"] = results[col].map(label_map)

        # --- Attack name and category mapping ---
        if 'labels' in df_input.columns:
            results['attack_name'] = df_input['labels']
            results['attack_category'] = df_input['labels'].apply(
                lambda x: "Normal" if x == "normal" else nsl_attack_to_category.get(x, "Other")
            )
        else:
            results['attack_name'] = results['Final'].map({0: "normal", 1: "attack"})
            results['attack_category'] = results['Final'].map({0: "Normal", 1: "Attack"})

        # --- Show prediction results ---
        st.subheader("🧠 Model Predictions")
        st.dataframe(results[[f"{c}_text" for c in preds.keys()] + ["Final_text", "attack_name", "attack_category"]],
                     use_container_width=True)

        # --- Summary counts visualization ---
        summary = results['attack_category'].value_counts().reset_index()
        summary.columns = ['Category', 'Count']
        st.plotly_chart(
            px.pie(summary, names='Category', values='Count', color='Category',
                   color_discrete_sequence=px.colors.qualitative.Safe,
                   title="Attack vs Normal Distribution"),
            use_container_width=True
        )

        # --- Download predictions ---
        csv_download = results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_download,
            file_name="intrusion_predictions.csv",
            mime="text/csv"
        )

    else:
        st.info("📤 Please upload a CSV file to begin prediction.")

    st.markdown("---")
    st.caption("Model: Voting Ensemble | Dataset: NSL-KDD | Enhanced Visualization + Attack Mapping")
