import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from preprocessors_safe.loader import transform_genomic, transform_clinical
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(page_title="Breast Cancer Relapse Predictor", layout="wide",initial_sidebar_state="expanded")

# ==================================================
# PAGE STATE
# ==================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ==================================================
# STYLES
# ==================================================
st.markdown("""
<style>

/* ================================
   GLOBAL APP BACKGROUND
================================ */
.stApp {
    background: linear-gradient(135deg, #dbeafe, #ede9fe, #fce7f3);
    font-family: "Segoe UI", sans-serif;
}

/* ================================
   HEADINGS
================================ */
h1, h2, h3, h4, h5, h6 {
    color: #111827 !important;
    font-weight: 800 !important;
}

/* ================================
   HIDE STREAMLIT DEFAULT HEADER
================================ */
header[data-testid="stHeader"] {
    display: none;
}

/* Remove top padding added by Streamlit */
div.block-container {
    padding-top: 90px !important;
}

/* ================================
   BODY TEXT (SAFE ELEMENTS ONLY)
================================ */
p, li, label {
    color: #1f2937 !important;
    font-size: 16px;
}

/* ================================
   BUTTON TEXT
================================ */
/* ================================
   TOP NAVBAR
================================ */
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100px;
    background: linear-gradient(90deg, #1e3a8a, #7e22ce);
    display: flex;
    align-items: center;
    padding-left: 320px; /* aligns with sidebar */
    z-index: 9999;
    box-shadow: 0 6px 18px rgba(0,0,0,0.25);
}


/* App title (LEFT) */
.nav-title {
    color: white;
    font-size: 30px;
    font-weight: 900;
}

/* Nav buttons */
.nav-buttons button {
    background: None;
    color: #1e293b;
    border: none;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 15px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.25s ease;
}

.nav-buttons button:hover {
    background: #e0e7ff;
    transform: translateY(-1px);
}


/* -------- Hide Streamlit default header -------- */
header[data-testid="stHeader"] {
    display: none;
}

/* -------- FORCE SIDEBAR ALWAYS OPEN -------- */
section[data-testid="stSidebar"] {
    min-width: 280px !important;
    max-width: 280px !important;
    transform: translateX(0px) !important;
    visibility: visible !important;
    background: linear-gradient(180deg, #1e3a8a, #7e22ce);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

/* Disable sidebar toggle */
button[data-testid="collapsedControl"] {
    display: none !important;
}

/* -------- SIDEBAR BUTTONS -------- */
section[data-testid="stSidebar"] .stButton > button {
    background: None;
    color: black !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 10px !important;
}


/* ================================
   TABS
================================ */
.stTabs [role="tab"] {
    color: #374151 !important;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    font-weight: 800;
}

/* ================================
   DATAFRAME CONTAINER
================================ */
.stDataFrame {
    background-color: white !important;
    border-radius: 8px;
}

/* ================================
   INPUT FIELDS LABELS
================================ */
label {
    color: #111827 !important;
    font-weight: 600;
}

/* ================================
   METRICS (if used)
================================ */
[data-testid="stMetricLabel"] {
    color: #374151 !important;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 22px;
    font-weight: 800;
}

/* ================================
   SAFE TITLE STYLE (NO TRANSPARENCY)
================================ */
.title {
    font-size: 36px;
    font-weight: 900;
    text-align: center;
    color: #1e40af;
    margin: 30px 0;
}
/* ================================
   SELECTBOX / DROPDOWN FIX
================================ */

/* Selected value text */
div[data-baseweb="select"] span {
    color: #111827 !important;
}

/* Dropdown menu background */
ul[role="listbox"] {
    background-color: white !important;
}

/* Dropdown options text */
li[role="option"] {
    color: #111827 !important;
    background-color: white !important;
}

/* Hovered option */
li[role="option"]:hover {
    background-color: #e0e7ff !important;
    color: #111827 !important;
}

/* Placeholder text */
div[data-baseweb="select"] input {
    color: #111827 !important;
}
/* =====================================
   SUBMIT BUTTON STYLE (st.button)
===================================== */

/* Main button */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #9333ea) !important;
    border-radius: 14px !important;
    padding: 14px 22px !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.35);
    transition: all 0.3s ease-in-out;
}

/* Button text */
.stButton > button span,
.stButton > button p {
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

/* Hover effect */
.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #7e22ce) !important;
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.45);
}

/* Active (click) effect */
.stButton > button:active {
    transform: scale(0.98);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Full-width button (when use_container_width=True) */
.stButton {
    width: 100%;
}


</style>
""", unsafe_allow_html=True)


# ==================================================
# NAVIGATION
# ==================================================
# Use gap parameter or adjust column ratios

# _, center, _ = st.columns([1, 2, 1])
# with center:
#     nav1, nav2, nav3 = st.columns(3, gap="small")
#     with nav1:
#         st.button("🏠 Home", use_container_width=True)
#     with nav2:
#         st.button("📊 EDA", use_container_width=True)
#     with nav3:
#         st.button("🧪 Test", use_container_width=True)
        
# ==================================================
# TOP NAVBAR
# ==================================================

st.markdown("""
<div class="navbar">
    <div class="nav-title">🧠 Breast Cancer Relapse Predictor</div>
</div>
""", unsafe_allow_html=True)

# ==================================================
# SIDEBAR NAVIGATION
# ==================================================
with st.sidebar:
    st.markdown("")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"

    if st.button("📊 EDA", use_container_width=True):
        st.session_state.page = "EDA"

    if st.button("🧪 Test", use_container_width=True):
        st.session_state.page = "TestForm"




# ==================================================
# MODEL
# ==================================================
FUSION_MODEL = "models_fusion/fusion_model.keras"
IMG_H, IMG_W = 224, 224

@st.cache_resource
def load_fusion_model():
    return tf.keras.models.load_model(FUSION_MODEL, compile=False)

fusion_model = load_fusion_model()

def preprocess_image(img_file):
    img = load_img(img_file, target_size=(IMG_H, IMG_W))
    return img_to_array(img).astype("float32") / 255.0

# ==================================================
# HOME PAGE
# ==================================================
# ==================================================
# HOME PAGE
# ==================================================
if st.session_state.page == "Home":

    st.markdown("<div class='title'>Breast Cancer Relapse Prediction System</div>",
                unsafe_allow_html=True)

    st.markdown("""
### 🧠 Project Overview
Breast cancer relapse remains a major challenge in oncology. Predicting the risk of recurrence at an early stage can significantly improve treatment planning and patient outcomes.

The **Breast Cancer Relapse Prediction System** is a **Multi-Modal Artificial Intelligence platform** that analyzes multiple medical data sources to predict the likelihood of cancer recurrence.

The system integrates:

• **Clinical Data** – patient age, tumor stage, treatment type, lymph nodes, and tumor size  
• **Genomic Features** – gene expression and molecular biomarkers  
• **Medical Imaging** – histopathology or mammogram images  

By combining these data sources, the system provides a **more accurate and reliable relapse prediction model**.

---

### 🎯 Objective
The primary goal of this system is to **detect the risk of breast cancer relapse at an early stage** and assist doctors in clinical decision-making.

The system aims to:

✔ Identify patients with high relapse risk  
✔ Support personalized treatment planning  
✔ Improve early monitoring and preventive care  
✔ Assist clinicians with AI-driven insights

---

### ⚙️ Key Features
• Multi-modal AI prediction model  
• Integration of clinical, genomic, and imaging data  
• Interactive data visualization (EDA)  
• User-friendly interface for testing predictions  
• Real-time relapse risk prediction

---

### 📊 System Workflow
1️⃣ User enters **clinical information**  
2️⃣ Uploads **genomic CSV data**  
3️⃣ Uploads **medical image**  
4️⃣ AI model processes multi-modal data  
5️⃣ System predicts **Relapse Risk (High / Low)**

---

### 💡 Impact
This system demonstrates how **Artificial Intelligence can support precision medicine**, helping doctors identify high-risk patients early and improve treatment strategies.
""")

# ==================================================
# EDA PAGE (CLINICAL + GENOMIC)
# ==================================================
if st.session_state.page == "EDA":

    st.title("📊 Exploratory Data Analysis (EDA)")

    if not os.path.exists("clinical.csv") or not os.path.exists("genomic.csv"):
        st.error("❌ clinical.csv or genomic.csv not found")
        st.stop()

    clinical_df = pd.read_csv("clinical.csv")
    genomic_df = pd.read_csv("genomic.csv")

    clinical_df.columns = clinical_df.columns.str.lower().str.replace(" ", "_")
    genomic_df.columns = genomic_df.columns.str.lower().str.replace(" ", "_")

    tab1, tab2 = st.tabs(["🩺 Clinical EDA", "🧬 Genomic EDA"])

    # ------------------------------
    # CLINICAL EDA
    # ------------------------------
    with tab1:
        st.subheader("📄 Clinical Dataset Preview")
        st.dataframe(clinical_df.head())

        def find_col(keys):
            for c in clinical_df.columns:
                for k in keys:
                    if k in c:
                        return c
            return None

        age = find_col(["age"])
        tumor = find_col(["tumor"])
        lymph = find_col(["lymph"])
        stage = find_col(["stage"])
        treatment = find_col(["treatment"])
        relapse = find_col(["relapse"])

        st.subheader("📈 Histograms")
        c1, c2, c3 = st.columns(3)

        with c1:
            if age:
                fig, ax = plt.subplots()
                sns.histplot(clinical_df[age], kde=True, ax=ax)
                ax.set_title("Age Distribution")
                st.pyplot(fig)

        with c2:
            if tumor:
                fig, ax = plt.subplots()
                sns.histplot(clinical_df[tumor], kde=True, ax=ax)
                ax.set_title("Tumor Size Distribution")
                st.pyplot(fig)

        with c3:
            if lymph:
                fig, ax = plt.subplots()
                sns.histplot(clinical_df[lymph], kde=True, ax=ax)
                ax.set_title("Lymph Nodes Distribution")
                st.pyplot(fig)

        st.subheader("📊 Categorical Distributions")
        c4, c5, c6 = st.columns(3)

        with c4:
            if stage:
                fig, ax = plt.subplots()
                sns.countplot(x=stage, data=clinical_df, ax=ax)
                ax.set_title("Tumor Stage")
                st.pyplot(fig)

        with c5:
            if treatment:
                fig, ax = plt.subplots()
                sns.countplot(x=treatment, data=clinical_df, ax=ax)
                ax.set_title("Treatment Type")
                st.pyplot(fig)

        with c6:
            if relapse:
                fig, ax = plt.subplots()
                sns.countplot(x=relapse, data=clinical_df, ax=ax)
                ax.set_title("Relapse Distribution")
                st.pyplot(fig)

        st.subheader("📦 Numerical vs Relapse")
        c7, c8, c9 = st.columns(3)

        with c7:
            if age and relapse:
                fig, ax = plt.subplots()
                sns.boxplot(x=relapse, y=age, data=clinical_df, ax=ax)
                st.pyplot(fig)

        with c8:
            if tumor and relapse:
                fig, ax = plt.subplots()
                sns.boxplot(x=relapse, y=tumor, data=clinical_df, ax=ax)
                st.pyplot(fig)

        with c9:
            if lymph and relapse:
                fig, ax = plt.subplots()
                sns.boxplot(x=relapse, y=lymph, data=clinical_df, ax=ax)
                st.pyplot(fig)

        # st.subheader("🔥 Correlation Heatmap")
        # num_df = clinical_df.select_dtypes(include=["int64", "float64"])
        # if not num_df.empty:
        #     fig, ax = plt.subplots(figsize=(4, 3))
        #     sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        #     st.pyplot(fig)
        st.subheader("🔥 Correlation Heatmap")

        num_df = clinical_df.select_dtypes(include=["int64", "float64"])
        if not num_df.empty:
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            sns.heatmap(
                num_df.corr(),
                annot=True,
                annot_kws={"size": 4},
                cmap="coolwarm",
                cbar=False,
                ax=ax
            )
            ax.tick_params(axis='x', labelsize=4, rotation=45)
            ax.tick_params(axis='y', labelsize=4)
            plt.tight_layout()
            st.pyplot(fig,use_container_width=False)


    # ------------------------------
    # GENOMIC EDA
    # ------------------------------
    with tab2:
        st.subheader("🧬 Genomic Dataset Preview")
        st.dataframe(genomic_df.head())

        numeric_genes = genomic_df.select_dtypes(include=["int64", "float64"])

        st.subheader("📈 Gene Expression Distribution")
        gene = st.selectbox("Select a Gene", numeric_genes.columns[:50])

        fig, ax = plt.subplots()
        sns.histplot(numeric_genes[gene], kde=True, ax=ax)
        ax.set_title(f"{gene} Expression")
        st.pyplot(fig)

        st.subheader("🔥 Genomic Correlation Heatmap (Top 20 Genes)")
        top_genes = numeric_genes.iloc[:, :20]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(top_genes.corr(), cmap="viridis", ax=ax)
        st.pyplot(fig)

# ==================================================
# TEST FORM PAGE
# ==================================================
# ==================================================
# TEST FORM PAGE (FIXED)
# ==================================================
if st.session_state.page == "TestForm":

    st.title("🧪 Breast Cancer Relapse Prediction")

    # -------------------------------
    # FORM (NO SUBMIT BUTTON INSIDE)
    # -------------------------------
    with st.form("prediction_form"):

        st.subheader("🩺 Clinical Information")
        c1, c2 = st.columns(2)

        with c1:
            age = st.number_input("Age", 18, 100, 50)
            lymph_nodes = st.number_input("Positive Lymph Nodes", 0, 40, 2)
            tumor_size_cm = st.number_input("Tumor Size (cm)", 0.0, 20.0, 3.0)

        with c2:
            tumor_stage = st.selectbox("Tumor Stage", ["I", "II", "III", "IV"])
            treatment_type = st.selectbox(
                "Treatment Type",
                ["Chemo", "Radiation", "Hormonal", "Surgery", "Other"]
            )

        st.subheader("🧬 Genomic CSV")
        genomic_file = st.file_uploader("Upload Genomic CSV", type=["csv"])
        

        st.subheader("🖼 Medical Image")
        img_file = st.file_uploader(
            "Upload Image",
            type=["jpg", "png", "jpeg"]
        )
      

        # ❌ NO submit button here
        st.form_submit_button("Form Ready")  # dummy to lock inputs

    # -------------------------------
    # REAL SUBMIT BUTTON (VISIBLE)
    # -------------------------------
    submit = st.button(
        "🧪 Predict Relapse Risk",
        use_container_width=True
    )

    if submit:
        if genomic_file is None or img_file is None:
            st.error("❌ Please upload all inputs.")
            st.stop()

        Xg = transform_genomic(pd.read_csv(genomic_file).iloc[:1])

        Xc = transform_clinical(pd.DataFrame([{
            "age": age,
            "tumor_stage": tumor_stage,
            "treatment_type": treatment_type,
            "lymph_nodes": lymph_nodes,
            "tumor_size_cm": tumor_size_cm
        }]))

        Xi = np.expand_dims(preprocess_image(img_file), axis=0)

        prob = fusion_model.predict([Xg, Xc, Xi])[0][0]

        if prob >= 0.5:
            st.error(f"🛑 HIGH RISK – Probability: {prob:.4f}")
        else:
            st.success(f"🟢 LOW RISK – Probability: {prob:.4f}")

