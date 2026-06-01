🎗 Breast Cancer Relapse Prediction using Multi-Modal AI

This project implements a multi-modal deep learning system to predict breast cancer relapse risk by combining genomic, clinical, and medical imaging data.
It provides an interactive Streamlit web application for real-time prediction using a trained fusion model.

📌 Project Overview

Breast cancer relapse prediction is a complex task that benefits from integrating multiple data sources.
This system uses:

🧬 Genomic data (500 gene expression features)

📋 Clinical data (age, tumor stage, treatment, lymph nodes, tumor size)

🖼 Medical images (MRI / Histopathology)

A fusion neural network combines all three modalities to predict relapse risk probability.

🧠 Key Features

Multi-modal AI (Genomic + Clinical + Imaging)

Safe, version-independent preprocessing (no sklearn dependency at runtime)

Interactive and animated Streamlit UI

Upload-based prediction (CSV + Image)

Clear HIGH RISK / LOW RISK classification with probability

🗂 Project Structure
📁 Project Root
│
├── app.py                          # Streamlit frontend application
├── create_preprocessors_safe.py    # One-time preprocessing generator
├── code.ipynb                      # Model training & experimentation notebook
│
├── clinical.csv                    # Clinical dataset (training)
├── genomic.csv                     # Genomic dataset (training)
├── genomic_500genes_high_risk.csv  # Sample test genomic input
│
├── preprocessors_safe/
│   ├── loader.py                   # Runtime preprocessing functions
│   ├── genomic_mean.npy
│   ├── genomic_std.npy
│   ├── genomic_cols.json
│   ├── clin_num_mean.npy
│   ├── clin_num_std.npy
│   ├── clin_numeric_cols.json
│   └── clin_cat_vocabs.json
│
├── models_fusion/
│   └── fusion_model.keras          # Trained fusion model
│
└── README.md

⚙️ Technologies Used

Python 3.9+

TensorFlow / Keras

Streamlit

NumPy, Pandas

Deep Learning (CNN + Dense Fusion)

🧪 Input Requirements
1️⃣ Genomic CSV

Must contain exactly 500 columns:

gene_1, gene_2, ..., gene_500


One row per patient

2️⃣ Clinical Inputs (via UI)

Age

Tumor Stage (I / II / III / IV)

Treatment Type

Positive Lymph Nodes

Tumor Size (cm)

3️⃣ Medical Image

Format: .jpg, .jpeg, or .png

MRI or histopathology image

🚀 How to Run the Project
Step 1: Create Virtual Environment (Recommended)
python -m venv fusionenv
cancerenv\Scripts\activate   # Windows

Step 2: Install Dependencies
pip install streamlit tensorflow keras numpy pandas pillow

Step 3: (One-Time) Generate Safe Preprocessors

Run this only once:

python create_preprocessors_safe.py


This creates the preprocessors_safe/ folder required by the app.

Step 4: Run Streamlit App
streamlit run app.py


The app will open in your browser:

http://localhost:8501

🖥 Application Workflow

Enter clinical details

Upload genomic CSV file

Upload medical image

Click Predict Relapse Risk

View AI-generated risk prediction

📊 Output Interpretation

🟢 LOW RISK → Probability < 0.5

🔴 HIGH RISK → Probability ≥ 0.5

The output includes:

Risk category

Exact relapse probability score

⚠️ This system is for academic and research purposes only, not for clinical diagnosis.

🔐 Model & Preprocessing Safety

No sklearn pickle files used

All preprocessing uses NumPy + JSON

Version-safe deployment

Streamlit-friendly caching

📈 Future Enhancements

True federated learning implementation

Explainable AI (SHAP / Grad-CAM)

Multi-patient batch prediction

Cloud deployment (Streamlit Cloud / AWS)