# 🩺 Breast Cancer Relapse Predictor

AI-powered multimodal breast cancer relapse prediction using **MobileNet CNN**, **Multi-Layer Perceptron (MLP)**, **Feature Fusion**, and **Streamlit**.

---

## 📌 Project Overview

Breast cancer recurrence remains a major challenge in oncology. Traditional prediction systems often rely on a single type of medical data, limiting predictive performance.

This project introduces a **Multi-Fusion Deep Learning Model** that combines:

* 🖼️ Medical Imaging Data
* 🧬 Genomic Data
* 📊 Clinical Data

The system uses **MobileNet** for image feature extraction and **MLP (Multi-Layer Perceptron)** for structured clinical/genomic data analysis. The extracted features are fused together to predict breast cancer relapse risk more accurately.

---

## 🚀 Features

* Deep Learning-based Relapse Prediction
* MobileNet CNN for Medical Image Analysis
* MLP for Clinical and Genomic Data Processing
* Feature Fusion Architecture
* Interactive Streamlit Web Application
* Real-time Prediction Results
* User-Friendly Interface
* Research-Oriented Healthcare AI Solution

---

## 🏗️ System Architecture

```text
Medical Images
      │
      ▼
 MobileNet CNN
      │
      ▼
Image Features
      │
      ├──────────────┐
      │              │
      ▼              ▼
Clinical Data    Genomic Data
      │
      ▼
     MLP
      │
      ▼
Structured Features
      │
      ▼
 Feature Fusion
      │
      ▼
 Fully Connected Layer
      │
      ▼
 Relapse Prediction
```

## 🛠️ Tech Stack

### Programming Language

* Python

### Deep Learning Frameworks

* TensorFlow
* Keras

### Libraries

* NumPy
* Pandas
* OpenCV
* Scikit-Learn
* Matplotlib

### Frontend

* Streamlit

### Development Tools

* VS Code
* Git
* GitHub

---

## 📂 Project Structure

```text
Breast-Cancer-Relapse-Predictor/
│
├── app.py
├── clinical.csv
├── genomic.csv
├── federated_global_classifier.keras
├── clin_pre.pkl
├── gen_pre.pkl
│
├── image/
├── models_fusion/
├── preprocessors_safe/
├── testingdata/
├── Notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/sejalP07/Breast-Cancer-Relapse-Predictor.git
cd Breast-Cancer-Relapse-Predictor
```

### Create Virtual Environment

```bash
python -m venv cancerenv
```

### Activate Environment

Windows:

```bash
cancerenv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will open at:

```text
http://localhost:8501
```

---

## 📈 Model Workflow

1. Upload Medical Image
2. Upload Clinical Data
3. Upload Genomic Data
4. Data Preprocessing
5. Feature Extraction
6. Feature Fusion
7. Prediction Generation
8. Display Relapse Risk

---

## 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

These metrics help assess the reliability and performance of relapse prediction.

---

## 🔬 Research Contribution

This project demonstrates how multimodal healthcare data can be integrated using deep learning techniques to improve breast cancer relapse prediction.

Key contributions include:

* Fusion of image and structured healthcare data
* Lightweight MobileNet architecture
* Efficient prediction pipeline
* Practical deployment using Streamlit

---

## 🎯 Future Enhancements

* Explainable AI (XAI)
* SHAP-based Interpretability
* Attention Heatmaps
* Federated Learning Integration
* Cloud Deployment
* Real-time Clinical Integration
* Multi-Cancer Prediction Support

---

## 👩‍💻 Author

**Sejal P**

Master of Computer Applications (MCA)

RV Institute of Technology and Management

Bengaluru, India

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share with others

---

## 📜 License

This project is intended for academic and research purposes.
