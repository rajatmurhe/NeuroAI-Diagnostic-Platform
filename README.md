# 🧠 NeuroAI: Explainable Clinical Diagnostic Platform

An end-to-end Artificial Intelligence diagnostic microservice built to predict cognitive impairment stages (**Cognitively Normal, Mild Cognitive Impairment, Alzheimer’s Disease**) from multimodal patient biomarkers. 

This project bridges the gap between traditional tabular Machine Learning and modern Natural Language Processing (NLP) by benchmarking ensemble algorithms against a fine-tuned sequence classifier (BERT) on structured clinical data.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-F7931E?logo=huggingface)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-green)

---

## 🖥️ Live Diagnostic Dashboard
> *Real-time inference and SHAP cryptographic feature attribution deployed via FastAPI.*

<img src="dashboard.png" alt="NeuroAI Dashboard" width="800">

---

## 📊 Performance Benchmarks & Terminal Logs

Evaluated on the OASIS Cross-Sectional MRI and Clinical dataset (stratified 80/20 split).

| Model Architecture | Task Type | Accuracy | Weighted F1-Score |
| :--- | :--- | :--- | :--- |
| **Random Forest** | Tabular Ensemble | 85.23% | 85.23% |
| **XGBoost** | Gradient Boosting | 85.23% | 83.68% |
| **Logistic Regression** | Tabular ML | 82.95% | 82.28% |
| **Fine-Tuned BERT** | Transformer Classifier | 81.82% | 82.35% |
| **SVM (RBF Kernel)** | Kernel ML | 79.55% | 79.34% |

### Detailed Evaluation Metrics

**1. Random Forest & Logistic Regression**
<br>
<img src="Screenshot 2026-08-24 at 3.13.05 PM.png" width="600">

**2. XGBoost**
<br>
<img src="XGBoost.png" width="600">

**3. Fine-Tuned BERT (Sequence Classifier)**
<br>
<img src="Bert.png" width="600">

**4. Support Vector Machine (RBF)**
<br>
<img src="SVM training.png" width="600">

---

## 🛠️ Project Architecture & Folder Structure

```text
NeuroAI-Diagnostic-Platform/
│
├── api.py                     # FastAPI server and inference endpoints
├── requirements.txt           # Environment dependencies
│
├── data/                      # Clinical datasets
│   └── oasis_cross-sectional.csv 
│
├── src/                       # Core ML pipelines
│   ├── preprocess.py          # Imputation, scaling, and NLP serialization
│   ├── train_baseline.py      # Script to train traditional ML models
│   ├── train_bert.py          # Hugging Face Trainer for BERT fine-tuning
│   └── explainability.py      # SHAP integration & plot generation
│
├── templates/                 # Frontend User Interface
│   └── index.html             # Tailwind CSS & JS dashboard
│
└── models/                    # Serialized joblib artifacts (Scalers, Encoders, Models)

```

---

## Local Installation & Setup

**1. Clone the repository**

```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/NeuroAI-Diagnostic-Platform.git](https://github.com/YOUR_GITHUB_USERNAME/NeuroAI-Diagnostic-Platform.git)
cd NeuroAI-Diagnostic-Platform

```

**2. Create a virtual environment & install dependencies**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

**3. Train the models**
*(Note: The fine-tuned BERT transformer weights are excluded from this repository due to GitHub storage limits. You can regenerate them locally.)*

```bash
python3 src/preprocess.py
python3 src/train_baseline.py
python3 src/train_bert.py      

```

**4. Launch the FastAPI Platform**

```bash
python3 api.py

```

**Access the dashboard:** Open your browser and navigate to `http://localhost:8000`.

---

## Clinical Biomarkers Utilized

The system analyzes the following features to generate predictions:

* **Age & Gender**
* **Education Level** (Educ)
* **Socioeconomic Status** (SES)
* **Mini-Mental State Examination** (MMSE)
* **Estimated Total Intracranial Volume** (eTIV)
* **Normalized Whole Brain Volume** (nWBV)
* **Atlas Scaling Factor** (ASF)

---

*Disclaimer: This tool is for research and portfolio demonstration purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.*

```

```
