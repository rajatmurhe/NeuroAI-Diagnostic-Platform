
# 🧠 NeuroAI: Explainable Clinical Diagnostic Platform

An end-to-end **Artificial Intelligence** diagnostic microservice built to predict cognitive impairment stages (**Cognitively Normal, Mild Cognitive Impairment, Alzheimer’s Disease**) from multimodal patient biomarkers. 

This project tackles the complex challenge of early cognitive decline detection by bridging the gap between traditional **Tabular Machine Learning** and modern **Natural Language Processing (NLP)**. By benchmarking classical **Ensemble Learning** algorithms against a fine-tuned **Transformer Sequence Classifier (BERT)**, this platform delivers highly accurate, interpretable clinical predictions.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![Transformers](https://img.shields.io/badge/Hugging%20Face-Transformers-F7931E?logo=huggingface)
![SHAP](https://img.shields.io/badge/Explainable%20AI-SHAP-green)

---

## 🖥️ Live Diagnostic Dashboard
> *Real-time inference and SHAP cryptographic feature attribution deployed via FastAPI.*

<img width="800" alt="dashboard" src="https://github.com/user-attachments/assets/df6a4e05-dc71-49bb-ba7a-a4095730ebe9" />

---

## ⚙️ Core Methodology & Engineering Pipeline

This project was built from the ground up to demonstrate a complete, production-ready **Machine Learning Lifecycle**, from raw data ingestion to a deployed, explainable web interface.

### 1. Data Preprocessing & Feature Engineering
* Ingested the **OASIS Cross-Sectional MRI and Clinical dataset**.
* Handled missing clinical values (imputation) and normalized continuous brain volume metrics using `StandardScaler` to optimize gradient descent and distance-based calculations.
* Applied **Label Encoding** to transform multi-class clinical targets into machine-readable tensors.

### 2. Tabular-to-Text Serialization (The "Tabular LLM" Approach)
* Engineered a custom serialization pipeline to convert structured tabular rows (e.g., Age, MMSE score, Brain Volume) into coherent text narratives. 
* This technique allows advanced **Deep Learning LLMs** to natively "read" and understand tabular clinical data without losing structural context.

### 3. Model Architecture Benchmarking
Developed and evaluated multiple paradigms of AI to find the optimal diagnostic engine:
* **Tree-Based Ensembles:** Trained **Random Forest** and **XGBoost (Gradient Boosting)** models to capture non-linear relationships in the data.
* **Kernel Methods:** Utilized a **Support Vector Machine (SVM)** with an RBF kernel to draw complex decision boundaries.
* **Deep Learning Transformers:** Fine-tuned `bert-base-uncased` via the **Hugging Face Trainer API**, utilizing Apple Silicon (MPS) hardware acceleration to update neural network weights for sequence classification.

### 4. Explainable AI (XAI) Integration
* Integrated **SHAP (SHapley Additive exPlanations)** to eliminate the "black-box" nature of AI. 
* The system generates real-time cryptographic feature attributions (waterfall plots) that visually prove exactly *how* each specific biomarker pushes the model toward or away from an Alzheimer's diagnosis, ensuring **Clinical Decision Support** transparency.

### 5. Full-Stack Microservice Deployment
* Packaged the serialized models and preprocessing artifacts (`.joblib`) into a high-concurrency **FastAPI Backend Microservice**.
* Built a responsive, dark-mode glassmorphism **Frontend UI** using **Tailwind CSS** and asynchronous JavaScript (`fetch` API) to deliver seamless, real-time diagnostic predictions directly in the browser.

---

## 📊 Performance Benchmarks & Terminal Logs

Evaluated using a highly imbalanced dataset (stratified 80/20 train-test split), prioritizing the **Weighted F1-Score** to accurately measure precision and recall across all cognitive stages.

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
<img width="600" alt="Screenshot 2026-08-24 at 3 13 05 PM" src="https://github.com/user-attachments/assets/b575642a-6987-4c87-95bb-4d7ee7079113" />

**2. XGBoost**
<br>
<img width="600" alt="XGBoost" src="https://github.com/user-attachments/assets/32070283-8553-4b77-ba48-7a1a11a83910" />

**3. Fine-Tuned BERT (Sequence Classifier)**
<br>
<img width="600" alt="Bert" src="https://github.com/user-attachments/assets/e2ee9021-bae2-4fd3-a1d5-c91c8e5cc1e8" />

**4. Support Vector Machine (RBF)**
<br>
<img width="600" alt="SVM training" src="https://github.com/user-attachments/assets/f707c5be-a530-4285-bde9-673d6b5ae0d0" />

---

## 🛠️ Project Architecture & Folder Structure

```text
NeuroAI-Diagnostic-Platform/
│
├── api.py                     # FastAPI server and asynchronous inference endpoints
├── requirements.txt           # Environment dependencies
│
├── data/                      # Clinical datasets (OASIS Cross-Sectional)
│   └── oasis_cross-sectional.csv 
│
├── src/                       # Core ML engineering pipelines
│   ├── preprocess.py          # Data Imputation, Scaling, and NLP text serialization
│   ├── train_baseline.py      # Algorithm training (Random Forest, XGBoost, SVM)
│   ├── train_bert.py          # Hugging Face Trainer script for deep learning fine-tuning
│   └── explainability.py      # SHAP integration and real-time plot generation
│
├── templates/                 # Frontend User Interface
│   └── index.html             # Tailwind CSS & JS asynchronous dashboard
│
└── models/                    # Serialized ML artifacts (Scalers, Encoders, Joblib weights)

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

**3. Train the models (Optional but recommended)**
*(Note: The fine-tuned BERT transformer weights are excluded from this repository due to GitHub storage limits. You can easily regenerate them locally using Apple Silicon or CUDA.)*

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

## 🧬 Clinical Biomarkers Utilized

The system analyzes the following multimodal features to generate predictions:

* **Demographics:** Age & Gender
* **Social Determinants:** Education Level (Educ), Socioeconomic Status (SES)
* **Cognitive Testing:** Mini-Mental State Examination (MMSE) Score
* **Volumetric MRI Data:**
* Estimated Total Intracranial Volume (eTIV)
* Normalized Whole Brain Volume (nWBV)
* Atlas Scaling Factor (ASF)



---

*Disclaimer: This AI tool is built strictly for research, benchmarking, and portfolio demonstration purposes. It is not FDA-approved and should not be used as a substitute for professional medical advice, diagnosis, or treatment.*

```

```
