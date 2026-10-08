# api.py
import warnings
warnings.filterwarnings("ignore")
import sys, io
# Fix Windows console encoding for non-ASCII characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import joblib
import time
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from src.explainability import get_shap_plot_base64

app = FastAPI(
    title="NeuroAI Alzheimer's Diagnostic Platform",
    description="Explainable Multi-Model AI for Cognitive Impairment Detection",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load all artifacts at startup ──────────────────────────────────────────────
scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
le            = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))

AVAILABLE_MODELS = {
    "Random_Forest":       joblib.load(os.path.join(MODELS_DIR, "Random_Forest.joblib")),
    "XGBoost":             joblib.load(os.path.join(MODELS_DIR, "XGBoost.joblib")),
    "Logistic_Regression": joblib.load(os.path.join(MODELS_DIR, "Logistic_Regression.joblib")),
    "SVM_RBF":             joblib.load(os.path.join(MODELS_DIR, "SVM_RBF.joblib")),
}

MODEL_METADATA = {
    "Random_Forest":       {"accuracy": 85.23, "f1": 0.8523, "type": "Ensemble"},
    "XGBoost":             {"accuracy": 85.23, "f1": 0.8368, "type": "Gradient Boosting"},
    "Logistic_Regression": {"accuracy": 82.95, "f1": 0.8228, "type": "Linear"},
    "SVM_RBF":             {"accuracy": 79.55, "f1": 0.7934, "type": "Kernel"},
}

print(f"[OK] NeuroAI: Loaded {len(AVAILABLE_MODELS)} models successfully.")


# ── Schema ─────────────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    Age:        float = Field(..., ge=0, le=120, description="Patient age in years")
    Educ:       float = Field(..., ge=1, le=5,   description="Education level 1-5")
    SES:        float = Field(..., ge=1, le=5,   description="Socioeconomic status 1-5")
    MMSE:       float = Field(..., ge=0, le=30,  description="Mini-Mental State Exam score")
    eTIV:       float = Field(..., description="Estimated Total Intracranial Volume")
    nWBV:       float = Field(..., description="Normalized Whole Brain Volume")
    ASF:        float = Field(..., description="Atlas Scaling Factor")
    Gender_Male: int  = Field(..., ge=0, le=1,   description="Gender: 1=Male, 0=Female")
    model:      Optional[str] = Field("Random_Forest", description="Model key to use for prediction")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/predict")
async def predict(data: PatientData):
    t_start = time.time()

    # Validate model selection
    model_key = data.model if data.model in AVAILABLE_MODELS else "Random_Forest"
    model = AVAILABLE_MODELS[model_key]

    # Build feature DataFrame
    input_dict = {
        "Age":  data.Age,
        "Educ": data.Educ,
        "SES":  data.SES,
        "MMSE": data.MMSE,
        "eTIV": data.eTIV,
        "nWBV": data.nWBV,
        "ASF":  data.ASF,
        "M/F_M": data.Gender_Male,
    }
    input_df = pd.DataFrame([input_dict])

    # Scale and predict
    scaled_input = scaler.transform(input_df)
    raw_pred     = model.predict(scaled_input)[0]

    # Decode label
    if isinstance(raw_pred, str):
        diagnosis = raw_pred
    else:
        diagnosis = le.inverse_transform([int(raw_pred)])[0]

    # Probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(scaled_input)[0]
        prob_dict = {
            str(le.classes_[i]): round(float(probs[i]) * 100, 2)
            for i in range(len(le.classes_))
        }
    else:
        # SVM without probability — decision function fallback
        prob_dict = {diagnosis: 100.0}

    # XAI — SHAP
    shap_b64 = get_shap_plot_base64(input_df)

    latency_ms = round((time.time() - t_start) * 1000, 1)

    return {
        "diagnosis":    diagnosis,
        "probabilities": prob_dict,
        "shap_image":   shap_b64,
        "model_used":   model_key,
        "model_meta":   MODEL_METADATA.get(model_key, {}),
        "latency_ms":   latency_ms,
    }


@app.get("/models")
async def list_models():
    """Return metadata for all available models."""
    return {
        "models": [
            {"key": k, **v}
            for k, v in MODEL_METADATA.items()
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": list(AVAILABLE_MODELS.keys()),
        "version": "2.0.0",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)