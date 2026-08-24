# api.py
import os
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.explainability import get_shap_plot_base64

app = FastAPI(title="Alzheimer's AI Diagnostic API")
templates = Jinja2Templates(directory="templates")

MODELS_DIR = "models"

# Load models at startup
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
rf_model = joblib.load(os.path.join(MODELS_DIR, "Random_Forest.joblib"))

class PatientData(BaseModel):
    Age: float
    Educ: float
    SES: float
    MMSE: float
    eTIV: float
    nWBV: float
    ASF: float
    Gender_Male: int

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(data: PatientData):
    input_dict = {
        "Age": data.Age, "Educ": data.Educ, "SES": data.SES,
        "MMSE": data.MMSE, "eTIV": data.eTIV, "nWBV": data.nWBV,
        "ASF": data.ASF, "M/F_M": data.Gender_Male
    }
    input_df = pd.DataFrame([input_dict])
    
    # Scale and Predict
    scaled_input = scaler.transform(input_df)
    pred_val = rf_model.predict(scaled_input)[0]
    
    prediction = pred_val if isinstance(pred_val, str) else le.inverse_transform([pred_val])[0]
    probabilities = rf_model.predict_proba(scaled_input)[0]
    
    # Format Probabilities
    prob_dict = {str(le.classes_[i]): round(float(probabilities[i]) * 100, 2) for i in range(len(le.classes_))}
    
    # Generate XAI Plot
    shap_base64 = get_shap_plot_base64(input_df)
    
    return {
        "diagnosis": prediction,
        "probabilities": prob_dict,
        "shap_image": shap_base64
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)