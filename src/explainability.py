# src/explainability.py
import os
import joblib
import shap
import matplotlib.pyplot as plt
import io
import base64

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

def get_shap_plot_base64(input_df):
    rf_model = joblib.load(os.path.join(MODELS_DIR, "Random_Forest.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    
    scaled_input = scaler.transform(input_df)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(scaled_input)
    
    pred_class_idx = rf_model.predict(scaled_input)[0]
    class_idx = list(le.classes_).index(pred_class_idx) if isinstance(pred_class_idx, str) else pred_class_idx
        
    # Formatting for dark mode UI
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('#1e293b') # Tailwind slate-800
    ax.set_facecolor('#1e293b')
    
    if isinstance(shap_values, list):
        vals = shap_values[class_idx][0]
        base = explainer.expected_value[class_idx]
    else:
        vals = shap_values[0, :, class_idx] if len(shap_values.shape) == 3 else shap_values[0]
        base = explainer.expected_value[class_idx] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value
        
    shap.plots.waterfall(
        shap.Explanation(values=vals, base_values=base, data=input_df.iloc[0], feature_names=feature_names),
        show=False
    )
    
    plt.tight_layout()
    
    # Convert plot to Base64 image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")