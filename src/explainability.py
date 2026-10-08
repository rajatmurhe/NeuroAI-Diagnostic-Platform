# src/explainability.py
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for server use
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import io
import base64

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

# Cache artifacts in memory so we don't re-load on every call
_rf_model      = None
_scaler        = None
_feature_names = None
_le            = None

def _load_artifacts():
    global _rf_model, _scaler, _feature_names, _le
    if _rf_model is None:
        _rf_model      = joblib.load(os.path.join(MODELS_DIR, "Random_Forest.joblib"))
        _scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        _feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.joblib"))
        _le            = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))


def get_shap_plot_base64(input_df):
    """
    Generate a feature attribution plot for the given input row.
    Uses tree SHAP if available, otherwise falls back gracefully to
    local tree feature attribution based on standard deviations and importances.
    Returns a base64-encoded PNG string styled for the dark UI.
    """
    _load_artifacts()

    scaled_input = _scaler.transform(input_df)
    raw_pred    = _rf_model.predict(scaled_input)[0]
    class_idx   = (
        list(_le.classes_).index(raw_pred)
        if isinstance(raw_pred, str)
        else int(raw_pred)
    )

    vals = None
    try:
        import shap
        explainer = shap.TreeExplainer(_rf_model)
        shap_values = explainer.shap_values(scaled_input)
        if isinstance(shap_values, list):
            vals = shap_values[class_idx][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, class_idx]
        else:
            vals = shap_values[0]
    except Exception:
        # Fallback attribution when SHAP/numba C-bindings are blocked by OS policy
        importances = _rf_model.feature_importances_
        # Direction determined by scaled deviations from mean (scaled_input is standard scaled)
        dev = scaled_input[0]
        vals = dev * importances

    # ── Dark-theme styling ───────────────────────────────────────────────────
    BG       = "#0d1526"
    FG       = "#e2e8f0"
    GRID     = "#1e2d45"
    POS_COL  = "#3b82f6"   # blue  → pushes toward diagnosis
    NEG_COL  = "#f43f5e"   # rose  → pushes away

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    feature_labels = _feature_names if _feature_names is not None else input_df.columns.tolist()
    n = len(vals)

    # Sort by absolute contribution value
    order   = np.argsort(np.abs(vals))
    s_vals  = vals[order]
    s_names = [feature_labels[i] for i in order]
    s_data  = input_df.iloc[0].values[order]

    colors = [POS_COL if v > 0 else NEG_COL for v in s_vals]
    y_pos  = np.arange(n)

    bars = ax.barh(y_pos, s_vals, color=colors, height=0.6,
                   edgecolor="none", linewidth=0)

    # Add glow effect on bars
    for bar, col in zip(bars, colors):
        bar.set_path_effects([
            pe.withSimplePatchShadow(
                offset=(0, 0), shadow_rgbFace=col, alpha=0.25, rho=1.8
            )
        ])

    # Value labels on bars
    for i, (v, label) in enumerate(zip(s_vals, s_data)):
        ha  = "left" if v > 0 else "right"
        off = 0.003
        sign = "+" if v > 0 else ""
        ax.text(v + (off if v > 0 else -off), i,
                f"{sign}{v:.3f}  [{label:.2f}]",
                va="center", ha=ha,
                color=FG, fontsize=8.5, fontweight="500")

    # Feature names on y-axis
    ax.set_yticks(y_pos)
    ax.set_yticklabels(s_names, color=FG, fontsize=9.5)
    ax.tick_params(axis="x", colors="#475569", labelsize=8)

    ax.axvline(0, color="#334155", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xlabel("Attribution Value (Impact on prediction)", color="#64748b", fontsize=8.5, labelpad=8)

    diag_name = _le.classes_[class_idx] if class_idx < len(_le.classes_) else "Unknown"
    ax.set_title(
        f"Feature Attribution Waterfall — Target: {diag_name}",
        color=FG, fontsize=11, fontweight="bold", pad=14
    )

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linestyle="--", linewidth=0.6, alpha=0.8)
    ax.yaxis.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=POS_COL, label="Increases risk contribution"),
        Patch(facecolor=NEG_COL, label="Protective / Decreases risk contribution"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.15,
              facecolor=BG, edgecolor="#334155", labelcolor=FG, fontsize=8)

    plt.tight_layout(pad=1.5)

    # ── Encode to base64 ─────────────────────────────────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")