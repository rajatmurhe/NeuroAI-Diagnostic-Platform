# src/train_baseline.py
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from preprocess import get_train_test_splits

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate_baselines():
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, y_train_enc, y_test_enc, le = get_train_test_splits()
    
    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "SVM_RBF": SVC(kernel='rbf', probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=42)
    }
    
    print("\n--- Training Baseline Models ---")
    for name, model in models.items():
        if name == "XGBoost":
            model.fit(X_train_scaled, y_train_enc)
            y_pred = model.predict(X_test_scaled)
            y_pred_labels = le.inverse_transform(y_pred)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred_labels = model.predict(X_test_scaled)
            
        acc = accuracy_score(y_test, y_pred_labels)
        f1_macro = f1_score(y_test, y_pred_labels, average="macro")
        f1_weighted = f1_score(y_test, y_pred_labels, average="weighted")
        
        print(f"\nModel: {name}")
        print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}")
        print(classification_report(y_test, y_pred_labels, target_names=le.classes_))
        
        # Save model
        joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.joblib"))
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred_labels, labels=le.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.savefig(os.path.join(RESULTS_DIR, f"cm_{name}.png"), bbox_inches="tight")
        plt.close()
        
    print("\nBaseline training complete. Artifacts saved in models/ and results/.")

if __name__ == "__main__":
    train_and_evaluate_baselines()