import os
import sys
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# PROJE YOL AYARI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Modül Importları
from src.features.build_features import apply_feature_engineering
from src.features.rebalancing import analyze_and_rebalance
from src.training.train_model import train_full_pipeline
from src.monitoring.quality_check import run_quality_check


# --- TASKS ---

@task(name="Data Loading & Advanced Engineering")
def prepare_data():
    """Döküman III.1: Data Representation & High-Cardinality Handling [cite: 28, 31, 32]"""
    data_file = os.path.join(BASE_DIR, 'processed_adv_data.csv')
    df = pd.read_csv(data_file)
    # Feature Cross ve Hashed Feature işlemleri bu fonksiyonun içinde
    df = apply_feature_engineering(df)
    return df


@task(name="Rebalancing & Data Splitting")
def rebalance_step(df):
    """Döküman III.2: Data Imbalance & Rebalancing Pattern [cite: 38]"""
    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']

    # 80/10/10 Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    upsampled_train = analyze_and_rebalance(train_df)

    X_train_res = upsampled_train.drop('Clicked on Ad', axis=1)
    y_train_res = upsampled_train['Clicked on Ad']
    return X_train_res, y_train_res, X_val, y_val, X_test, y_test


@task(name="Ensemble Training & MLflow Registry")
def train_and_register(X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    """Döküman II.1 & III.2: MLflow Registry, Ensembles & Checkpoints [cite: 17, 37, 40]"""
    mlflow.set_experiment("MLOps_Term_Project_Ad_Click")

    with mlflow.start_run(run_name="Prefect_Orchestrated_Run"):
        # Bagging, Boosting ve Voting Ensemble eğitimi [cite: 37]
        # XGBoost checkpoint kullanımı bu fonksiyonun içinde [cite: 40]
        rf, xgb, ensemble = train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test)

        # MLflow Model Registry Kaydı
        mlflow.sklearn.log_model(
            sk_model=ensemble,
            artifact_path="model",
            registered_model_name="AdClick_Ensemble_Model"
        )

        # Stateless Serving için yerel kayıt [cite: 43]
        save_path = os.path.join(BASE_DIR, "final_deployment_model.pkl")
        joblib.dump(ensemble, save_path)
        return save_path


# --- MAIN FLOW ---

@flow(name="End-to-End Ad Click Pipeline")
def main_flow():
    """Döküman I: MLOps Level 2 Maturity - CI/CD Pipeline Automation """

    # 1. Veri Hazırlığı (Feature Cross & Hashing dahil)
    raw_data = prepare_data()

    # 2. Rebalancing (Upsampling) ve Veri Ayrımı
    X_tr, y_tr, X_v, y_v, X_te, y_te = rebalance_step(raw_data)

    # 3. Model Eğitimi, Checkpoint ve Registry
    model_path = train_and_register(X_tr, y_tr, X_v, y_v, X_te, y_te)

    # 4. Continuous Evaluation (CME) & Algorithmic Fallback
    model = joblib.load(model_path)
    y_pred = model.predict(X_te)

    # Performans düşüşü kontrolü ve fallback stratejisi [cite: 49]
    status = run_quality_check(y_te, y_pred, threshold=0.90)

    if status == "TRIGGER_FALLBACK":
        print("[FALLBACK] Model performansı yetersiz, eski sürüme dönülüyor...")
    else:
        print("[CME] Model kalite testlerinden geçti. Servis güncel.")


if __name__ == "__main__":
    main_flow()