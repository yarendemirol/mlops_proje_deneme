import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# PROJE YOL AYARI
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Senin Modül Importların (Doğru metrikler buradan geliyor)
from src.features.build_features import apply_feature_engineering
from src.features.rebalancing import analyze_and_rebalance
from src.evaluation.before_after_analysis import run_before_after_comparison
from src.training.train_model import train_full_pipeline
from src.monitoring.quality_check import run_quality_check  # Yeni: İzleme için


@task(name="Data Preparation")
def prepare_data():
    data_file = os.path.join(BASE_DIR, 'processed_adv_data.csv')
    df = pd.read_csv(data_file)
    return apply_feature_engineering(df)


@task(name="Data Splitting & Rebalancing")
def process_data(df):
    X = df.drop('Clicked on Ad', axis=1)
    y = df['Clicked on Ad']
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    upsampled_train = analyze_and_rebalance(train_df)
    X_train_res = upsampled_train.drop('Clicked on Ad', axis=1)
    y_train_res = upsampled_train['Clicked on Ad']

    # Karşılaştırma Tablosu (Outputta görünür)
    run_before_after_comparison(X_train, y_train, X_train_res, y_train_res, X_test, y_test)

    return X_train_res, y_train_res, X_val, y_val, X_test, y_test


@task(name="Model Training & Logging")
def train_models(X_train_res, y_train_res, X_val, y_val, X_test, y_test):
    # Senin train_full_pipeline fonksiyonun içindeki her şey (Checkpoint, Ensemble) burada çalışır
    rf, xgb, ensemble = train_full_pipeline(X_train_res, y_train_res, X_val, y_val, X_test, y_test)
    return rf, xgb, ensemble


@task(name="Final Performance Table & Charts")
def generate_reports(rf, xgb, ensemble, X_test, y_test):
    results = []
    models_dict = {"Bagging (RF)": rf, "Boosting (XGB)": xgb, "Ensemble (Voting)": ensemble}

    for name, model in models_dict.items():
        p = model.predict(X_test)
        pr = model.predict_proba(X_test)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, p), 2),
            "Precision": round(precision_score(y_test, p), 6),
            "Recall": round(recall_score(y_test, p), 2),
            "F1 Score": round(f1_score(y_test, p), 6),
            "AUC-ROC": round(roc_auc_score(y_test, pr), 4)
        })

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("MLOps Project - Model Performance Table (Corrected)")
    print("=" * 90)
    print(results_df.to_string(index=False, justify='right'))

    # Confusion Matrix
    y_pred_ensemble = ensemble.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_ensemble)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Ensemble (Voting) Confusion Matrix")
    plt.show()
    return results_df


@flow(name="MLOps_Full_Pipeline_Orchestrated")
def main_flow():
    # MLflow Parent Run
    with mlflow.start_run(run_name="Full_MLOps_Pipeline_Flow"):
        # 1. Veri Hazırlığı
        df = prepare_data()

        # 2. Rebalancing
        X_tr_res, y_tr_res, X_v, y_v, X_te, y_te = process_data(df)

        # 3. Model Eğitimi (Checkpoint ve Ensemble dahil)
        rf, xgb, ensemble = train_models(X_tr_res, y_tr_res, X_v, y_v, X_te, y_te)

        # 4. Raporlama
        generate_reports(rf, xgb, ensemble, X_te, y_te)

        # 5. Monitoring & Fallback (CME Pattern)
        y_pred = ensemble.predict(X_te)
        status = run_quality_check(y_te, y_pred)  # Bu adım senin koduna eklenen resilient parçadır

        if status == "TRIGGER_FALLBACK":
            print("[FALLBACK] Model performansı kritik seviyenin altında!")


if __name__ == "__main__":
    main_flow()