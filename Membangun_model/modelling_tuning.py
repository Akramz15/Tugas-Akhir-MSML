import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow

# Setup MLflow Tracking URI lokal (Abaikan jika dijalankan oleh MLflow CLI atau CI)
import os
if not os.getenv("GITHUB_ACTIONS") and not os.getenv("MLFLOW_RUN_ID"):
    mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment("Membangun_Model_Tuning_akram")

import os

def train_tuned_model():
    print("Memuat dataset dari folder namadataset_preprocessing...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "namadataset_preprocessing/churn_dataset_preprocessing.csv")
    df = pd.read_csv(data_path)
    
    # Memisahkan fitur (X) dan target (y)
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Melakukan Hyperparameter Tuning dengan GridSearchCV...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Prediksi menggunakan model terbaik
    y_pred = best_model.predict(X_test)
    
    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # MLflow Manual Logging sesuai instruksi Kriteria 2 (Skilled)
    with mlflow.start_run(run_name="tuned_rf_run"):
        print("Mencatat hasil ke MLflow secara manual...")
        
        # Log Hyperparameters
        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)
            
        # Log Metrics
        mlflow.log_metric("training_score", grid_search.best_score_)
        mlflow.log_metric("accuracy_score", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision_score", precision)
        mlflow.log_metric("recall_score", recall)
        
        # Log Model Artefak
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"Model berhasil dilatih dan dicatat di MLflow.")
        print(f"Parameter Terbaik: {best_params}")
        print(f"Test Akurasi: {accuracy:.4f}, F1-Score: {f1:.4f}")
    
    # Ekspor model sebagai .pkl untuk disimpan ke cloud/GitHub artifact
    import joblib
    artifact_path = os.path.join(base_dir, "model_artefak.pkl")
    joblib.dump(best_model, artifact_path)
    print(f"File artefak model berhasil disimpan di: {artifact_path}")

if __name__ == "__main__":
    train_tuned_model()
