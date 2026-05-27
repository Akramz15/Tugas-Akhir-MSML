import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Setup MLflow Tracking URI lokal (Abaikan jika berjalan di GitHub Actions)
import os
if not os.getenv("GITHUB_ACTIONS"):
    mlflow.set_tracking_uri("http://127.0.0.1:5001/")
mlflow.set_experiment("Membangun_Model_Basic_akram")

# Autolog aktif sesuai instruksi Kriteria 2 (Basic)
mlflow.autolog()

import os

def train_basic_model():
    print("Memuat dataset dari folder namadataset_preprocessing...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../Eksperimen_SML_akram/preprocessing/namadataset_preprocessing/churn_dataset_preprocessing.csv")
    df = pd.read_csv(data_path)
    
    # Memisahkan fitur (X) dan target (y)
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pelatihan dengan MLflow Tracking
    with mlflow.start_run(run_name="basic_rf_run"):
        print("Melatih model RandomForest Basic...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi
        score = model.score(X_test, y_test)
        print(f"Akurasi model pada data test: {score:.4f}")
        print("Proses training basic selesai.")

if __name__ == "__main__":
    train_basic_model()
