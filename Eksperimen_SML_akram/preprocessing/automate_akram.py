import pandas as pd
import os

def run_preprocessing(raw_data_path, output_dir):
    print(f"Memuat dataset dari: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    
    # 1. Menghapus kolom yang tidak relevan untuk pemodelan
    columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # 2. Mengatasi missing value (menghapus baris yang kosong)
    df = df.dropna()
    
    # 3. Encoding Kategorikal: Geography dan Gender
    # Menggunakan one-hot encoding dan membuang kolom pertama untuk menghindari dummy variable trap
    df = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)
    
    # 4. Memastikan folder output tersedia
    os.makedirs(output_dir, exist_ok=True)
    
    # 5. Menyimpan data hasil preprocessing
    output_path = os.path.join(output_dir, "churn_dataset_preprocessing.csv")
    df.to_csv(output_path, index=False)
    print(f"Preprocessing selesai. Data disimpan di: {output_path}")
    
    return df

if __name__ == "__main__":
    # Path berdasarkan lokasi script ini
    base_dir = os.path.dirname(os.path.abspath(__file__))
    RAW_PATH = os.path.join(base_dir, "../churn_dataset_raw/Churn_Modelling.csv")
    OUTPUT_DIR = os.path.join(base_dir, "namadataset_preprocessing")
    
    # Menjalankan fungsi preprocessing
    run_preprocessing(RAW_PATH, OUTPUT_DIR)
