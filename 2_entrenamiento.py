import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# Rutas
INPUT_FILE = "data_processed/dataset_features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def entrenar():
    print("Cargando dataset procesado...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("¡ERROR! No encuentro el dataset. Ejecuta primero '1_procesamiento.py'")
        return

    # 1. VALIDACIÓN POR SUJETOS (Leave-Subjects-Out)
    print("Separando datos (Sujetos 8 y 10 para Test)...")
    train_df = df[~df['sujeto'].isin([8, 10])]
    test_df = df[df['sujeto'].isin([8, 10])]

    X_train = train_df.drop(columns=['label', 'sujeto'])
    y_train = train_df['label']
    X_test = test_df.drop(columns=['label', 'sujeto'])
    y_test = test_df['label']

    print(f"Entrenando con {len(train_df)} ventanas. Probando con {len(test_df)} ventanas.")

    # 2. ENTRENAR RANDOM FOREST
    # n_estimators=100 es un buen balance velocidad/precisión
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 3. EVALUACIÓN
    print("\n--- Resultados del Modelo ---")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. EXPORTAR A ONNX
    print("Exportando modelo a formato ONNX...")
    
    # Definimos que la entrada es un array de Floats de tamaño [1, N_features]
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    
    onnx_model = convert_sklearn(rf, initial_types=initial_type)
    
    output_path = os.path.join(MODEL_DIR, "actividad_humana.onnx")
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"✅ ¡Modelo guardado exitosamente en: {output_path}!")

if __name__ == "__main__":
    entrenar()