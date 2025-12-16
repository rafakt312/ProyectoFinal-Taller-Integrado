import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import os

# Rutas
INPUT_FILE = "data_processed/dataset_features.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def entrenar():
    print("Cargando dataset (Sujetos 1-8)...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("¡ERROR! Ejecuta primero '1_procesamiento.py'")
        return

    X = df.drop(columns=['label', 'sujeto'])
    y = df['label']

    # Split para validación interna
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Entrenando con {len(X_train)} ventanas...")

    # --- CAMBIO CLAVE: VOLVER A LA CONFIGURACIÓN AGRESIVA ---
    # Quitamos 'min_samples_leaf' y 'max_depth'.
    # Esto permite que el modelo memorice patrones exactos y de 99% de confianza.
    rf = RandomForestClassifier(
        n_estimators=100,      
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 3. EVALUACIÓN
    print("\n--- Métricas de Validación ---")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. EXPORTAR A ONNX
    print("Exportando modelo a formato ONNX...")
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(rf, initial_types=initial_type)
    
    output_path = os.path.join(MODEL_DIR, "actividad_humana.onnx")
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"✅ ¡Modelo de ALTA CONFIANZA guardado en: {output_path}!")

if __name__ == "__main__":
    entrenar()