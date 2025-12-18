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
    print("Cargando dataset...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print("¡ERROR! Falta dataset.")
        return

    X = df.drop(columns=['label', 'sujeto'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print(f"Entrenando modelo AGRESIVO con {len(X_train)} muestras...")

    # --- CAMBIO EXTREMO PARA ALTA CONFIANZA ---
    # 500 árboles, bootstrap=False (usa toda la data), sin límite de profundidad.
    # Esto fuerza probabilidades cercanas al 100%.
    rf = RandomForestClassifier(
        n_estimators=500,      
        criterion='entropy',   # Entropy suele ser más decisivo
        bootstrap=False,       # CRÍTICO: Elimina la aleatoriedad de selección de filas
        max_features=None,     # CRÍTICO: Mira TODAS las columnas para decidir (fuerza pureza)
        random_state=42, 
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    print("\n--- Validación ---")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    print("Exportando ONNX...")
    initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
    onnx_model = convert_sklearn(rf, initial_types=initial_type)
    
    output_path = os.path.join(MODEL_DIR, "actividad_humana.onnx")
    with open(output_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
        
    print(f"✅ MODELO GENERADO: {output_path}")

if __name__ == "__main__":
    entrenar()