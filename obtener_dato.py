import pandas as pd
import json

# Cargar tu dataset
df = pd.read_csv("data_processed/dataset_features.csv")

# Tomar una fila cualquiera (ej: la fila 100)
# Drop 'label' y 'sujeto' porque el modelo solo quiere features numéricas
fila = df.iloc[100].drop(['label', 'sujeto'])

# Convertir a lista
lista_datos = fila.values.tolist()

# Crear el JSON exacto que pide FastAPI
json_output = {"features": lista_datos}

print("\n--- COPIA LO DE ABAJO ---")
print(json.dumps(json_output))
print("--- FIN DE LA COPIA ---\n")