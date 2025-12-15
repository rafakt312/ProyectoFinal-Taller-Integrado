import pandas as pd
import numpy as np
import glob
import os

# --- CONFIGURACIÓN ---
WINDOW_SECONDS = 2
FREQ = 50  # Hz
WINDOW_SIZE = WINDOW_SECONDS * FREQ  # 100 muestras
OVERLAP_PERCENT = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENT)) # 50 muestras de avance

# Rutas
RAW_PATH = "data_raw/"
OUTPUT_FILE = "data_processed/dataset_features.csv"

# Crear carpeta de salida si no existe
os.makedirs("data_processed", exist_ok=True)

def calcular_magnitud(df, col_x, col_y, col_z):
    """Calcula la magnitud vectorial de 3 ejes"""
    return np.sqrt(df[col_x]**2 + df[col_y]**2 + df[col_z]**2)

def extraer_features(ventana_df):
    """
    Colapsa 100 filas (una ventana) en 1 sola fila con estadísticas.
    """
    features = {}
    for col in ventana_df.columns:
        datos = ventana_df[col].values
        features[f"{col}_mean"] = np.mean(datos)
        features[f"{col}_std"] = np.std(datos)
        features[f"{col}_max"] = np.max(datos)
        features[f"{col}_min"] = np.min(datos)
        # Puedes agregar más stats si quieres (kurtosis, skew, etc.)
    return features

def procesar_sujeto(filepath, sujeto_id):
    print(f"--> Procesando Sujeto {sujeto_id}...")
    
    # Cargar archivo LOG (MHealth no tiene headers)
    # Asumimos 23 columnas de sensores + etiqueta (MHealth standard)
    try:
        df = pd.read_csv(filepath, sep="\t", header=None)
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return pd.DataFrame()

    # 1. GENERAR MAGNITUDES (Ingeniería de datos)
    # Indices aproximados MHealth:
    # 0-2: Accel Pecho, 5-7: Accel Tobillo, 14-16: Accel Brazo
    # Ajustamos nombres de columnas para trabajar mejor
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]
    
    # Crear columnas sintéticas (Magnitudes)
    # Nota: Asegúrate de que los índices coincidan con tu dataset real
    df['mag_pecho'] = calcular_magnitud(df, df.columns[0], df.columns[1], df.columns[2])
    df['mag_tobillo'] = calcular_magnitud(df, df.columns[5], df.columns[6], df.columns[7])
    df['mag_brazo'] = calcular_magnitud(df, df.columns[14], df.columns[15], df.columns[16])
    
    # La etiqueta suele ser la última columna (índice 23 en original, ahora desplazado)
    # Buscamos la columna original de etiqueta (columna 23)
    col_etiqueta = "sensor_23" 
    
    dataset_ventanas = []

    # 2. VENTANEO DESLIZANTE (Sliding Window)
    # Recorremos el archivo dando saltos de 50 muestras (1 seg)
    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        ventana = df.iloc[start:end]
        
        # Obtener la etiqueta mayoritaria (Moda)
        labels = ventana[col_etiqueta]
        moda_label = labels.mode()[0]
        
        # Ignorar clase 0 (Null/Reposon sin actividad definida)
        if moda_label == 0:
            continue
            
        # 3. EXTRAER CARACTERÍSTICAS
        # Quitamos la etiqueta para calcular estadísticas
        datos_sensores = ventana.drop(columns=[col_etiqueta])
        
        fila_features = extraer_features(datos_sensores)
        
        # Agregamos meta-datos
        fila_features['sujeto'] = sujeto_id
        fila_features['label'] = moda_label
        
        dataset_ventanas.append(fila_features)
        
    return pd.DataFrame(dataset_ventanas)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    todos_los_datos = []
    # Busca archivos que terminen en .log en la carpeta data_raw
    archivos = glob.glob(os.path.join(RAW_PATH, "*.log"))
    
    if not archivos:
        print("¡ERROR! No encontré archivos .log en 'data_raw/'. Por favor pon los datos ahí.")
    else:
        for archivo in archivos:
            # Extraer ID del sujeto del nombre (ej: mhealth_subject1.log -> 1)
            try:
                nombre = os.path.basename(archivo)
                sujeto_id = int(''.join(filter(str.isdigit, nombre)))
            except:
                sujeto_id = 99
                
            df_sujeto = procesar_sujeto(archivo, sujeto_id)
            if not df_sujeto.empty:
                todos_los_datos.append(df_sujeto)

        if todos_los_datos:
            df_final = pd.concat(todos_los_datos, ignore_index=True)
            df_final.to_csv(OUTPUT_FILE, index=False)
            print(f"\n✅ ÉXITO: Dataset guardado en {OUTPUT_FILE}")
            print(f"Dimensiones finales: {df_final.shape}")
        else:
            print("No se generaron datos.")