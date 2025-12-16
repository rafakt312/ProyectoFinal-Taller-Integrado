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
    Colapsa 100 filas (una ventana) en 1 sola fila con 7 estadísticas.
    """
    features = {}
    for col in ventana_df.columns:
        datos = ventana_df[col].values
        # --- LAS 7 ESTADÍSTICAS (Versión 182 Features) ---
        features[f"{col}_mean"] = np.mean(datos)
        features[f"{col}_std"] = np.std(datos)
        features[f"{col}_max"] = np.max(datos)
        features[f"{col}_min"] = np.min(datos)
        features[f"{col}_med"] = np.median(datos)
        features[f"{col}_ptp"] = np.ptp(datos)
        features[f"{col}_var"] = np.var(datos)
    return features

def procesar_sujeto(filepath, sujeto_id):
    print(f"--> Procesando Sujeto {sujeto_id}...")
    
    # Cargar archivo LOG
    try:
        # Intenta leer con tabuladores o espacios (MHealth a veces varia)
        try:
            df = pd.read_csv(filepath, sep="\t", header=None)
        except:
            df = pd.read_csv(filepath, delim_whitespace=True, header=None)
    except Exception as e:
        print(f"Error leyendo {filepath}: {e}")
        return pd.DataFrame()

    # 1. GENERAR MAGNITUDES
    df.columns = [f"sensor_{i}" for i in range(df.shape[1])]
    
    # Asumiendo índices MHealth estándar
    df['mag_pecho'] = calcular_magnitud(df, df.columns[0], df.columns[1], df.columns[2])
    df['mag_tobillo'] = calcular_magnitud(df, df.columns[5], df.columns[6], df.columns[7])
    df['mag_brazo'] = calcular_magnitud(df, df.columns[14], df.columns[15], df.columns[16])
    
    # La etiqueta suele ser la última columna (sensor_23)
    col_etiqueta = "sensor_23" 
    
    dataset_ventanas = []

    # 2. VENTANEO
    for start in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        end = start + WINDOW_SIZE
        ventana = df.iloc[start:end]
        
        if len(ventana) < WINDOW_SIZE: continue

        # Moda de la etiqueta
        labels = ventana[col_etiqueta]
        moda_label = labels.mode()[0]
        
        # Ignorar clase 0 (Null)
        if moda_label == 0:
            continue
            
        # 3. EXTRAER CARACTERÍSTICAS
        datos_sensores = ventana.drop(columns=[col_etiqueta])
        
        fila_features = extraer_features(datos_sensores)
        fila_features['sujeto'] = sujeto_id
        fila_features['label'] = moda_label
        
        dataset_ventanas.append(fila_features)
        
    return pd.DataFrame(dataset_ventanas)

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    todos_los_datos = []
    archivos = glob.glob(os.path.join(RAW_PATH, "*.log"))
    
    if not archivos:
        print("¡ERROR! No encontré archivos .log en 'data_raw/'.")
    else:
        for archivo in archivos:
            try:
                nombre = os.path.basename(archivo)
                # Extraer número del nombre (ej: mhealth_subject2.log -> 2)
                sujeto_id = int(''.join(filter(str.isdigit, nombre)))
            except:
                sujeto_id = 99
            
            # --- FILTRO MAESTRO: SOLO 1 AL 8 ---
            if sujeto_id > 8:
                print(f"🚫 Saltando Sujeto {sujeto_id} (No solicitado para entrenamiento)")
                continue
            
            df_sujeto = procesar_sujeto(archivo, sujeto_id)
            if not df_sujeto.empty:
                todos_los_datos.append(df_sujeto)

        if todos_los_datos:
            df_final = pd.concat(todos_los_datos, ignore_index=True)
            df_final.to_csv(OUTPUT_FILE, index=False)
            print(f"\n✅ ÉXITO: Dataset generado SOLO con sujetos 1-8.")
            print(f"Archivo guardado en: {OUTPUT_FILE}")
            print(f"Dimensiones: {df_final.shape}")
        else:
            print("No se generaron datos. Revisa tus archivos raw.")