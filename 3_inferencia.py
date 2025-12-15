import onnxruntime as rt
import numpy as np
import pandas as pd

# Configuración
MODEL_PATH = "models/actividad_humana.onnx"
DATA_PATH = "data_processed/dataset_features.csv"
UMBRAL_PUREZA = 0.70  # Confianza mínima requerida (70%)

def simular_produccion():
    print("Iniciando motor de inferencia ONNX...")
    
    # 1. Cargar el "Cerebro" (Modelo ONNX)
    try:
        sess = rt.InferenceSession(MODEL_PATH)
    except Exception as e:
        print(f"Error cargando modelo: {e}. ¿Ejecutaste el paso 2?")
        return

    # Obtener nombres de entrada/salida del modelo
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    proba_name = sess.get_outputs()[1].name

    # 2. Tomar datos de prueba (Simulando sensores en tiempo real)
    try:
        df = pd.read_csv(DATA_PATH)
        # Tomamos 5 muestras aleatorias de los sujetos de prueba
        muestras = df[df['sujeto'].isin([8, 10])].sample(5)
    except:
        print("No hay datos procesados. Usando datos dummy.")
        return

    # Preparar datos (Quitar etiquetas y convertir a float32 para ONNX)
    X_input = muestras.drop(columns=['label', 'sujeto']).values.astype(np.float32)
    etiquetas_reales = muestras['label'].values

    # 3. PREDICCIÓN (Inferencia)
    # ONNX devuelve [lista_predicciones, lista_diccionarios_probabilidades]
    predicciones = sess.run([label_name, proba_name], {input_name: X_input})
    
    preds_clase = predicciones[0]
    preds_probs = predicciones[1]

    # 4. RESULTADOS CON FILTRO DE PUREZA
    print("\n--- SIMULACIÓN TIMELINE (En Vivo) ---")
    actividades_map = {1: "De pie", 2: "Sentado", 3: "Acostado", 4: "Caminando", 
                       5: "Subir Esc.", 6: "Doblar Cintura", 7: "Brazos arriba", 
                       8: "Agacharse", 9: "Ciclismo", 10: "Trotar", 11: "Correr", 12: "Saltar"}

    for i in range(len(preds_clase)):
        clase_predicha = preds_clase[i]
        prob_dict = preds_probs[i]
        
        # Obtener confianza de la clase ganadora
        confianza = prob_dict[clase_predicha]
        nombre_actividad = actividades_map.get(clase_predicha, f"Actividad {clase_predicha}")
        
        real = actividades_map.get(etiquetas_reales[i], "Desconocido")

        # Lógica de Negocio (El Filtro)
        if confianza >= UMBRAL_PUREZA:
            status = "✅ VALIDADO"
            accion = f"Mostrar '{nombre_actividad}' en pantalla"
        else:
            status = "⚠️ DESCARTADO"
            accion = "Ignorar (Ruido/Transición)"

        print(f"Ventana {i+1} | Real: {real} -> Pred: {nombre_actividad} ({confianza:.2%}) | {status}")

if __name__ == "__main__":
    simular_produccion()