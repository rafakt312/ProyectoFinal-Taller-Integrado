from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import onnxruntime as rt
import numpy as np
import pandas as pd
import io
from typing import List

# --- CONFIGURACIÓN ---
app = FastAPI(title="Reconocimiento Humano Pro", version="6.0")
MODEL_PATH = "models/actividad_humana.onnx"

WINDOW_SECONDS = 2
FREQ = 50
WINDOW_SIZE = WINDOW_SECONDS * FREQ
OVERLAP_PERCENT = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENT))

ACTIVIDADES = {
    1: "De pie", 2: "Sentado", 3: "Acostado", 4: "Caminando", 
    5: "Subir Esc.", 6: "Doblar Cintura", 7: "Brazos arriba", 
    8: "Agacharse", 9: "Ciclismo", 10: "Trotar", 11: "Correr", 12: "Saltar"
}

sess = None
input_name = None
label_name = None
proba_name = None

# --- HTML FRONTEND AVANZADO ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Analizador Forense de Actividad</title>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <style>
        body { font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; padding: 20px; background-color: #f0f2f5; color: #333; }
        .container { max-width: 1100px; margin: 0 auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
        h1 { text-align: center; color: #2c3e50; margin-bottom: 10px; }
        
        .upload-section { background: #eef2f7; border: 2px dashed #bdc3c7; border-radius: 10px; padding: 30px; text-align: center; transition: all 0.3s; }
        .upload-section:hover { border-color: #3498db; background: #e8f4fc; }
        
        button { background-color: #3498db; color: white; padding: 12px 25px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600; margin-top: 15px; }
        button:hover { background-color: #2980b9; }
        
        /* Ajustamos la altura para que sea una barra compacta */
        #timeline_chart { height: 200px; width: 100%; margin-top: 30px; }
        
        .stats-section { margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 0.9em; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f8f9fa; }
        .total-row { font-weight: bold; background-color: #eef2f7; }

        /* Tooltip estilos */
        .custom-tooltip { padding: 10px; border: 1px solid #ccc; background-color: #fff; min-width: 150px; z-index: 1000; }
        .tooltip-title { font-weight: bold; border-bottom: 1px solid #eee; margin-bottom: 5px; color: #333; }
        .tooltip-warning { color: #e74c3c; font-weight: bold; font-size: 0.9em; margin-top: 5px; border-top: 1px dotted #ccc; padding-top: 3px; }
        
        .loading { display: none; margin-top: 15px; color: #7f8c8d; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏃‍♂️ Línea de Tiempo Continua</h1>
        
        <div class="upload-section">
            <p>Sube tu archivo <strong>.log</strong>:</p>
            <input type="file" id="logFile" accept=".log,.txt">
            <br>
            <button onclick="procesarArchivo()">Analizar</button>
            <p id="loadingMsg" class="loading">Procesando...</p>
        </div>

        <div id="timeline_chart"></div>

        <div id="statsSection" class="stats-section" style="display:none;">
            <h3>Resumen Estadístico</h3>
            <table id="statsTable">
                <thead>
                    <tr><th>Actividad</th><th>Tiempo Total</th><th>%</th><th>Confianza Media</th></tr>
                </thead>
                <tbody></tbody>
            </table>
        </div>
    </div>

    <script>
        google.charts.load('current', {'packages':['timeline']});

        async function procesarArchivo() {
            const fileInput = document.getElementById('logFile');
            const loading = document.getElementById('loadingMsg');
            
            if (fileInput.files.length === 0) { alert("Sube un archivo"); return; }
            
            loading.style.display = 'block';
            document.getElementById('timeline_chart').innerHTML = '';
            document.getElementById('statsSection').style.display = 'none';

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/procesar_log_completo', { method: 'POST', body: formData });
                if (!response.ok) throw new Error("Error API");
                const data = await response.json();
                
                dibujarGrafica(data.timeline);
                renderizarTabla(data.estadisticas);
            } catch (e) { alert(e); } 
            finally { loading.style.display = 'none'; }
        }

        function dibujarGrafica(timelineData) {
            const container = document.getElementById('timeline_chart');
            const chart = new google.visualization.Timeline(container);
            const dataTable = new google.visualization.DataTable();

            // 1. Row Label (Fila), 2. Bar Label (Nombre Actividad), 3. Tooltip, 4. Inicio, 5. Fin
            dataTable.addColumn({ type: 'string', id: 'Track' });
            dataTable.addColumn({ type: 'string', id: 'Actividad' });
            dataTable.addColumn({ type: 'string', role: 'tooltip', p: {html: true} });
            dataTable.addColumn({ type: 'date', id: 'Start' });
            dataTable.addColumn({ type: 'date', id: 'End' });

            const rows = timelineData.map(item => {
                // Tooltip HTML Avanzado
                let tooltipHtml = `
                    <div class="custom-tooltip">
                        <div class="tooltip-title">${item.actividad}</div>
                        <div>⏱ ${item.inicio.toFixed(1)}s - ${item.fin.toFixed(1)}s</div>
                        <div>Confianza: <strong>${(item.confianza * 100).toFixed(1)}%</strong></div>
                `;
                if (item.confianza < 0.80 && item.confundido_con) {
                    tooltipHtml += `<div class="tooltip-warning">⚠️ Posible confusión con:<br>${item.confundido_con} (${(item.prob_confusion*100).toFixed(0)}%)</div>`;
                }
                tooltipHtml += `</div>`;

                return [
                    'Sujeto 1', // <--- TRUCO: Todo va a la misma fila "Sujeto 1"
                    item.actividad, // Esto define el color y el texto de la barra
                    tooltipHtml,
                    new Date(0,0,0,0,0, Math.floor(item.inicio), (item.inicio%1)*1000),
                    new Date(0,0,0,0,0, Math.floor(item.fin), (item.fin%1)*1000)
                ];
            });

            dataTable.addRows(rows);
            
            // Opciones limpias para una sola barra
            const options = {
                tooltip: { isHtml: true },
                timeline: { 
                    showRowLabels: false, // Ocultamos la etiqueta "Sujeto 1" para que se vea limpio
                    groupByRowLabel: true
                },
                backgroundColor: '#fff',
                height: 150 // Altura fija pequeña porque es solo una barra
            };

            chart.draw(dataTable, options);
        }

        function renderizarTabla(stats) {
            const tbody = document.querySelector("#statsTable tbody");
            document.getElementById("statsSection").style.display = 'block';
            tbody.innerHTML = "";
            let total = 0;
            stats.forEach(s => total += s.duracion_total);

            stats.forEach(s => {
                tbody.innerHTML += `<tr>
                    <td><strong>${s.actividad}</strong></td>
                    <td>${s.duracion_total.toFixed(2)}s</td>
                    <td>${((s.duracion_total/total)*100).toFixed(1)}%</td>
                    <td>${(s.confianza_promedio*100).toFixed(1)}%</td>
                </tr>`;
            });
        }
    </script>
</body>
</html>
"""

# --- FUNCIONES MATEMÁTICAS ---
def extraer_features_fila(ventana_df):
    features = []
    for col in ventana_df.columns:
        datos = ventana_df[col].values
        features.extend([np.mean(datos), np.std(datos), np.max(datos), 
                         np.min(datos), np.median(datos), np.ptp(datos), np.var(datos)])
    return np.array(features, dtype=np.float32)

def calcular_magnitud(df, c1, c2, c3):
    return np.sqrt(df[c1]**2 + df[c2]**2 + df[c3]**2)

# --- ALGORITMO DE INTERVALOS (CON DETECCIÓN DE CONFUSIÓN) ---
def generar_resumen(lista_preds):
    if not lista_preds: return [], []

    timeline = []
    
    # Inicializar primer bloque
    bloque = lista_preds[0].copy()
    bloque["inicio"] = bloque["segundo"]
    bloque["fin"] = bloque["segundo"] + 1
    bloque["confianzas"] = [bloque["confianza"]]
    # Para la confusión, guardamos la del primer elemento del bloque como referencia
    del bloque["segundo"]

    for i in range(1, len(lista_preds)):
        pred = lista_preds[i]
        
        # Si es la misma actividad, extendemos el bloque
        if pred["actividad"] == bloque["actividad"]:
            bloque["fin"] = pred["segundo"] + 1
            bloque["confianzas"].append(pred["confianza"])
            
            # Lógica extra: Si este nuevo segmento tiene una confusión fuerte y el bloque no tenía, actualizamos
            if "confundido_con" in pred and "confundido_con" not in bloque:
                 bloque["confundido_con"] = pred["confundido_con"]
                 bloque["prob_confusion"] = pred["prob_confusion"]
        else:
            # Cerrar bloque anterior
            bloque["confianza"] = np.mean(bloque["confianzas"])
            del bloque["confianzas"]
            timeline.append(bloque)
            
            # Iniciar nuevo bloque
            bloque = pred.copy()
            bloque["inicio"] = bloque["segundo"]
            bloque["fin"] = bloque["segundo"] + 1
            bloque["confianzas"] = [bloque["confianza"]]
            del bloque["segundo"]

    # Cerrar último bloque
    bloque["confianza"] = np.mean(bloque["confianzas"])
    del bloque["confianzas"]
    timeline.append(bloque)

    # --- CALCULAR ESTADÍSTICAS GLOBALES ---
    stats_dict = {}
    for item in timeline:
        nombre = item["actividad"]
        duracion = item["fin"] - item["inicio"]
        
        if nombre not in stats_dict:
            stats_dict[nombre] = {"duracion": 0, "confianza_sum": 0, "count": 0}
        
        stats_dict[nombre]["duracion"] += duracion
        stats_dict[nombre]["confianza_sum"] += (item["confianza"] * duracion) # Ponderado por tiempo
        stats_dict[nombre]["count"] += 1 # Esto es para contar bloques, pero usaremos ponderación

    lista_stats = []
    for nombre, data in stats_dict.items():
        if data["duracion"] > 0:
            avg_conf = data["confianza_sum"] / data["duracion"]
            lista_stats.append({
                "actividad": nombre,
                "duracion_total": data["duracion"],
                "confianza_promedio": avg_conf
            })
    
    # Ordenar por duración descendente
    lista_stats.sort(key=lambda x: x["duracion_total"], reverse=True)

    return timeline, lista_stats

# --- SERVIDOR ---
@app.on_event("startup")
def load_model():
    global sess, input_name, label_name, proba_name
    try:
        sess = rt.InferenceSession(MODEL_PATH)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        proba_name = sess.get_outputs()[1].name
        print("✅ Modelo ONNX (182 features) cargado.")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

@app.get("/", response_class=HTMLResponse)
def home(): return html_content

@app.post("/procesar_log_completo")
async def procesar_log(file: UploadFile = File(...)):
    if sess is None: raise HTTPException(500, "Modelo no cargado")
    
    try:
        contents = await file.read()
        try:
            df = pd.read_csv(io.BytesIO(contents), delim_whitespace=True, header=None)
        except:
            df = pd.read_csv(io.BytesIO(contents), sep=",", header=None)

        # Pre-procesamiento de columnas
        df.columns = [f"sensor_{i}" for i in range(df.shape[1])]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Magnitudes
        df['mag_pecho'] = calcular_magnitud(df, df.columns[0], df.columns[1], df.columns[2])
        df['mag_tobillo'] = calcular_magnitud(df, df.columns[5], df.columns[6], df.columns[7])
        df['mag_brazo'] = calcular_magnitud(df, df.columns[14], df.columns[15], df.columns[16])
        
        # Eliminar etiqueta si existe (columna 23 original)
        cols = [c for c in df.columns if "sensor_23" not in c]
        df_clean = df[cols]

        raw_predictions = []

        # Barrido de Ventanas
        for start in range(0, len(df_clean) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            ventana = df_clean.iloc[start:end]
            if len(ventana) < WINDOW_SIZE: continue

            features = extraer_features_fila(ventana).reshape(1, -1)
            
            # Inferencia ONNX
            pred_onx = sess.run([label_name, proba_name], {input_name: features})
            probs_dict = pred_onx[1][0] # Diccionario {clase: prob}

            # --- LÓGICA DE CONFUSIÓN (TOP 2) ---
            # Ordenamos las clases por probabilidad descendente
            # probs_dict.items() devuelve [(1, 0.01), (4, 0.85), ...]
            clases_ordenadas = sorted(probs_dict, key=probs_dict.get, reverse=True)
            
            top_1_id = clases_ordenadas[0]
            top_1_prob = float(probs_dict[top_1_id])
            
            # Objeto base
            pred_obj = {
                "segundo": round(start / FREQ, 2),
                "actividad": ACTIVIDADES.get(top_1_id, str(top_1_id)),
                "confianza": top_1_prob
            }

            # Si la confianza es baja (< 80%), buscamos el Top 2
            if top_1_prob < 0.80 and len(clases_ordenadas) > 1:
                top_2_id = clases_ordenadas[1]
                top_2_prob = float(probs_dict[top_2_id])
                
                # Agregamos info de confusión al objeto
                pred_obj["confundido_con"] = ACTIVIDADES.get(top_2_id, str(top_2_id))
                pred_obj["prob_confusion"] = top_2_prob

            # Filtro de Pureza (Solo mostrar si > 60% para ver más datos, o 70%)
            if top_1_prob >= 0.60: 
                raw_predictions.append(pred_obj)

        timeline, stats = generar_resumen(raw_predictions)

        return {"archivo": file.filename, "timeline": timeline, "estadisticas": stats}

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(400, f"Error procesando: {str(e)}")