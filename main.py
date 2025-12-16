from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import onnxruntime as rt
import numpy as np
import pandas as pd
import io
from typing import List, Dict, Any
from collections import deque, Counter
from sklearn.metrics import confusion_matrix

# --- CONFIGURACIÓN ---
app = FastAPI(title="Reconocimiento Humano Final", version="14.0")
MODEL_PATH = "models/actividad_humana.onnx"

WINDOW_SECONDS = 2
FREQ = 50
WINDOW_SIZE = WINDOW_SECONDS * FREQ
OVERLAP_PERCENT = 0.5
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PERCENT))
MIN_SEGUNDOS_BLOQUE = 5.0

# Mapa de actividades
ACTIVIDADES = {
    0: "Null", 1: "De pie", 2: "Sentado", 3: "Acostado", 4: "Caminando", 
    5: "Subir Esc.", 6: "Doblar Cintura", 7: "Brazos arriba", 
    8: "Agacharse", 9: "Ciclismo", 10: "Trotar", 11: "Correr", 12: "Saltar"
}

sess = None
input_name = None
label_name = None
proba_name = None

# --- HTML FRONTEND (SIN DETALLE TÉCNICO) ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Dashboard IA</title>
    <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; padding: 20px; background-color: #f4f6f9; color: #2c3e50; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 5px 20px rgba(0,0,0,0.1); }
        h1 { text-align: center; font-size: 2em; margin-bottom: 10px; }
        
        .upload-section { background: #eef2f7; border: 3px dashed #bdc3c7; border-radius: 12px; padding: 30px; text-align: center; }
        button { background-color: #27ae60; color: white; padding: 15px 40px; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; margin-top: 15px; }
        button:hover { background-color: #219150; }
        
        .section-title { border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 50px; margin-bottom: 20px; color: #34495e; }
        #timeline_chart { height: 600px; width: 100%; }
        
        table { width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 1em; }
        th, td { padding: 10px; text-align: center; border: 1px solid #ddd; }
        th { background-color: #f8f9fa; font-weight: bold; }
        .text-left { text-align: left; }
        
        /* MATRIZ DE CONFUSIÓN */
        .matrix-container { overflow-x: auto; display: flex; justify-content: center; }
        .matrix-cell { font-weight: bold; transition: background 0.2s; width: 50px; height: 40px;}
        .matrix-cell:hover { border: 2px solid #333; }
        .diagonal { border: 2px solid #27ae60; }
        
        .loading { display: none; font-size: 1.5em; color: #7f8c8d; margin-top: 20px; }
        .hidden { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Dashboard de Reconocimiento Humano</h1>
        
        <div class="upload-section">
            <p>Sube tu archivo <strong>.log</strong>:</p>
            <input type="file" id="logFile" accept=".log,.txt">
            <br>
            <button onclick="procesarArchivo()">Analizar Completo</button>
            <p id="loadingMsg" class="loading">🧠 Procesando...</p>
        </div>

        <h2 class="section-title">1. Línea de Tiempo (Sin Huecos)</h2>
        <div id="timeline_chart"></div>

        <div id="matrixSection" class="hidden">
            <h2 class="section-title">2. Matriz de Confusión (Predicción vs Realidad)</h2>
            <div class="matrix-container">
                <table id="confMatrix"></table>
            </div>
        </div>

        <div id="statsSection" class="hidden">
            <h2 class="section-title">3. Resumen de la Sesión</h2>
            <table id="statsTable">
                <thead><tr><th>Actividad</th><th>Tiempo Total</th><th>%</th><th>Confianza Media</th></tr></thead>
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
            document.getElementById('matrixSection').classList.add('hidden');
            document.getElementById('statsSection').classList.add('hidden');

            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            try {
                const response = await fetch('/procesar_log_completo', { method: 'POST', body: formData });
                if (!response.ok) throw new Error("Error API");
                const data = await response.json();
                
                dibujarGraficaSplit(data.timeline);
                renderizarTablaStats(data.estadisticas);
                
                if (data.confusion_matrix) {
                    renderizarMatriz(data.confusion_matrix, data.labels_presentes);
                    document.getElementById('matrixSection').classList.remove('hidden');
                }

            } catch (e) { alert(e); } 
            finally { loading.style.display = 'none'; }
        }

        function dibujarGraficaSplit(timelineData) {
            const container = document.getElementById('timeline_chart');
            const chart = new google.visualization.Timeline(container);
            const dataTable = new google.visualization.DataTable();

            dataTable.addColumn({ type: 'string', id: 'Parte' });
            dataTable.addColumn({ type: 'string', id: 'Actividad' });
            dataTable.addColumn({ type: 'string', role: 'tooltip', p: {html: true} });
            dataTable.addColumn({ type: 'date', id: 'Start' });
            dataTable.addColumn({ type: 'date', id: 'End' });

            if(timelineData.length === 0) { container.innerHTML = "No hay datos."; return; }

            const maxTime = timelineData[timelineData.length - 1].fin;
            const splitPoint = maxTime / 2;
            const rows = [];

            timelineData.forEach(item => {
                let tooltip = `
                    <div style="padding:10px; border:1px solid #ccc; font-family:sans-serif;">
                        <b>${item.actividad}</b><br>
                        ⏱ ${item.inicio.toFixed(1)}s - ${item.fin.toFixed(1)}s<br>
                        Confianza: ${(item.confianza * 100).toFixed(0)}%
                    </div>`;

                if (item.fin <= splitPoint) {
                    rows.push(['1. Primera Mitad', item.actividad, tooltip, date(item.inicio), date(item.fin)]);
                } else if (item.inicio >= splitPoint) {
                    rows.push(['2. Segunda Mitad', item.actividad, tooltip, date(item.inicio - splitPoint), date(item.fin - splitPoint)]);
                } else {
                    rows.push(['1. Primera Mitad', item.actividad, tooltip, date(item.inicio), date(splitPoint)]);
                    rows.push(['2. Segunda Mitad', item.actividad, tooltip, date(0), date(item.fin - splitPoint)]);
                }
            });

            dataTable.addRows(rows);
            chart.draw(dataTable, { timeline: { groupByRowLabel: true }, height: 600, hAxis: {format: 'mm:ss'} });
        }
        function date(sec) { return new Date(0,0,0,0,0, Math.floor(sec), (sec%1)*1000); }

        function renderizarMatriz(matrix, labels) {
            const table = document.getElementById("confMatrix");
            table.innerHTML = "";
            let thead = "<thead><tr><th>Real \\ Pred</th>";
            labels.forEach(l => thead += `<th>${l}</th>`);
            thead += "</tr></thead>";
            table.innerHTML += thead;
            let tbody = "<tbody>";
            matrix.forEach((row, i) => {
                tbody += `<tr><th class="text-left">${labels[i]}</th>`;
                let rowMax = Math.max(...row);
                row.forEach((val, j) => {
                    let color = "white";
                    let isDiag = (i === j);
                    if (val > 0) {
                        let alpha = (val / rowMax).toFixed(2);
                        if (isDiag) color = `rgba(46, 204, 113, ${alpha})`;
                        else color = `rgba(231, 76, 60, ${alpha})`;
                    }
                    tbody += `<td class="matrix-cell ${isDiag ? 'diagonal' : ''}" style="background-color:${color}">${val}</td>`;
                });
                tbody += "</tr>";
            });
            tbody += "</tbody>";
            table.innerHTML += tbody;
        }

        function renderizarTablaStats(stats) {
            const tbody = document.querySelector("#statsTable tbody");
            document.getElementById("statsSection").classList.remove('hidden');
            tbody.innerHTML = "";
            let total = 0; stats.forEach(s => total += s.duracion_total);
            stats.forEach(s => {
                tbody.innerHTML += `<tr>
                    <td class="text-left"><strong>${s.actividad}</strong></td>
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

# --- BACKEND ---
def extraer_features_fila(ventana_df):
    features = []
    for col in ventana_df.columns:
        datos = ventana_df[col].values
        features.extend([np.mean(datos), np.std(datos), np.max(datos), 
                         np.min(datos), np.median(datos), np.ptp(datos), np.var(datos)])
    return np.array(features, dtype=np.float32)

def calcular_magnitud(df, c1, c2, c3):
    return np.sqrt(df[c1]**2 + df[c2]**2 + df[c3]**2)

def generar_resumen(lista_preds):
    if not lista_preds: return []
    timeline = []
    
    bloque = lista_preds[0].copy()
    bloque["inicio"] = bloque["segundo"]
    bloque["fin"] = bloque["segundo"] + 1
    bloque["confianzas"] = [bloque["confianza"]]
    del bloque["segundo"]

    for i in range(1, len(lista_preds)):
        pred = lista_preds[i]
        tiempo_actual = pred["segundo"]
        
        if pred["actividad"] == bloque["actividad"]:
            bloque["fin"] = tiempo_actual + 1
            bloque["confianzas"].append(pred["confianza"])
        else:
            bloque["confianza"] = np.mean(bloque["confianzas"])
            del bloque["confianzas"]
            bloque["fin"] = max(bloque["fin"], tiempo_actual)
            timeline.append(bloque)
            
            bloque = pred.copy()
            bloque["inicio"] = tiempo_actual
            bloque["fin"] = tiempo_actual + 1
            bloque["confianzas"] = [bloque["confianza"]]
            del bloque["segundo"]
            
    bloque["confianza"] = np.mean(bloque["confianzas"])
    del bloque["confianzas"]
    timeline.append(bloque)
    return timeline

def limpiar_rebotes(timeline):
    if not timeline: return []
    timeline_limpio = [timeline[0]]
    for i in range(1, len(timeline)):
        bloque_actual = timeline[i]
        bloque_anterior = timeline_limpio[-1]
        duracion = bloque_actual["fin"] - bloque_actual["inicio"]
        if duracion < MIN_SEGUNDOS_BLOQUE or bloque_actual["actividad"] == bloque_anterior["actividad"]:
            bloque_anterior["fin"] = bloque_actual["fin"]
            bloque_anterior["confianza"] = (bloque_anterior["confianza"] + bloque_actual["confianza"]) / 2
        else:
            timeline_limpio.append(bloque_actual)
    return timeline_limpio

def calcular_estadisticas(timeline):
    stats_dict = {}
    for item in timeline:
        nombre = item["actividad"]
        duracion = item["fin"] - item["inicio"]
        if nombre not in stats_dict: stats_dict[nombre] = {"duracion": 0, "confianza_sum": 0}
        stats_dict[nombre]["duracion"] += duracion
        stats_dict[nombre]["confianza_sum"] += (item["confianza"] * duracion)
    lista_stats = []
    for nombre, data in stats_dict.items():
        if data["duracion"] > 0:
            lista_stats.append({
                "actividad": nombre,
                "duracion_total": data["duracion"],
                "confianza_promedio": data["confianza_sum"] / data["duracion"]
            })
    lista_stats.sort(key=lambda x: x["duracion_total"], reverse=True)
    return lista_stats

@app.on_event("startup")
def load_model():
    global sess, input_name, label_name, proba_name
    try:
        sess = rt.InferenceSession(MODEL_PATH)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        proba_name = sess.get_outputs()[1].name
        print("✅ Modelo ONNX cargado.")
    except Exception as e: print(f"❌ Error: {e}")

@app.get("/", response_class=HTMLResponse)
def home(): return html_content

@app.post("/procesar_log_completo")
async def procesar_log(file: UploadFile = File(...)):
    if sess is None: raise HTTPException(500, "Modelo no cargado")
    try:
        contents = await file.read()
        try: df = pd.read_csv(io.BytesIO(contents), sep='\\s+', header=None)
        except: df = pd.read_csv(io.BytesIO(contents), sep=",", header=None)

        df.columns = [f"sensor_{i}" for i in range(df.shape[1])]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        df['mag_pecho'] = calcular_magnitud(df, df.columns[0], df.columns[1], df.columns[2])
        df['mag_tobillo'] = calcular_magnitud(df, df.columns[5], df.columns[6], df.columns[7])
        df['mag_brazo'] = calcular_magnitud(df, df.columns[14], df.columns[15], df.columns[16])
        
        col_label_idx = 23 
        has_labels = False
        y_true = []
        y_pred = []
        cols = [c for c in df.columns if f"sensor_{col_label_idx}" not in c]
        df_features = df[cols]
        
        pred_buffer = deque(maxlen=50) 
        raw_predictions = []

        for start in range(0, len(df_features) - WINDOW_SIZE, STEP_SIZE):
            end = start + WINDOW_SIZE
            ventana = df_features.iloc[start:end]
            if len(ventana) < WINDOW_SIZE: continue
            
            current_true_label = 0
            if df.shape[1] > col_label_idx:
                try:
                    labels_window = df.iloc[start:end, col_label_idx]
                    current_true_label = int(labels_window.mode()[0])
                    if current_true_label != 0: has_labels = True
                except: pass

            features = extraer_features_fila(ventana).reshape(1, -1)
            pred_onx = sess.run([label_name, proba_name], {input_name: features})
            probs_dict = pred_onx[1][0]
            
            top_1_id = sorted(probs_dict, key=probs_dict.get, reverse=True)[0]
            pred_buffer.append(top_1_id)
            ganador_id, _ = Counter(pred_buffer).most_common(1)[0]
            confianza = float(probs_dict[ganador_id])

            if confianza >= 0.0:
                raw_predictions.append({
                    "segundo": round(start / FREQ, 2),
                    "actividad": ACTIVIDADES.get(ganador_id, str(ganador_id)),
                    "confianza": confianza
                })
                
                if current_true_label != 0:
                    y_true.append(current_true_label)
                    y_pred.append(ganador_id)

        timeline_raw = generar_resumen(raw_predictions)
        timeline_clean = limpiar_rebotes(timeline_raw)
        stats = calcular_estadisticas(timeline_clean)

        matrix_data = None
        labels_str = []
        
        if has_labels and len(y_true) > 0:
            unique_labels = sorted(list(set(y_true) | set(y_pred)))
            labels_str = [ACTIVIDADES.get(l, str(l)) for l in unique_labels]
            cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
            matrix_data = cm.tolist()

        return {
            "timeline": timeline_clean, 
            "estadisticas": stats,
            "confusion_matrix": matrix_data,
            "labels_presentes": labels_str
        }

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(400, f"Error: {str(e)}")