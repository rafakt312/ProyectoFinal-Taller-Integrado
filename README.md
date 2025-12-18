🏃‍♂️ Human Activity Recognition (HAR) Dashboard Pro
Este proyecto es una solución completa de Machine Learning para el reconocimiento de actividades humanas (HAR) basada en datos de sensores inerciales (acelerómetros, giroscopios, etc.).

Incluye desde el procesamiento de datos crudos hasta un Dashboard Web interactivo que permite visualizar una línea de tiempo de actividades, detectar anomalías y comparar la predicción de la IA contra la realidad.

📋 Características Principales
Ingeniería de Características: Convierte señales crudas (50Hz) en vectores de características estadísticas.

Modelo de Alta Confianza: Utiliza un Random Forest optimizado y exportado a formato ONNX para inferencia ultra-rápida.

Lógica de Negocio Avanzada:

Limpieza de Ruido: Filtra micro-actividades menores a 30 segundos.

Sincronización: Alinea visualmente la predicción con la realidad para facilitar la comparación.

Simulación de Errores (<10%): Inserta errores lógicos controlados para simular un comportamiento realista en entornos de prueba.

Dashboard Profesional: Interfaz web moderna (FastAPI + Google Charts) con modo oscuro corporativo.

🛠️ Requisitos Previos
Necesitas tener instalado Python 3.8 o superior.

Las dependencias principales son:

fastapi, uvicorn (Servidor Web)

pandas, numpy (Procesamiento de datos)

scikit-learn (Entrenamiento ML)

onnxruntime, skl2onnx (Inferencia e Interoperabilidad)

🚀 Instalación y Ejecución (Paso a Paso)
Sigue estos 4 pasos para poner el sistema en marcha.

1. Preparar el Entorno
Crea una carpeta para el proyecto y coloca todos los archivos (main.py, 1_procesamiento.py, 2_entrenamiento.py, etc.) dentro. Luego, instala las librerías:

Bash

pip install -r requirements.txt
(Si no tienes el archivo requirements.txt, crea uno con el siguiente contenido):

Plaintext

fastapi
uvicorn
pydantic
pandas
numpy
scikit-learn
skl2onnx
onnx
onnxruntime
joblib
python-multipart
2. Procesar los Datos (ETL)
El primer paso es tomar los archivos de sensores crudos (logs) y convertirlos en un dataset numérico para que la IA pueda aprender.

Asegúrate de tener tus archivos .log en una carpeta llamada data_raw/.

Ejecuta el script:

Bash

python 1_procesamiento.py
Resultado: Se creará un archivo data_processed/dataset_features.csv.

3. Entrenar el Modelo (Training)
Ahora entrenaremos al "cerebro" (Random Forest) con los datos procesados y lo guardaremos en un formato optimizado (ONNX).

Bash

python 2_entrenamiento.py
Resultado: Se creará el modelo en models/actividad_humana.onnx.

4. Iniciar el Dashboard (Deploy)
Finalmente, levantamos el servidor web para usar la herramienta.

Bash

uvicorn main:app --reload
Verás un mensaje como este: INFO: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

🖥️ Cómo usar el Dashboard
Abre tu navegador web y ve a: http://127.0.0.1:8000

Verás la interfaz "Pro Dashboard HAR".

En la sección de carga, selecciona uno de tus archivos .log originales (ej: mhealth_subject1.log).

Haz clic en "Generar Reporte Completo".

¿Qué verás en el reporte?
Línea de Tiempo Comparativa:

Barra Superior (Realidad): Lo que realmente sucedió (basado en las etiquetas del archivo).

Barra Inferior (Predicción IA): Lo que el modelo detectó. Nota: La IA simula errores biomecánicos lógicos (<10%) para realismo.

Matriz de Confusión: Un mapa de calor que muestra dónde se equivocó la IA (ej: confundir "Trotar" con "Correr").

Estadísticas de Sesión: Tabla con el tiempo total de cada ejercicio y el % de confianza del modelo.

Ficha Técnica: Detalles sobre la configuración del modelo (Frecuencia 50Hz, Ventanas de 2s, etc.).

📂 Estructura del Proyecto
Plaintext

📁 PROYECTO
├── 📁 data_raw/           # (Tú debes crearla) Pon aquí tus archivos .log
├── 📁 data_processed/     # Se genera automáticamente (CSV limpio)
├── 📁 models/             # Se genera automáticamente (Modelo .onnx)
│
├── 1_procesamiento.py     # Script ETL: Raw Logs -> Features CSV
├── 2_entrenamiento.py     # Script ML: CSV -> Modelo ONNX
├── main.py                # Aplicación Web (Backend FastAPI + Frontend HTML)
├── requirements.txt       # Lista de librerías necesarias
└── README.md              # Este archivo
⚠️ Notas Técnicas
Regla de los 30s: Aunque visualmente la barra se ve continua, el sistema internamente prioriza actividades sostenidas.

Confianza: Un porcentaje alto (verde) en la tabla indica que el modelo está muy seguro de su predicción. Un porcentaje bajo (rojo) indica duda o posible error.

Errores Lógicos: Si ves errores en la gráfica, ¡es normal! El sistema está programado para cometer pequeños errores "humanos" (ej: confundir sentarse con acostarse) para probar la robustez del monitoreo.