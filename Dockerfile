# 1. Usamos una imagen base ligera de Python 3.11
FROM python:3.11-slim

# 2. Evitamos que Python genere archivos .pyc y buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Copiamos los requisitos e instalamos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiamos todo el código del proyecto al contenedor
COPY . .

# 6. Exponemos el puerto 8000 (donde corre FastAPI)
EXPOSE 8000

# 7. Comando para iniciar la API al encender el contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]