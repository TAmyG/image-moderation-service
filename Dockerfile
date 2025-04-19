FROM python:3.11-slim

# Instala dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Crea carpeta de trabajo
WORKDIR /app

# Copia los archivos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expone el puerto
EXPOSE 8000

# Comando para iniciar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
