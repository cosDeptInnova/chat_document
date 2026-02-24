# Imagen base con PyTorch 2.2.2, CUDA 11.8, cuDNN, y Python 3.10
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app

# Dependencias del sistema necesarias
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    git \
    tesseract-ocr \
    poppler-utils \
    && apt-get clean

# Actualizar pip y copiar requisitos
COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt

# Dependencias adicionales
RUN pip install qdrant-client python-multipart spacy openpyxl sentence-transformers mammoth prometheus-client starlette
RUN pip install -U scikit-learn

# Descargar modelo de spaCy
RUN python -m spacy download es_core_news_md
RUN pip install numpy==1.26.4 pandas --force-reinstall
# Copiar código de la app
COPY . .

EXPOSE 5000

CMD ["uvicorn", "main_v3_3:app", "--host", "0.0.0.0", "--port", "5000"]
