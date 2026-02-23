# Usa una imagen oficial de Python como base
FROM python:3.10.12

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copiar el archivo requirements.txt a la carpeta de trabajo
COPY requirements.txt .

# Instalar las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install openpyxl prometheus-client starlette sqlalchemy

# Copiar el resto del código de la aplicación a la carpeta de trabajo
COPY . .

# Exponer el puerto en el que la aplicación FastAPI correrá
EXPOSE 8000

# Comando por defecto para correr el servicio FastAPI con Uvicorn
CMD ["uvicorn", "modelo_negocio_v2:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
