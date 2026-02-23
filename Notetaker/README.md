# Cosmos Notetaker - Paquete de ProducciÃ³n

Este es el paquete de producciÃ³n del proyecto Cosmos Notetaker.

## Estructura

\\\
backend/
â”œâ”€â”€ app/              # CÃ³digo fuente de la aplicaciÃ³n
â”œâ”€â”€ alembic/          # Migraciones de base de datos
â”œâ”€â”€ alembic.ini       # ConfiguraciÃ³n de Alembic
â”œâ”€â”€ requirements.txt  # Dependencias Python
â”œâ”€â”€ env.example       # Template de variables de entorno
â””â”€â”€ create_tables.py  # Script para crear tablas inicialmente
\\\

## InstalaciÃ³n

Consulta el archivo INSTALACION_PRODUCCION.md para instrucciones detalladas.

## Pasos RÃ¡pidos

1. Descomprimir este ZIP en el servidor
2. Crear archivo .env desde env.example
3. Instalar dependencias: pip install -r requirements.txt
4. Ejecutar create_tables.py para crear tablas
5. Iniciar servidor: uvicorn app.main:app --host 0.0.0.0 --port 7000

Para mÃ¡s detalles, ver INSTALACION_PRODUCCION.md
