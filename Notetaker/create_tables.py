"""
Script temporal para crear tablas de base de datos.
Úsalo solo para desarrollo/testing rápido.

Para producción, usa Alembic migrations.
"""
import sys
import os
from pathlib import Path

# Agregar el directorio actual al path
sys.path.insert(0, os.path.dirname(__file__))

# Cambiar al directorio del script para asegurar que se encuentre el .env
script_dir = Path(__file__).parent.absolute()
os.chdir(script_dir)

# Cargar .env ANTES de importar config
def load_env_manual():
    """Carga .env manualmente antes de que config.py lo intente."""
    from pathlib import Path
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    # Remover comentarios
                    if '#' in value:
                        parts = value.split('#', 1)
                        if parts[0].count('"') % 2 == 0 and parts[0].count("'") % 2 == 0:
                            value = parts[0].strip()
                    # Remover comillas
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    if key and not os.environ.get(key):
                        os.environ[key] = value

# Cargar .env manualmente
load_env_manual()

# Verificar que las dependencias estén instaladas antes de importar
def check_dependencies():
    """Verifica que las dependencias necesarias estén instaladas."""
    missing = []
    try:
        import sqlalchemy
    except ImportError:
        missing.append("sqlalchemy")
    
    try:
        import psycopg2
    except ImportError:
        missing.append("psycopg2-binary")
    
    if missing:
        print("[FAIL] Faltan dependencias requeridas:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nPor favor, ejecuta:")
        print("  1. Activa el venv: venv\\Scripts\\activate")
        print("  2. Instala dependencias: pip install -r requirements.txt")
        sys.exit(1)

check_dependencies()

def create_tables():
    """Crear todas las tablas."""
    try:
        # Cargar configuración mínima para database_url
        # Si no existe .env, intentar usar valores por defecto
        if not os.path.exists('.env'):
            print("[WARN] Archivo .env no encontrado")
            print("       Creando .env desde env.example...")
            if os.path.exists('env.example'):
                import shutil
                shutil.copy('env.example', '.env')
                print("[OK] Archivo .env creado desde env.example")
                print("     IMPORTANTE: Edita .env con tus valores antes de continuar")
                sys.exit(1)
            else:
                print("[FAIL] No se encontró env.example tampoco")
                print("       Por favor, crea un archivo .env con al menos:")
                print("       DATABASE_URL=postgresql://user:password@localhost:5432/cosmos_notetaker")
                sys.exit(1)
        
        # Importar después de asegurar que .env existe
        try:
            from app.database import engine, Base
            from app.models import User, Meeting, Transcription, TranscriptionSegment, Summary
        except Exception as config_error:
            print(f"[FAIL] Error cargando configuración: {config_error}")
            print("\nEl archivo .env necesita ser configurado con todas las variables requeridas.")
            print("Revisa env.example para ver qué variables necesitas configurar.")
            sys.exit(1)
        
        print("Creando tablas de base de datos...")
        try:
            Base.metadata.create_all(bind=engine)
            print("[OK] Tablas creadas correctamente")
            
            # Mostrar tablas creadas
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            print(f"\nTablas creadas ({len(tables)}):")
            for table in tables:
                print(f"  - {table}")
        except Exception as db_error:
            print(f"[FAIL] Error conectando a la base de datos: {db_error}")
            print("\nVerifica que:")
            print("  1. PostgreSQL esté corriendo")
            print("  2. DATABASE_URL en .env sea correcta")
            print("  3. La base de datos 'cosmos_notetaker' exista")
            sys.exit(1)
            
    except Exception as e:
        print(f"[FAIL] Error creando tablas: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    create_tables()

