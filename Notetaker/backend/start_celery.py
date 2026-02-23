"""Script para iniciar el worker de Celery."""
import subprocess
import sys
import os
from pathlib import Path

if __name__ == "__main__":
    # Cambiar al directorio del backend
    backend_dir = Path(__file__).parent.absolute()
    os.chdir(backend_dir)
    
    # Buscar el Python del entorno virtual
    venv_python = backend_dir / "notetaker" / "Scripts" / "python.exe"
    
    if not venv_python.exists():
        # Si no existe, intentar con el Python actual
        python_exe = sys.executable
        print(f"⚠️ No se encontró entorno virtual en {venv_python}")
        print(f"Usando Python: {python_exe}")
    else:
        python_exe = str(venv_python)
        print(f"✅ Usando Python del entorno virtual: {python_exe}")
    
    # Ejecutar Celery worker
    # Escuchar ambas colas: celery (tareas generales) y summary_queue (procesamiento de IA)
    cmd = [
        python_exe,
        "-m",
        "celery",
        "-A",
        "app.celery_app",
        "worker",
        "--loglevel=info",
        "--pool=solo",  # Usar solo para Windows (evita problemas con fork)
        "-Q", "celery,summary_queue",  # Escuchar ambas colas
        "-c", "1",  # Concurrencia 1 para no saturar la IA (una tarea a la vez)
        "--prefetch-multiplier=1",  # Solo tomar una tarea a la vez
    ]
    
    # Solo mostrar mensajes si no estamos ejecutando como servicio
    # (NSSM captura stdout/stderr, así que estos mensajes van a los logs)
    import sys
    is_service = os.environ.get('NSSM_SERVICE') or '--service' in sys.argv
    
    if not is_service:
        print("🚀 Iniciando Celery worker...")
        print(f"Comando: {' '.join(cmd)}")
        print(f"Directorio de trabajo: {backend_dir}")
        print("\nPresiona Ctrl+C para detener el worker\n")
    
    try:
        subprocess.run(cmd, check=True, cwd=str(backend_dir))
    except KeyboardInterrupt:
        print("\n\n✅ Celery worker detenido")
    except Exception as e:
        print(f"\n❌ Error iniciando Celery worker: {e}")
        print("\n💡 Asegúrate de que:")
        print("   1. Celery está instalado: pip install celery==5.3.4")
        print("   2. Redis está corriendo")
        print("   3. Estás usando el entorno virtual correcto")
        sys.exit(1)

